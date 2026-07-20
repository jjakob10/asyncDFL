from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import torch

from app.config import Settings
from app.federation import (
    StateDict,
    clone_state_dict,
    deserialize_state_dict,
    fedavg,
    serialize_state_dict,
)
from app.metrics import MetricsWriter
from app.training import DataBundle, Trainer, load_trainer, seed_everything

LOGGER = logging.getLogger("dfl.client")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClientRuntime:
    def __init__(
        self,
        settings: Settings,
        data: DataBundle | None = None,
        trainer: Trainer | None = None,
    ) -> None:
        self.settings = settings
        seed_everything(settings.seed)
        torch.set_num_threads(settings.torch_num_threads)
        self.trainer = trainer or load_trainer(settings.trainer_module)
        self.data = data or self.trainer.load_data(settings)
        self.metrics_writer = MetricsWriter(settings.results_dir, settings.client_id)
        self._lock = threading.RLock()
        self._model: StateDict = clone_state_dict(
            self.trainer.create_model().state_dict()
        )
        self._run_id: str | None = None
        self._state = "waiting"
        self._model_version = 0
        self._completed_round = 0
        self._started_at: str | None = None
        self._finished_at: str | None = None
        self._latest_metrics: dict[str, Any] | None = None
        self._errors: list[dict[str, str]] = []
        self._worker: threading.Thread | None = None

    def start(self, run_id: str) -> str:
        with self._lock:
            if self._run_id == run_id:
                return "already_started"
            if self._run_id is not None:
                return "conflict"
            self._run_id = run_id
            self._state = "running"
            self._started_at = _utc_now()
            self._worker = threading.Thread(
                target=self._execute,
                name=f"dfl-client-{self.settings.client_id}",
                daemon=True,
            )
            self._worker.start()
            return "started"

    def model_snapshot(self) -> tuple[bytes, dict[str, str]]:
        with self._lock:
            payload = serialize_state_dict(self._model)
            headers = {
                "X-Client-ID": str(self.settings.client_id),
                "X-Model-Version": str(self._model_version),
                "X-Completed-Round": str(self._completed_round),
                "X-Sample-Count": str(self.data.sample_count),
            }
        return payload, headers

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "client_id": self.settings.client_id,
                "client_name": self.settings.client_name,
                "is_head": self.settings.is_head,
                "state": self._state,
                "run_id": self._run_id,
                "current_round": self._completed_round,
                "total_rounds": self.settings.rounds,
                "model_version": self._model_version,
                "sample_count": self.data.sample_count,
                "neighbors": list(self.settings.neighbors),
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "latest_metrics": self._latest_metrics,
                "recent_errors": list(self._errors[-20:]),
            }

    def wait(self, timeout: float | None = None) -> None:
        worker = self._worker
        if worker:
            worker.join(timeout)

    def _record_error(self, operation: str, target: str, message: str) -> None:
        error = {
            "time": _utc_now(),
            "operation": operation,
            "target": target,
            "message": message,
        }
        with self._lock:
            self._errors.append(error)
        LOGGER.warning("%s failed for %s: %s", operation, target, message)

    def _request_with_retries(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.request_retries + 1):
            try:
                response = httpx.request(
                    method,
                    url,
                    timeout=self.settings.request_timeout,
                    **kwargs,
                )
                response.raise_for_status()
                return response
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < self.settings.request_retries:
                    time.sleep(self.settings.retry_delay)
        assert last_error is not None
        raise last_error

    def _forward_to_neighbor(self, neighbor: str) -> None:
        try:
            self._request_with_retries(
                "POST",
                f"{neighbor}/start",
                json={"run_id": self._run_id},
            )
        except Exception as exc:
            self._record_error("start propagation", neighbor, str(exc))

    def _forward_start(self) -> None:
        with ThreadPoolExecutor(max_workers=max(1, len(self.settings.neighbors))) as executor:
            list(executor.map(self._forward_to_neighbor, self.settings.neighbors))

    def _fetch_neighbor(
        self,
        neighbor: str,
        reference: StateDict,
    ) -> tuple[StateDict, int, dict[str, Any]] | None:
        try:
            response = self._request_with_retries("GET", f"{neighbor}/model")
            state = deserialize_state_dict(response.content, reference)
            sample_count = int(response.headers["X-Sample-Count"])
            if sample_count <= 0:
                raise ValueError("neighbor returned a non-positive sample count")
            metadata = {
                "url": neighbor,
                "client_id": int(response.headers["X-Client-ID"]),
                "model_version": int(response.headers["X-Model-Version"]),
                "completed_round": int(response.headers["X-Completed-Round"]),
                "sample_count": sample_count,
            }
            return state, sample_count, metadata
        except Exception as exc:
            self._record_error("model fetch", neighbor, str(exc))
            return None

    def _save_final_model(self) -> None:
        destination = Path(self.settings.results_dir) / f"client_{self.settings.client_id}_final.pt"
        temporary = destination.with_suffix(".tmp")
        with self._lock:
            state = clone_state_dict(self._model)
        torch.save(state, temporary)
        os.replace(temporary, destination)

    def _execute(self) -> None:
        try:
            self._forward_start()
            for round_number in range(1, self.settings.rounds + 1):
                with self._lock:
                    base_state = clone_state_dict(self._model)
                    error_cursor = len(self._errors)
                local_state, train_metrics = self.trainer.train_local(
                    base_state,
                    self.data.train_loader,
                    self.settings,
                )

                weighted_states: list[tuple[StateDict, int]] = [
                    (local_state, self.data.sample_count)
                ]
                neighbor_versions: list[dict[str, Any]] = []
                for neighbor in self.settings.neighbors:
                    fetched = self._fetch_neighbor(neighbor, local_state)
                    if fetched is not None:
                        state, sample_count, metadata = fetched
                        weighted_states.append((state, sample_count))
                        neighbor_versions.append(metadata)

                aggregated = fedavg(weighted_states)
                test_metrics = self.trainer.evaluate(
                    aggregated, self.data.test_loader
                )
                with self._lock:
                    round_errors = list(self._errors[error_cursor:])
                record = {
                    "time": _utc_now(),
                    "run_id": self._run_id,
                    "client_id": self.settings.client_id,
                    "round": round_number,
                    "model_version": round_number,
                    "participants": len(weighted_states),
                    "neighbor_versions": neighbor_versions,
                    "neighbor_errors": round_errors,
                    "train": train_metrics,
                    "test": test_metrics,
                }
                with self._lock:
                    self._model = clone_state_dict(aggregated)
                    self._model_version += 1
                    self._completed_round = round_number
                    record["model_version"] = self._model_version
                    self._latest_metrics = record
                self.metrics_writer.append(record)
                LOGGER.info(
                    "client=%s round=%s/%s train_loss=%.4f test_accuracy=%.4f peers=%s",
                    self.settings.client_id,
                    round_number,
                    self.settings.rounds,
                    train_metrics["loss"],
                    test_metrics["accuracy"],
                    len(weighted_states) - 1,
                )

            self._save_final_model()
            with self._lock:
                self._state = "completed"
                self._finished_at = _utc_now()
        except Exception as exc:
            LOGGER.exception("client %s experiment failed", self.settings.client_id)
            self._record_error("experiment", self.settings.client_name, str(exc))
            with self._lock:
                self._state = "failed"
                self._finished_at = _utc_now()
