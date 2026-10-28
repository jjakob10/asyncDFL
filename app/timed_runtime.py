"""Opt-in time-budgeted M0 runtime; legacy round-based clients stay unchanged."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from app.federation import StateDict, clone_state_dict, deserialize_state_dict, fedavg, serialize_state_dict
from app.run_control import BudgetExpired, RunControl
from app.runtime import ClientRuntime, _utc_now


class ModelUnavailable(Exception):
    pass


class Jsonl:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def append(self, value: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")
            stream.flush()


def peak_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


class TimedClientRuntime(ClientRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for capability in ("train_local_controlled", "evaluate_detailed"):
            if not callable(getattr(self.trainer, capability, None)):
                raise ValueError(f"timed training requires trainer capability {capability}")
        root = Path(self.settings.results_dir)
        prefix = f"client_{self.settings.client_id}"
        self.events = Jsonl(root / f"{prefix}_events.jsonl")
        self.evaluations = Jsonl(root / f"{prefix}_evaluations.jsonl")
        self.control: RunControl | None = None
        self._io = {"model_bytes_sent": 0, "model_bytes_received": 0,
                    "fetch_failures": 0, "expected_unavailable": 0,
                    "communication_seconds": 0.0, "evaluation_seconds": 0.0}
        self._stop_observers = threading.Event()
        self._observer_error: Exception | None = None
        self._first_mix = False
        self._returned_snapshot = None
        self._published_steps = 0

    def start(self, run_id: str, start_at: float | None = None) -> str:
        with self._lock:
            if self._run_id is None:
                if start_at is None or not time.time() < start_at < time.time() + 3600:
                    raise ValueError("timed runs require start_at within the next hour")
                start = time.monotonic() + start_at - time.time()
                self.control = RunControl(
                    start, self.settings.run_duration_seconds,
                    self.settings.client_id in self.settings.offline_client_ids,
                    self.settings.offline_start_fraction, self.settings.offline_end_fraction,
                    emit=self._event,
                )
            return super().start(run_id, start_at)

    def _event(self, event: str, **extra) -> None:
        with self._lock:
            version = self._model_version
            snapshot = clone_state_dict(self._model) if event in {"offline", "returned"} else None
            if event == "returned":
                self._returned_snapshot = (clone_state_dict(self._model), version)
            record = {"time": _utc_now(), "run_id": self._run_id,
                      "client_id": self.settings.client_id, "event": event,
                      "elapsed_seconds": self.control.elapsed if self.control else 0,
                      "model_version": version, **extra}
        if snapshot is not None:
            payload = serialize_state_dict(snapshot)
            name = f"client_{self.settings.client_id}_{event}.pt"
            (Path(self.settings.results_dir) / name).write_bytes(payload)
            record.update(snapshot=name, snapshot_sha256=hashlib.sha256(payload).hexdigest())
        self.events.append(record)

    def model_snapshot(self) -> tuple[bytes, dict[str, str]]:
        if self.control is None or not self.control.online:
            raise ModelUnavailable()
        payload, headers = super().model_snapshot()
        if not self.control.online:
            self._event("response_crossed_offline")
            raise ModelUnavailable()
        with self._lock:
            self._io["model_bytes_sent"] += len(payload)
        return payload, headers

    def status(self) -> dict[str, Any]:
        value = super().status()
        value.update(availability="online" if self.control and self.control.online else "offline",
                     elapsed_seconds=self.control.elapsed if self.control else None,
                     run_duration_seconds=self.settings.run_duration_seconds,
                     total_rounds=None)
        return value

    def _fetch_neighbor(self, neighbor: str, reference: StateDict) -> tuple[StateDict, int, dict] | None:
        assert self.control is not None
        try:
            for attempt in range(self.settings.request_retries):
                timeout = self.control.request_timeout(self.settings.request_timeout)
                before = self.control.elapsed
                try:
                    began = time.monotonic()
                    try:
                        response = httpx.get(f"{neighbor}/model", timeout=timeout)
                    finally:
                        with self._lock:
                            self._io["communication_seconds"] += time.monotonic() - began
                        if self.control.offline and before < self.control.leave <= self.control.elapsed:
                            self._event("request_crossed_offline", target=neighbor)
                    if response.status_code == 503 and response.json().get("detail") == "client_offline":
                        with self._lock:
                            self._io["expected_unavailable"] += 1
                        return None
                    response.raise_for_status()
                    with self._lock:
                        self._io["model_bytes_received"] += len(response.content)
                    self.control.checkpoint()
                    state = deserialize_state_dict(response.content, reference)
                    count = int(response.headers["X-Sample-Count"])
                    if count <= 0:
                        raise ValueError("Johann M0 requires positive sample counts")
                    metadata = {"url": neighbor, "client_id": int(response.headers["X-Client-ID"]),
                                "model_version": int(response.headers["X-Model-Version"]),
                                "completed_round": int(response.headers["X-Completed-Round"]),
                                "sample_count": count, "aggregation_weight": count}
                    return state, count, metadata
                except httpx.HTTPError:
                    if attempt + 1 == self.settings.request_retries:
                        raise
                    until = time.monotonic() + self.settings.retry_delay
                    while time.monotonic() < until:
                        self.control.checkpoint()
                        time.sleep(min(0.02, max(0, until - time.monotonic())))
        except BudgetExpired:
            raise
        except Exception as exc:
            with self._lock:
                self._io["fetch_failures"] += 1
            self._record_error("model fetch", neighbor, str(exc))
        return None

    def _evaluate_snapshot(self, state, version, kind="periodic", **extra):
        assert self.control is not None
        captured = extra.pop("captured_seconds", self.control.elapsed)
        began = time.monotonic()
        metrics = self.trainer.evaluate_detailed(state, self.data.test_loader)
        seconds = time.monotonic() - began
        with self._lock:
            self._io["evaluation_seconds"] += seconds
        self.evaluations.append({
            "time": _utc_now(), "run_id": self._run_id, "client_id": self.settings.client_id,
            "model_version": version, "kind": kind, "captured_seconds": captured,
            "evaluation_seconds": seconds, "evaluation_split": self.settings.evaluation_split,
            "test": metrics, **extra,
        })
        return metrics

    def _observe_events(self):
        try:
            while not self._stop_observers.wait(0.01):
                self.control.transitions()
        except Exception as exc:
            self._observer_error = exc

    def _observe_models(self):
        try:
            interval = self.settings.evaluation_interval_seconds
            next_at = 0.0
            cached = None
            while not self._stop_observers.is_set():
                elapsed = self.control.elapsed
                if elapsed < next_at:
                    self._stop_observers.wait(min(0.05, next_at - elapsed))
                    continue
                captured = self.control.elapsed
                with self._lock:
                    state, version = clone_state_dict(self._model), self._model_version
                if self.control.online or cached is None or version != cached[0]:
                    metrics = self._evaluate_snapshot(state, version, captured_seconds=captured,
                                                      scheduled_seconds=next_at)
                    cached = version, metrics
                else:
                    self.evaluations.append({
                        "time": _utc_now(), "run_id": self._run_id,
                        "client_id": self.settings.client_id, "model_version": version,
                        "kind": "periodic", "captured_seconds": captured,
                        "scheduled_seconds": next_at, "evaluation_seconds": 0,
                        "evaluation_split": self.settings.evaluation_split,
                        "frozen": True, "test": cached[1],
                    })
                if captured >= self.control.duration:
                    return
                next_at += interval
                now = self.control.elapsed
                if now > next_at:
                    missed = int((now - next_at) // interval) + 1
                    self._event("evaluations_missed", count=missed, first_scheduled_seconds=next_at)
                    next_at += missed * interval
                next_at = min(next_at, self.control.duration)
        except Exception as exc:
            self._observer_error = exc

    def _execute(self) -> None:
        observers = []
        finished_state = "failed"
        try:
            self._save_class_distribution()
            self._forward_start()
            self.control.checkpoint()
            observers = [threading.Thread(target=f, daemon=True)
                         for f in (self._observe_events, self._observe_models)]
            for observer in observers:
                observer.start()
            try:
                while True:
                    self.control.checkpoint()
                    if self._observer_error:
                        raise self._observer_error
                    with self._lock:
                        base = clone_state_dict(self._model)
                    steps_before = self.control.steps
                    local, train = self.trainer.train_local_controlled(
                        base, self.data.train_loader, self.settings, self.control)
                    self.control.checkpoint()
                    weighted = [(local, self.data.sample_count)]
                    neighbors = []
                    for neighbor in self.settings.neighbors:
                        fetched = self._fetch_neighbor(neighbor, local)
                        if fetched is not None:
                            weighted.append((fetched[0], fetched[1]))
                            neighbors.append(fetched[2])
                    self.control.checkpoint()
                    first_mix = (self.control.offline and self.control.elapsed >= self.control.return_at
                                 and not self._first_mix)
                    if first_mix:
                        self._evaluate_snapshot(local, self._model_version,
                                                "before_first_mix", published=False)
                    aggregated = fedavg(weighted)
                    # Copy before the publication lock; commit only while still online.
                    published = clone_state_dict(aggregated)
                    while True:
                        self.control.checkpoint()
                        if self.control.offline and self.control.elapsed >= self.control.return_at and not self._first_mix and not first_mix:
                            first_mix = True
                            self._evaluate_snapshot(local, self._model_version,
                                                    "before_first_mix", published=False)
                        with self._lock:
                            if not self.control.online:
                                continue
                            self._model = published
                            self._model_version += 1
                            self._completed_round += 1
                            version = self._model_version
                            record = {"time": _utc_now(), "started_at": self._started_at,
                                      "run_id": self._run_id, "client_id": self.settings.client_id,
                                      "experiment_id": self.settings.experiment_id,
                                      "round": version, "model_version": version,
                                      "elapsed_seconds": self.control.elapsed,
                                      "participants": len(weighted), "sample_count": self.data.sample_count,
                                      "aggregation_weight": self.data.sample_count,
                                      "neighbor_versions": neighbors, "train": train, "test": {},
                                      "steps_in_cycle": self.control.steps - steps_before,
                                      **self.control.counters()}
                            self._latest_metrics = record
                            self._published_steps = self.control.steps
                            break
                    self.metrics_writer.append(record)
                    if first_mix:
                        self._first_mix = True
                        self._event("first_mix_after_return")
                        self._evaluate_snapshot(aggregated, version, "after_first_mix")
            except BudgetExpired:
                self._event("partial_cycle_discarded", unpublished_steps=(
                    self.control.steps - self._published_steps))
            self.control.transitions()
            # Capture the final published snapshot at the budget boundary, before serialization.
            with self._lock:
                final_state, final_version = clone_state_dict(self._model), self._model_version
                captured = self.control.elapsed
            self._stop_observers.set()
            for observer in observers:
                observer.join()
            if self._observer_error:
                raise self._observer_error
            self._evaluate_snapshot(final_state, final_version, "final", captured_seconds=captured)
            if self._returned_snapshot is not None:
                state, version = self._returned_snapshot
                self._evaluate_snapshot(state, version, "preserved_at_return",
                                        captured_seconds=self.control.return_at)
            self._save_final_model()
            began = time.monotonic()
            self._save_final_confusion()
            with self._lock:
                self._io["evaluation_seconds"] += time.monotonic() - began
            finished_state = "completed"
        except Exception as exc:
            self._record_error("experiment", self.settings.client_name, str(exc))
        finally:
            self._stop_observers.set()
            for observer in observers:
                observer.join()
            finished_at = _utc_now()
            with self._lock:
                cost = {**self._io, **(self.control.counters() if self.control else {}),
                        "completed_cycles": self._model_version,
                        "unpublished_optimizer_steps": self.control.steps - self._published_steps if self.control else 0,
                        "peak_process_memory_bytes": peak_memory_bytes(),
                        "run_id": self._run_id, "client_id": self.settings.client_id,
                        "state": finished_state, "finished_at": finished_at}
            destination = Path(self.settings.results_dir) / f"client_{self.settings.client_id}_cost.json"
            try:
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(json.dumps(cost, allow_nan=False, indent=2), encoding="utf-8")
                temporary.replace(destination)
            except OSError as exc:
                finished_state = "failed"
                self._record_error("cost artifact", str(destination), str(exc))
            with self._lock:
                self._state, self._finished_at = finished_state, finished_at
