import json
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Settings
from app.federation import clone_state_dict
from app.main import create_app
from app.run_control import BudgetExpired, RunControl
from app.timed_runtime import ModelUnavailable, TimedClientRuntime
from app.training import DataBundle
from experiments.johann.trainer import metrics_from_counts, train_controlled


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_clock_pauses_and_resumes_once_at_batch_boundary():
    clock, events = FakeClock(), []
    control = RunControl(0, 10, True, clock=clock, sleep=clock.sleep,
                         emit=lambda name, **kw: events.append((name, clock.now, kw)))
    control.checkpoint()
    clock.now = 3.2
    control.checkpoint()
    assert clock.now >= 6
    assert [name for name, _, _ in events] == ["started", "offline", "training_paused", "returned", "training_resumed"]
    control.checkpoint()
    clock.now = 10
    with pytest.raises(BudgetExpired):
        control.checkpoint()
    assert sum(name == "returned" for name, _, _ in events) == 1


def test_controlled_training_preserves_model_and_optimizer_during_pause(monkeypatch):
    import experiments.johann.trainer as shared

    class Base:
        @staticmethod
        def create_model():
            return nn.Linear(1, 10)

    monkeypatch.setattr(shared, "base_trainer", lambda _: Base)
    initial = Base.create_model().state_dict()
    loader = DataLoader(TensorDataset(torch.ones(6, 1), torch.arange(6)), batch_size=1)
    settings = Settings(client_id=0, client_name="client-0", batch_size=1)
    states = []
    for offline in (False, True):
        clock = FakeClock()
        control = RunControl(0, 100, offline, clock=clock, sleep=clock.sleep)
        original = control.batch_finished

        def batch(examples, seconds):
            original(examples, seconds)
            clock.now += 11

        control.batch_finished = batch
        state, _ = train_controlled(initial, loader, settings, control, "mnist")
        states.append(state)
        assert control.steps == 6 and control.examples == 6
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])


class TinyTrainer:
    def create_model(self):
        return nn.Linear(1, 10)

    def train_local_controlled(self, base, loader, settings, control):
        state = clone_state_dict(base)
        for _ in range(2):
            control.checkpoint()
            time.sleep(.008)
            for tensor in state.values():
                tensor.add_(.001)
            control.batch_finished(1, .008)
        return state, {"loss": 1.0, "accuracy": .1}

    def evaluate_detailed(self, state, loader):
        return metrics_from_counts(np_eye(), 1.0)

    def evaluate_confusion(self, state, loader):
        return {"classification_type": "multiclass", "labels": [str(i) for i in range(10)],
                "sample_count": 10, "counts": np_eye().tolist()}


def np_eye():
    import numpy as np
    return np.eye(10, dtype=int)


def runtime_settings(tmp_path, client=0, duration=1):
    return Settings(client_id=client, client_name=f"client-{client}", rounds=1,
                    run_duration_seconds=duration, evaluation_interval_seconds=.05,
                    offline_client_ids=(0, 4), auto_start_head=False, results_dir=str(tmp_path))


def bundle():
    loader = DataLoader(TensorDataset(torch.zeros(10, 1), torch.arange(10)), batch_size=1)
    return DataBundle(loader, loader, 10)


def test_offline_api_status_health_and_shared_start_conflicts(tmp_path):
    settings = runtime_settings(tmp_path)
    runtime = TimedClientRuntime(settings, bundle(), TinyTrainer())
    clock = FakeClock()
    runtime.control = RunControl(0, 10, True, clock=clock, sleep=clock.sleep)
    with TestClient(create_app(settings, runtime, auto_start_head=False)) as client:
        clock.now = 2
        response = client.get("/model")
        assert response.status_code == 200
        assert response.headers["X-Model-Version"] == "0"
        clock.now = 3
        assert client.get("/model").status_code == 503
        assert client.get("/health").status_code == 200
        assert client.get("/status").json()["availability"] == "offline"
        assert client.post("/start", json={"run_id": "test"}).status_code == 422
        runtime._execute = lambda: None
        at = time.time() + 30
        assert client.post("/start", json={"run_id": "test", "start_at": at}).status_code == 202
        assert client.post("/start", json={"run_id": "test", "start_at": at}).json()["result"] == "already_started"
        assert client.post("/start", json={"run_id": "test", "start_at": at + 1}).status_code == 409
        assert client.post("/start", json={"run_id": "other", "start_at": at}).status_code == 409


def test_expected_503_has_no_retry_and_no_error(tmp_path, monkeypatch):
    settings = runtime_settings(tmp_path)
    runtime = TimedClientRuntime(settings, bundle(), TinyTrainer())
    clock = FakeClock()
    clock.now = 2
    runtime.control = RunControl(0, 10, False, clock=clock)
    calls = []

    def offline(url, **kwargs):
        calls.append(url)
        return httpx.Response(503, json={"detail": "client_offline"}, request=httpx.Request("GET", url))

    monkeypatch.setattr("app.timed_runtime.httpx.get", offline)
    assert runtime._fetch_neighbor("http://neighbor", runtime._model) is None
    assert len(calls) == 1
    assert runtime._io["expected_unavailable"] == 1
    assert runtime._io["model_bytes_received"] == 0
    assert not runtime.status()["recent_errors"]


def test_completion_is_published_after_cost_artifact(tmp_path, monkeypatch):
    settings = runtime_settings(tmp_path, client=1, duration=.12)
    runtime = TimedClientRuntime(settings, bundle(), TinyTrainer())
    original_replace = Path.replace
    observed_states = []

    def replace_file(path, target):
        if path.name == "client_1_cost.tmp":
            observed_states.append(runtime.status()["state"])
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_file)
    runtime.start("cost-order", time.time() + .05)
    runtime.wait(5)
    assert observed_states == ["running"]
    assert runtime.status()["state"] == "completed"
    assert json.loads((tmp_path / "client_1_cost.json").read_text())["state"] == "completed"


def test_cost_write_failure_is_reported_as_failed(tmp_path, monkeypatch):
    settings = runtime_settings(tmp_path, client=1, duration=.1)
    runtime = TimedClientRuntime(settings, bundle(), TinyTrainer())
    original_replace = Path.replace

    def replace_file(path, target):
        if path.name == "client_1_cost.tmp":
            raise OSError("simulated write failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_file)
    runtime.start("cost-failure", time.time() + .05)
    runtime.wait(5)
    assert runtime.status()["state"] == "failed"
    assert runtime.status()["recent_errors"][-1]["operation"] == "cost artifact"


def test_eight_clients_continue_through_return_and_stop_by_time(tmp_path, monkeypatch):
    runtimes = {}
    for client in range(8):
        neighbors = ((client - 1) % 8, (client + 1) % 8, (client + 4) % 8)
        folder = tmp_path / str(client)
        settings = replace(runtime_settings(folder, client, 1.5),
                           neighbors=tuple(f"http://client-{n}" for n in neighbors))
        runtime = TimedClientRuntime(settings, bundle(), TinyTrainer())
        monkeypatch.setattr(runtime, "_forward_start", lambda: None)
        runtimes[f"http://client-{client}"] = runtime

    def get(url, **kwargs):
        runtime = runtimes[url.removesuffix("/model")]
        try:
            payload, headers = runtime.model_snapshot()
            return httpx.Response(200, content=payload, headers=headers, request=httpx.Request("GET", url))
        except ModelUnavailable:
            return httpx.Response(503, json={"detail": "client_offline"}, request=httpx.Request("GET", url))

    monkeypatch.setattr("app.timed_runtime.httpx.get", get)
    start_at = time.time() + .25
    for runtime in runtimes.values():
        runtime.start("network", start_at)
    for runtime in runtimes.values():
        runtime.wait(10)
    for runtime in runtimes.values():
        status = runtime.status()
        assert status["state"] == "completed", status["recent_errors"]
        assert status["model_version"] > 1  # legacy ROUNDS=1 cannot stop a timed client
        root = Path(runtime.settings.results_dir)
        rows = [json.loads(line) for line in (root / f"client_{runtime.settings.client_id}_metrics.jsonl").read_text().splitlines()]
        assert rows[-1]["elapsed_seconds"] > .9
        assert rows[-1]["elapsed_seconds"] < 1.6
        events = [json.loads(line) for line in runtime.events.path.read_text().splitlines()]
        if runtime.settings.client_id in {0, 4}:
            assert sum(e["event"] == "returned" for e in events) == 1
            assert sum(e["event"] == "first_mix_after_return" for e in events) == 1
            offline = torch.load(root / f"client_{runtime.settings.client_id}_offline.pt", weights_only=True)
            returned = torch.load(root / f"client_{runtime.settings.client_id}_returned.pt", weights_only=True)
            assert all(torch.equal(offline[k], returned[k]) for k in offline)
        else:
            assert any(.5 < row["elapsed_seconds"] < .85 for row in rows)
    assert sum(r._io["expected_unavailable"] for r in runtimes.values()) > 0
