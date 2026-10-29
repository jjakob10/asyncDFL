import json
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runtime import ClientRuntime
from app.timed_runtime import TimedClientRuntime
from tests.test_runtime import TinyTrainer, tiny_data
from tests.test_johann_runtime import TinyTrainer as TimedTrainer, bundle


def wait_until(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.005)
    assert predicate()


def make_runtime(tmp_path, **kwargs):
    return ClientRuntime(Settings(client_id=0, client_name="client-0", rounds=1,
                                  request_retries=1, retry_delay=.005, request_timeout=.1,
                                  neighbors=("http://peer",), results_dir=str(tmp_path), **kwargs),
                         tiny_data(), TinyTrainer())


def response(url, code=202, **body):
    return httpx.Response(code, json=body, request=httpx.Request("POST", url))


def test_retries_survive_training_and_lost_acknowledgement(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    attempts = []
    online = threading.Event()
    accepted = False

    def send(method, url, **kwargs):
        nonlocal accepted
        attempts.append(kwargs["json"])
        if not online.is_set() or not accepted:
            accepted = online.is_set()
            raise httpx.ReadTimeout("lost acknowledgement")
        return response(url, result="already_started", run_id="run")

    monkeypatch.setattr("app.runtime.httpx.request", send)
    monkeypatch.setattr(runtime, "_fetch_neighbor", lambda *a: None)
    try:
        assert runtime.start("run") == "started"
        runtime.wait(3)
        assert runtime.status()["state"] == "completed"
        wait_until(lambda: len(attempts) > runtime.settings.request_retries + 2)
        assert runtime.start("run") == "already_started"
        online.set()
        wait_until(lambda: runtime.status()["start_propagation"]["http://peer"]["state"] == "confirmed")
        assert runtime.trainer.train_calls == 1
        assert not runtime.status()["recent_errors"]
        assert all(body == {"run_id": "run"} for body in attempts)
    finally:
        runtime.close()


@pytest.mark.parametrize("code", [408, 429, 500, 503])
def test_transient_status_retries_even_when_training_fails(tmp_path, monkeypatch, code):
    runtime = make_runtime(tmp_path)
    calls = []

    def send(method, url, **kwargs):
        calls.append(url)
        if len(calls) < 4:
            return response(url, code, detail="temporary")
        return response(url, result="started", run_id="run")

    def fail():
        raise OSError("disk error")

    monkeypatch.setattr("app.runtime.httpx.request", send)
    monkeypatch.setattr(runtime, "_save_class_distribution", fail)
    try:
        runtime.start("run")
        runtime.wait(3)
        assert runtime.status()["state"] == "failed"
        wait_until(lambda: runtime.status()["start_propagation"]["http://peer"]["state"] == "confirmed")
        assert len(calls) == 4
    finally:
        runtime.close()


@pytest.mark.parametrize("code,body,state", [
    (409, {"detail": "different run"}, "conflict"),
    (422, {"detail": "run_expired"}, "failed"),
    (401, {"detail": "unauthorized"}, "failed"),
    (202, {"result": "started", "run_id": "other"}, "failed"),
    (202, {"result": "unknown", "run_id": "run"}, "failed"),
    (202, [], "failed"),
])
def test_permanent_failure_stops_only_that_neighbor(tmp_path, monkeypatch, code, body, state):
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(runtime, "_execute", lambda: None)
    monkeypatch.setattr("app.runtime.httpx.request", lambda method, url, **kw:
                        httpx.Response(code, json=body, request=httpx.Request(method, url)))
    try:
        runtime.start("run")
        wait_until(lambda: runtime.status()["start_propagation"]["http://peer"]["state"] == state)
        assert runtime.status()["start_propagation"]["http://peer"]["attempts"] == 1
        assert len(runtime.status()["recent_errors"]) == 1
    finally:
        runtime.close()


def test_shutdown_cancels_pending_retry(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path)
    monkeypatch.setattr(runtime, "_execute", lambda: None)

    def fail(*args, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("app.runtime.httpx.request", fail)
    runtime.start("run")
    wait_until(lambda: runtime.status()["start_propagation"]["http://peer"]["state"] == "retrying")
    runtime.close()
    assert all(not thread.is_alive() for thread in runtime._propagation_threads)
    assert runtime.status()["start_propagation"]["http://peer"]["state"] == "cancelled"


def test_one_offline_neighbor_does_not_block_another(tmp_path, monkeypatch):
    runtime = ClientRuntime(Settings(client_id=0, client_name="client-0", rounds=1,
                                    neighbors=("http://offline", "http://online"),
                                    retry_delay=.005, results_dir=str(tmp_path)),
                            tiny_data(), TinyTrainer())

    def send(method, url, **kwargs):
        if "offline" in url:
            raise httpx.ConnectError("unavailable")
        return response(url, result="started", run_id="run")

    monkeypatch.setattr("app.runtime.httpx.request", send)
    monkeypatch.setattr(runtime, "_fetch_neighbor", lambda *a: None)
    try:
        runtime.start("run")
        runtime.wait(3)
        wait_until(lambda: runtime.status()["start_propagation"]["http://online"]["state"] == "confirmed")
        assert runtime.status()["state"] == "completed"
        assert runtime.status()["start_propagation"]["http://offline"]["state"] == "retrying"
        topology = json.loads((tmp_path / "client_0_topology.json").read_text())
        assert topology["neighbors"] == ["http://offline", "http://online"]
        assert topology["run_id"] == "run"
    finally:
        runtime.close()


def test_timed_late_start_keeps_window_and_expired_start_is_explicit(tmp_path, monkeypatch):
    settings = Settings(client_id=0, client_name="client-0", run_duration_seconds=10,
                        results_dir=str(tmp_path))
    runtime = TimedClientRuntime(settings, bundle(), TimedTrainer())
    monkeypatch.setattr(runtime, "_execute", lambda: None)
    at = time.time() - 4
    with TestClient(create_app(settings, runtime, auto_start_head=False)) as client:
        assert client.post("/start", json={"run_id": "late", "start_at": at}).status_code == 202
        assert runtime.control.elapsed == pytest.approx(4, abs=.5)
        assert runtime.control.duration == 10
        monkeypatch.setattr("app.timed_runtime.time.time", lambda: at + 20)
        assert client.post("/start", json={"run_id": "late", "start_at": at}).json()["result"] == "already_started"
    expired = TimedClientRuntime(settings, bundle(), TimedTrainer())
    with TestClient(create_app(settings, expired, auto_start_head=False)) as client:
        result = client.post("/start", json={"run_id": "expired", "start_at": at})
        assert result.status_code == 422
        assert "run_expired" in result.json()["detail"]
        assert expired.status()["run_id"] is None
