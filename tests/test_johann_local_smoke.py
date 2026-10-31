import json

import pytest

from experiments.johann.catalog import compose_for, load_spec
from scripts.johann_local_smoke import execute_local_smoke


def test_local_backend_refuses_scientific_collection(tmp_path):
    with pytest.raises(ValueError, match="restricted"):
        execute_local_smoke(tmp_path, {"mode": "collection", "evaluation_split": "test"}, tmp_path / "data")
    assert not list(tmp_path.iterdir())


def test_local_smoke_does_not_stop_existing_services(tmp_path, monkeypatch):
    import socket

    class Occupied:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            return 0

    monkeypatch.setattr(socket, "socket", lambda: Occupied())
    spec = load_spec("mnist", "concentrated-r", mode="smoke")
    spec["run_id"] = "unit"
    assert execute_local_smoke(tmp_path, spec, tmp_path / "data") is False
    assert "already in use" in json.loads((tmp_path / "outcome.json").read_text())["error"]
    assert not (tmp_path / "data").exists()


def test_local_smoke_cleans_up_only_processes_it_created(tmp_path, monkeypatch):
    import socket
    from experiments.johann import trainer
    from scripts import johann_local_smoke

    class Available:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def connect_ex(self, address):
            return 1

    monkeypatch.setattr(socket, "socket", lambda: Available())
    monkeypatch.setattr(trainer, "load_data", lambda *args: None)
    events = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, **kwargs):
            events.append("wait")

    def spawn(*args, **kwargs):
        if events:
            raise OSError("simulated second process failure")
        events.append("spawn")
        return Process()

    monkeypatch.setattr(johann_local_smoke.subprocess, "Popen", spawn)
    spec = load_spec("mnist", "concentrated-r", mode="smoke")
    spec["run_id"] = "unit"
    (tmp_path / "compose.json").write_text(json.dumps(compose_for(spec, "unit", tmp_path)))
    assert execute_local_smoke(tmp_path, spec, tmp_path / "data") is False
    assert events == ["spawn", "terminate", "wait"]
