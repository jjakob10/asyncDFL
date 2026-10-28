import json

import pytest

from experiments.johann.catalog import load_spec
from scripts import run_johann_experiments as runner


def test_prepare_refuses_overwrite_and_records_reproducibility(tmp_path):
    spec = load_spec("mnist", "concentrated-r")
    directory, resolved = runner.prepare(spec, "test", tmp_path)
    assert directory == tmp_path / "mnist" / "concentrated-r" / "test-s7"
    assert resolved["code_revision"]
    assert resolved["source_sha256"]["app/runtime.py"]
    compose = json.loads((directory / "compose.json").read_text())
    assert compose["services"]["client-0"]["environment"]["RUN_ID"] == "test-s7"
    with pytest.raises(FileExistsError):
        runner.prepare(spec, "test", tmp_path)
    for bad in ("..", "../outside", "name/child", "C:\\escape"):
        with pytest.raises(ValueError):
            runner.prepare(spec, bad, tmp_path)


def test_schedule_requires_all_acknowledgements_before_start(monkeypatch):
    calls = []

    class Response:
        def raise_for_status(self):
            pass

    def post(url, json, **kwargs):
        calls.append(json)
        return Response()

    monkeypatch.setattr(runner.httpx, "post", post)
    monkeypatch.setattr(runner, "api_statuses", lambda: [
        {"run_id": "test", "start_at": calls[0]["start_at"], "state": "running"} for _ in range(8)])
    assert runner.schedule_run("test") == calls[0]["start_at"]
    assert len(calls) == 8 and all(c == calls[0] for c in calls)
    monkeypatch.setattr(runner, "api_statuses", lambda: [{"run_id": "wrong"}] * 8)
    with pytest.raises(RuntimeError, match="acknowledged"):
        runner.schedule_run("test")


@pytest.mark.parametrize("command,expected_split", [("smoke", "validation"), ("pilot", "validation"), ("run", "test")])
def test_commands_prepare_requested_network_without_docker(command, expected_split, tmp_path, monkeypatch):
    calls = []

    def execute(directory, spec):
        assert spec["evaluation_split"] == expected_split
        assert spec["seed"] == 8
        assert directory.is_relative_to(tmp_path)
        calls.append(spec)
        return True

    monkeypatch.setattr(runner, "execute", execute)
    assert runner.main([command, "--dataset", "all", "--all", "--seeds", "8",
                        "--run-id", command, "--results-root", str(tmp_path)]) == 0
    assert len(calls) == 12
    assert all(s["mode"] == ("collection" if command == "run" else command) for s in calls)


def test_failure_collects_logs_before_cleanup(tmp_path, monkeypatch):
    import subprocess
    directory, spec = runner.prepare(load_spec("mnist", "concentrated-r"), "failure", tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "up" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.execute(directory, spec) is False
    assert json.loads((directory / "outcome.json").read_text())["state"] == "failed"
    assert next(i for i, c in enumerate(calls) if "logs" in c) < next(i for i, c in enumerate(calls) if "down" in c)
