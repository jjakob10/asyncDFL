"""Eight real HTTP processes for functional smokes when Docker is unavailable.

This backend is deliberately restricted to smoke/validation, not scientific
collection. It preserves capped data, the common schedule and runtime methods.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx


def execute_local_smoke(directory: Path, spec: dict, cache: Path) -> bool:
    from app.config import Settings
    from experiments.johann.trainer import load_data
    from scripts.run_johann_experiments import ROOT, api_statuses, json_write, schedule_run

    if spec["mode"] != "smoke" or spec["evaluation_split"] != "validation":
        raise ValueError("local execution is restricted to functional validation smokes")
    outcome = {"state": "failed", "run_id": spec["run_id"], "execution_backend": "local_smoke"}
    processes, logs = [], []
    try:
        # Refuse to interfere with services already bound to the experimental ports.
        import socket
        for port in range(8000, 8008):
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"port {port} already in use")
        cache.mkdir(parents=True, exist_ok=True)
        # Download once, before spawning the eight clients; no concurrent extraction.
        preload = Settings(client_id=0, client_name="preload", client_count=8,
                           seed=spec["seed"], max_train_samples=800, max_test_samples=100,
                           data_dir=str(cache), download_data=True, evaluation_split="validation",
                           johann_distribution=spec["scenario"].rsplit("-", 1)[0])
        load_data(preload, spec["dataset"])
        compose = json.loads((directory / "compose.json").read_text(encoding="utf-8"))
        for client in range(8):
            environment = os.environ.copy()
            environment.update({key: str(value) for key, value in compose["services"][f"client-{client}"]["environment"].items()})
            adjacent = sorted(b if a == client else a for a, b in spec["edges"] if client in (a, b))
            environment.update(DATA_DIR=str(cache), RESULTS_DIR=str(directory), DOWNLOAD_DATA="false",
                               API_PORT=str(8000 + client), PYTHONUNBUFFERED="1",
                               NEIGHBORS=",".join(f"http://127.0.0.1:{8000 + peer}" for peer in adjacent))
            log = (directory / f"local_client_{client}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(8000 + client)],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))
        deadline = time.monotonic() + 300
        while True:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("a local client exited; inspect local_client_*.log")
            try:
                statuses = api_statuses()
                if all(s["state"] == "waiting" for s in statuses):
                    break
                if any(s["state"] == "failed" for s in statuses):
                    raise RuntimeError("a local client failed before scheduling")
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("local clients did not become ready within five minutes")
            time.sleep(.5)
        dependencies = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (directory / "container_dependencies.txt").write_text(dependencies.stdout, encoding="utf-8")
        start_at = schedule_run(spec["run_id"])
        spec["start_at"] = start_at
        json_write(directory / "resolved_experiment.json", spec)
        json_write(directory / "schedule.json", {"start_at": start_at, "execution_backend": "local_smoke",
                   "offline_at": start_at + spec["duration_seconds"] * spec["offline_start_fraction"],
                   "return_at": start_at + spec["duration_seconds"] * spec["offline_end_fraction"],
                   "end_at": start_at + spec["duration_seconds"], "offline_client_ids": spec["offline_client_ids"]})
        deadline = time.monotonic() + spec["duration_seconds"] + 310
        last_report = 0
        while time.monotonic() < deadline:
            if any(process.poll() is not None for process in processes):
                raise RuntimeError("a local client exited during the run")
            statuses = api_statuses()
            json_write(directory / "last_status.json", statuses)
            if any(s["state"] == "failed" for s in statuses):
                raise RuntimeError("a local client failed; inspect last_status.json")
            if all(s["state"] == "completed" for s in statuses):
                from experiments.johann.analysis import summarize_run
                summarize_run(directory)
                outcome["state"] = "completed"
                return True
            if time.monotonic() - last_report >= 15:
                print(f"local smoke {spec['dataset']}/{spec['method']}: " + " | ".join(
                    f"{s['client_id']}:{s['availability']}/{s['reintegration_phase']}" for s in statuses), flush=True)
                last_report = time.monotonic()
            time.sleep(.5)
        raise TimeoutError("local smoke exceeded completion grace period")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, httpx.HTTPError) as exc:
        outcome["error"] = str(exc)
        print(f"error: {exc}", file=sys.stderr)
        return False
    finally:
        outcome["finished_at_unix"] = time.time()
        json_write(directory / "outcome.json", outcome)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
