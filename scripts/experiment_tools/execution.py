"""Common rendering, monitoring and lifecycle for round-based experiments."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ROOT as REPOSITORY_ROOT
from .catalog import ExperimentSpec
from .compose import build_compose
from .reporting import summarize_run
from scripts.plot_metrics import load_experiments, plot_metrics

DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / "experiments"


@contextmanager
def compose_lifecycle(compose_path: Path, run_directory: Path):
    """Save logs and remove only this Compose project, even after a failed start."""
    base = ["docker", "compose", "-f", str(compose_path)]
    try:
        yield base
    finally:
        cleanup_compose(compose_path, run_directory)


def cleanup_compose(compose_path: Path, run_directory: Path) -> None:
    """Persist logs before tearing down the named project."""
    base = ["docker", "compose", "-f", str(compose_path)]
    try:
        logs = subprocess.run([*base, "logs", "--no-color"], cwd=REPOSITORY_ROOT,
                              capture_output=True, text=True, errors="replace")
        (run_directory / "container.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
    except OSError as exc:
        (run_directory / "container.log").write_text(str(exc), encoding="utf-8")
    finally:
        try:
            subprocess.run([*base, "down"], cwd=REPOSITORY_ROOT, check=False)
        except OSError:
            pass


def run_compose(compose_path: Path, experiment: ExperimentSpec, run_directory: Path,
                show_client_points: bool = False, *, build: bool = True,
                timeout: float = 86400, poll_interval: float = 5,
                charts: bool = False) -> int:
    """Run a round-based experiment with the shared monitoring and cleanup."""
    try:
        with compose_lifecycle(compose_path, run_directory) as base:
            command = [*base, "up", "-d", "--build" if build else "--no-build"]
            subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
            if not wait_for_completion(experiment.client_count, poll_interval, timeout):
                raise RuntimeError("one or more clients failed")
            summarize_run(run_directory, show_client_points)
            if charts:
                plot_metrics(load_experiments([run_directory]), run_directory / "charts",
                             "round", "both", show_neighbor_count=True)
            (run_directory / "outcome.json").write_text(
                json.dumps({"completed": True}), encoding="utf-8")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        (run_directory / "outcome.json").write_text(
            json.dumps({"completed": False, "error": str(exc)}), encoding="utf-8")
        print(f"error: {exc}")
        return 2


def default_run_id(experiment_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_id}-{timestamp}"


def resolved_spec(
    experiment: ExperimentSpec, run_id: str, smoke: bool
) -> dict[str, Any]:
    value = asdict(experiment)
    value["run_id"] = run_id
    value["smoke"] = smoke
    return value


def render_compose(
    experiment: ExperimentSpec,
    run_id: str,
    output_path: Path,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    smoke: bool = False,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_directory = (results_root / experiment.experiment_id / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    compose = build_compose(experiment, run_id, run_directory, smoke)
    output_path.write_text(
        json.dumps(compose, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spec_path = run_directory / "resolved_experiment.json"
    spec_path.write_text(
        json.dumps(resolved_spec(experiment, run_id, smoke), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, run_directory


def wait_for_completion(
    client_count: int,
    poll_interval: float = 5.0,
    timeout: float = 86_400.0,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses: list[dict[str, Any]] = []
        try:
            for client_id in range(client_count):
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{8000 + client_id}/status",
                    timeout=3,
                ) as response:
                    statuses.append(json.load(response))
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(poll_interval)
            continue
        print(
            " | ".join(
                f"{item['client_id']}:{item['state']} "
                f"{item['current_round']}/{item['total_rounds']}"
                for item in statuses
            ),
            flush=True,
        )
        states = {item["state"] for item in statuses}
        if "failed" in states:
            return False
        if states == {"completed"}:
            return True
        time.sleep(poll_interval)
    raise TimeoutError("experiment did not complete before the runner timeout")
