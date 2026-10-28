"""Run the time-budgeted Johann M0 experiment matrix sequentially."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.johann.catalog import SCENARIOS, compose_for, load_spec


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(spec, run_id, results_root):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", run_id):
        raise ValueError("run ID must be a safe path component of at most 91 characters")
    directory = (results_root / spec["dataset"] / spec["scenario"] / f"{run_id}-s{spec['seed']}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    spec = {**spec, "run_id": directory.name, "batch_run_id": run_id}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, capture_output=True)
    (directory / "working_tree.patch").write_bytes(diff.stdout)
    # Include hashes of untracked implementation files as well as tracked changes.
    import hashlib
    sources = {}
    for folder in ("app", "experiments", "scripts"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"}:
                sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec.update(code_revision=revision.stdout.strip(), working_tree_dirty=bool(dirty.stdout.strip()),
                source_sha256=sources, created_at=datetime.now(timezone.utc).isoformat(),
                host_dependencies={name: importlib.metadata.version(name)
                                   for name in ("torch", "torchvision", "numpy", "httpx", "matplotlib")})
    json_write(directory / "resolved_experiment.json", spec)
    json_write(directory / "compose.json", compose_for(spec, spec["run_id"], directory))
    return directory, spec


def api_statuses():
    def read(client):
        response = httpx.get(f"http://127.0.0.1:{8000 + client}/status", timeout=2)
        response.raise_for_status()
        return response.json()
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(read, range(8)))


def schedule_run(run_id, lead_seconds=10):
    start_at = time.time() + lead_seconds
    def schedule(client):
        response = httpx.post(f"http://127.0.0.1:{8000 + client}/start",
                              json={"run_id": run_id, "start_at": start_at}, timeout=3)
        response.raise_for_status()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(schedule, range(8)))
    statuses = api_statuses()
    if time.time() >= start_at or any(
        s.get("run_id") != run_id or s.get("start_at") != start_at or s.get("state") == "failed"
        for s in statuses
    ):
        raise RuntimeError("not all eight clients acknowledged before the shared start")
    return start_at


def execute(directory, spec):
    base = ["docker", "compose", "-f", str(directory / "compose.json")]
    outcome = {"state": "failed", "run_id": spec["run_id"]}
    try:
        with (directory / "docker_start.log").open("w", encoding="utf-8") as log:
            subprocess.run([*base, "up", "--build", "-d"], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        deadline = time.monotonic() + 300
        while True:
            try:
                statuses = api_statuses()
                if len(statuses) == 8 and all(s["state"] == "waiting" for s in statuses):
                    break
                if any(s["state"] == "failed" for s in statuses):
                    raise RuntimeError("a client failed before scheduling")
            except (httpx.HTTPError, KeyError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("clients were not ready within five minutes")
            time.sleep(1)
        inspect = subprocess.run(["docker", "image", "inspect", f"dfl-johann-{spec['dataset']}:latest"],
                                 capture_output=True, text=True, check=True)
        json_write(directory / "image.json", json.loads(inspect.stdout))
        spec["image_id"] = json.loads(inspect.stdout)[0]["Id"]
        dependencies = subprocess.run([*base, "exec", "-T", "client-0", "python", "-m", "pip", "freeze"],
                                      capture_output=True, text=True, check=True)
        (directory / "container_dependencies.txt").write_text(dependencies.stdout, encoding="utf-8")
        start_at = schedule_run(spec["run_id"])
        spec["start_at"] = start_at
        json_write(directory / "resolved_experiment.json", spec)
        json_write(directory / "schedule.json", {
            "start_at": start_at, "offline_at": start_at + spec["duration_seconds"] * spec["offline_start_fraction"],
            "return_at": start_at + spec["duration_seconds"] * spec["offline_end_fraction"],
            "end_at": start_at + spec["duration_seconds"], "offline_client_ids": spec["offline_client_ids"],
        })
        deadline = time.monotonic() + spec["duration_seconds"] + 310
        last_report = 0
        while time.monotonic() < deadline:
            statuses = api_statuses()
            json_write(directory / "last_status.json", statuses)
            if any(s["state"] == "failed" for s in statuses):
                raise RuntimeError("a client failed; inspect last_status.json and container.log")
            if all(s["state"] == "completed" for s in statuses):
                from experiments.johann.analysis import summarize_run
                summarize_run(directory)
                outcome["state"] = "completed"
                return True
            if time.monotonic() - last_report >= 15:
                print(f"{spec['dataset']} {spec['scenario']} seed={spec['seed']}: " + " | ".join(
                    f"{s['client_id']}:{s['availability']} cycles={s['current_round']}" for s in statuses), flush=True)
                last_report = time.monotonic()
            time.sleep(1)
        raise TimeoutError("run exceeded its completion grace period")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, httpx.HTTPError) as exc:
        outcome["error"] = str(exc)
        print(f"error: {exc}", file=sys.stderr)
        return False
    finally:
        outcome["finished_at"] = datetime.now(timezone.utc).isoformat()
        json_write(directory / "outcome.json", outcome)
        try:
            with (directory / "container.log").open("w", encoding="utf-8") as log:
                subprocess.run([*base, "logs", "--no-color"], stdout=log, stderr=subprocess.STDOUT)
            result = subprocess.run([*base, "ps", "-a", "--format", "json"], capture_output=True, text=True)
            (directory / "container_status.jsonl").write_text(result.stdout, encoding="utf-8")
            subprocess.run([*base, "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("list", "render-compose", "smoke", "pilot", "run"):
        child = sub.add_parser(command)
        child.add_argument("--dataset", choices=("mnist", "cifar10", "all"), default="mnist")
        child.add_argument("--scenario", choices=SCENARIOS)
        child.add_argument("--all", action="store_true", help="all six scenarios")
        child.add_argument("--seeds", type=int, nargs="+")
        child.add_argument("--duration", type=float)
        child.add_argument("--interval", type=float)
        child.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        child.add_argument("--results-root", type=Path, default=ROOT / "results" / "johann")
    summary = sub.add_parser("summarize")
    summary.add_argument("path", type=Path)
    summary.add_argument("--run-id", help="restrict a collection to one batch ID, without -s<seed>")
    summary.add_argument("--mode", choices=("smoke", "pilot", "collection"))
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "summarize":
            from experiments.johann.analysis import summarize_collection, summarize_run
            if (args.path / "resolved_experiment.json").is_file():
                summarize_run(args.path)
            else:
                summarize_collection(args.path, run_id=args.run_id, mode=args.mode)
            return 0
        datasets = ("mnist", "cifar10") if args.dataset == "all" else (args.dataset,)
        scenarios = SCENARIOS if args.all or args.command == "list" else (args.scenario or "concentrated-r",)
        if args.all and args.scenario:
            raise ValueError("choose --all or --scenario")
        seeds = args.seeds or ([42, 43, 44, 45, 46] if args.command == "run" else [7])
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be unique")
        # Validate the entire selection before creating artifacts or launching Docker.
        specs = []
        for dataset in datasets:
            for scenario in scenarios:
                for seed in seeds:
                    mode = "collection" if args.command == "run" else "pilot"
                    duration, interval = args.duration, args.interval
                    if args.command == "smoke":
                        mode = "smoke"
                        duration = duration if duration is not None else 30
                        interval = interval if interval is not None else duration / 20
                    specs.append(load_spec(dataset, scenario, seed, duration, interval, mode))
        if args.command == "list":
            for spec in specs:
                print(f"{spec['dataset']}/{spec['scenario']}")
            return 0
        for spec in specs:
            directory, spec = prepare(spec, args.run_id, args.results_root.resolve())
            print(directory, flush=True)
            if args.command != "render-compose" and not execute(directory, spec):
                return 1
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
