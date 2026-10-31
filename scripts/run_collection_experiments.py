"""List, validate and run the six Thiago simulated-collection examples."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.simulated_collection.catalog import (
    MANIFEST, CollectionExperiment, client_settings, collection_compose, load_catalog,
)
from experiments.simulated_collection.validate import validate_stocks
from app.collection import read_json
from scripts.plot_metrics import load_experiments, plot_metrics
from scripts.run_paper_experiments import summarize_run, wait_for_completion


def render(experiment: CollectionExperiment, run_id: str, results_root: Path) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", experiment.spec.experiment_id):
        raise ValueError("experiment ID must use letters, digits, hyphens or underscores")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run ID must use 1..80 letters, digits, hyphens or underscores")
    directory = (results_root / experiment.spec.experiment_id / run_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    compose = collection_compose(experiment, run_id, directory)
    resolved = {
        "experiment": asdict(experiment.spec), "mode": experiment.mode,
        "window_rounds": experiment.window_rounds, "run_id": run_id,
        "smoke": experiment.smoke,
        "settings": [{**asdict(client_settings(experiment, client.client_id)), "run_id": run_id}
                     for client in experiment.spec.clients],
    }
    for name, value in (("compose.json", compose), ("resolved_experiment.json", resolved),
                        ("schedule.json", experiment.schedule)):
        (directory / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return directory


def docker_validate(directory: Path) -> None:
    compose = json.loads((directory / "compose.json").read_text())
    image = compose["services"]["client-0"]["image"]
    subprocess.run(["docker", "compose", "-f", str(directory / "compose.json"),
                    "build", "client-0"], cwd=ROOT, check=True)
    subprocess.run(["docker", "run", "--rm", "--mount",
                    f"type=bind,source={directory},target=/results", image,
                    "python", "-m", "experiments.simulated_collection.validate"],
                   cwd=ROOT, check=True)


def run_stack(directory: Path, experiment: CollectionExperiment) -> None:
    base = ["docker", "compose", "-f", str(directory / "compose.json")]
    try:
        subprocess.run([*base, "up", "-d", "--no-build"], cwd=ROOT, check=True)
        if not wait_for_completion(experiment.spec.client_count, poll_interval=2,
                                   timeout=1200 if experiment.smoke else 86400):
            raise RuntimeError("one or more collection clients failed")
        summarize_run(directory)
        plot_metrics(load_experiments([directory]), directory / "charts",
                     "round", "both", show_neighbor_count=True)
        (directory / "outcome.json").write_text(json.dumps({"completed": True}), encoding="utf-8")
    except Exception as exc:
        (directory / "outcome.json").write_text(
            json.dumps({"completed": False, "error": str(exc)}), encoding="utf-8")
        raise
    finally:
        logs = subprocess.run([*base, "logs", "--no-color"], cwd=ROOT, capture_output=True, text=True)
        (directory / "container.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
        subprocess.run([*base, "down"], cwd=ROOT, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "validate", "render-compose", "run", "smoke"))
    parser.add_argument("experiment_ids", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", choices=("mnist", "cifar10", "harth"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--smoke", action="store_true", help="use the five-round smoke agenda")
    parser.add_argument("--data-dir", type=Path, help="validate local prepared data instead of Docker")
    args = parser.parse_args(argv)
    try:
        if args.results_root is None:
            args.results_root = ROOT / read_json(args.manifest).get(
                "results_root", "results/thiago-coleta-simulada")
        catalog = load_catalog(args.manifest, smoke=args.smoke or args.command == "smoke")
        selected = [item for item in catalog
                    if (not args.experiment_ids or item.spec.experiment_id in args.experiment_ids)
                    and (not args.dataset or item.spec.dataset == args.dataset)]
        missing = set(args.experiment_ids) - {item.spec.experiment_id for item in selected}
        if missing or not selected:
            raise ValueError(f"unknown or filtered experiments: {sorted(missing)}")
        if args.command == "list":
            for item in selected:
                print(f"{item.spec.experiment_id}: {item.mode}, rounds={item.spec.rounds}")
            return 0
        if not args.experiment_ids and not args.all:
            raise ValueError("provide experiment IDs or --all")
        if args.data_dir and args.command != "validate":
            raise ValueError("--data-dir is only supported by validate")
        directories = [(item, render(item, args.run_id, args.results_root)) for item in selected]
        # Validate ALL selected schedules against actual data before starting any client stack.
        for item, directory in directories:
            if args.command == "render-compose":
                print(directory / "compose.json")
            elif args.data_dir:
                resolved = json.loads((directory / "resolved_experiment.json").read_text())
                result = validate_stocks(resolved, directory, str(args.data_dir.resolve()))
                (directory / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"Validated {len(result['clients'])} client stocks: {directory}")
            else:
                docker_validate(directory)
        if args.command in {"run", "smoke"}:
            for item, directory in directories:
                run_stack(directory, item)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
