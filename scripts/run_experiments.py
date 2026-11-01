"""Run and summarize fixed-data experiments from an explicit manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.experiment_tools.catalog import (
    ExperimentSpec, build_catalog, experiment_by_id, load_manifest, reject_collection_manifest,
)
from scripts.experiment_tools.execution import (
    DEFAULT_RESULTS_ROOT, default_run_id, render_compose, run_compose,
)
from scripts.experiment_tools.reporting import summarize_collection, summarize_run


def _filtered_catalog(
    dataset: str | None, manifest: dict[str, Any],
) -> list[ExperimentSpec]:
    return [
        item for item in build_catalog(manifest) if dataset is None or item.dataset == dataset
    ]

def _selected_experiments(args: argparse.Namespace) -> list[ExperimentSpec]:
    catalog = list(build_catalog(load_manifest(args.manifest)))
    requested = set(args.experiment_ids)
    selected = [
        item
        for item in catalog
        if (not requested or item.experiment_id in requested)
        and (args.dataset is None or item.dataset == args.dataset)
        and (args.topology is None or item.topology_name == args.topology)
        and (
            args.local_epochs is None
            or item.local_epochs == args.local_epochs
        )
    ]
    missing = requested - {item.experiment_id for item in selected}
    if missing:
        raise KeyError(f"unknown or filtered experiment(s): {', '.join(sorted(missing))}")
    if not selected:
        raise ValueError("no experiments matched the selection")
    if not requested and not args.all:
        raise ValueError("provide experiment IDs or pass --all with optional filters")
    reject_collection_manifest(load_manifest(args.manifest))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="list experiment IDs")
    list_parser.add_argument("--dataset", choices=("mnist", "cifar10", "harth"))

    render = subparsers.add_parser("render-compose")
    render.add_argument("experiment_id")
    render.add_argument("--run-id")
    render.add_argument("--output", type=Path)
    render.add_argument("--results-root", type=Path)

    for command in ("run", "smoke"):
        child = subparsers.add_parser(command)
        child.add_argument("experiment_ids", nargs="*")
        child.add_argument("--show-client-points", action="store_true")
        child.add_argument("--all", action="store_true")
        child.add_argument("--dataset", choices=("mnist", "cifar10", "harth"))
        child.add_argument("--topology")
        child.add_argument("--local-epochs", type=int)
        child.add_argument("--run-id")
        child.add_argument("--results-root", type=Path)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("run_directory", type=Path)
    summary.add_argument("--show-client-points", action="store_true")
    for name in ("list", "render-compose", "run", "smoke"):
        subparsers.choices[name].add_argument(
            "--manifest", type=Path, required=True, help="explicit experiment manifest",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "results_root"):
        try:
            manifest = load_manifest(args.manifest)
            reject_collection_manifest(manifest)
            if args.results_root is None:
                args.results_root = REPOSITORY_ROOT / manifest.get(
                    "results_root", DEFAULT_RESULTS_ROOT,
                )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.command == "list":
        try:
            manifest = load_manifest(args.manifest)
            reject_collection_manifest(manifest)
            for experiment in _filtered_catalog(args.dataset, manifest):
                print(experiment.experiment_id)
        except (KeyError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0
    if args.command == "summarize":
        try:
            if list(args.run_directory.glob("client_*_metrics.jsonl")):
                csv_path, markdown_path, charts = summarize_run(
                    args.run_directory, args.show_client_points
                )
                written = [csv_path, markdown_path, *charts]
            else:
                written = summarize_collection(args.run_directory, args.show_client_points)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        for path in written:
            print(path)
        return 0

    if args.command == "render-compose":
        try:
            experiment = experiment_by_id(
                args.experiment_id,
                load_manifest(args.manifest),
            )
            run_id = args.run_id or default_run_id(experiment.experiment_id)
            output = args.output or (
                args.results_root / ".generated" / f"{experiment.experiment_id}-{run_id}.json"
            )
            compose_path, run_directory = render_compose(
                experiment,
                run_id,
                output,
                args.results_root,
            )
        except (KeyError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(compose_path)
        print(run_directory)
        return 0

    try:
        experiments = _selected_experiments(args)
    except (KeyError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for experiment in experiments:
        run_id = args.run_id or default_run_id(experiment.experiment_id)
        output = (
            args.results_root / ".generated" / f"{experiment.experiment_id}-{run_id}.json"
        )
        try:
            compose_path, run_directory = render_compose(
                experiment,
                run_id,
                output,
                args.results_root,
                smoke=args.command == "smoke",
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(compose_path)
        print(run_directory)
        result = run_compose(compose_path, experiment, run_directory, args.show_client_points)
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
