"""Generate, run, and summarize the paper's decentralized experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.paper.catalog import (
    PaperExperiment,
    build_catalog,
    experiment_by_id,
)


from scripts.plot_metrics import load_experiments, plot_final_metrics


DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / "paper"
DEFAULT_GENERATED_ROOT = DEFAULT_RESULTS_ROOT / ".generated"


def default_run_id(experiment_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_id}-{timestamp}"


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(item) for item in values)


def _service_environment(
    experiment: PaperExperiment,
    client_id: int,
    run_id: str,
    results_directory: Path,
    smoke: bool,
    ton_iot_path: Path | None,
) -> dict[str, str]:
    client = experiment.clients[client_id]
    neighbors = ",".join(
        f"http://client-{neighbor}:8000" for neighbor in client.neighbors
    )
    environment = {
        "TRAINER_MODULE": experiment.trainer_module,
        "CLIENT_COUNT": str(experiment.client_count),
        "CLIENT_ID": str(client_id),
        "CLIENT_NAME": f"client-{client_id}",
        "HEAD_CLIENT_IDS": _csv(experiment.head_client_ids),
        "RUN_ID": run_id,
        "EMPTY_CLIENT_IDS": _csv(experiment.empty_client_ids),
        "NEIGHBORS": neighbors,
        "ROUNDS": "1" if smoke else str(experiment.rounds),
        "LOCAL_EPOCHS": "1" if smoke else str(experiment.local_epochs),
        "BATCH_SIZE": str(experiment.batch_size),
        "LEARNING_RATE": str(experiment.learning_rate),
        "MOMENTUM": str(experiment.momentum),
        "SEED": str(experiment.seed),
        "PARTITION_MODE": "iid",
        "HEAD_START_DELAY": "8",
        "REQUEST_TIMEOUT": "5",
        "REQUEST_RETRIES": "6",
        "RETRY_DELAY": "1",
        "RESULTS_DIR": "/results",
        "EXPERIMENT_ID": experiment.experiment_id,
        "TOPOLOGY_NAME": experiment.topology_name,
        "COMPONENT_ID": str(client.component_id),
        "COMPONENT_SIZE": str(client.component_size),
        "PAPER_ROLE": client.paper_role,
        "MAX_TRAIN_SAMPLES": "400" if smoke else "0",
        "MAX_TEST_SAMPLES": "200" if smoke else "0",
        "TORCH_NUM_THREADS": "1",
    }
    if experiment.dataset == "ton_iot":
        if ton_iot_path is None:
            raise ValueError("TON_IOT_PATH or --ton-iot-path is required")
        environment["TON_IOT_PATH"] = "/data/ton_iot/Train_Test_Network.csv"
    return environment


def resolved_spec(
    experiment: PaperExperiment, run_id: str, smoke: bool
) -> dict[str, Any]:
    value = asdict(experiment)
    value["run_id"] = run_id
    value["smoke"] = smoke
    return value


def render_compose(
    experiment: PaperExperiment,
    run_id: str,
    output_path: Path,
    results_root: Path = DEFAULT_RESULTS_ROOT,
    smoke: bool = False,
    ton_iot_path: Path | None = None,
) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_directory = (results_root / experiment.experiment_id / run_id).resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    image_name = f"dfl-paper-{experiment.dataset}:latest"
    services: dict[str, Any] = {}
    for client in experiment.clients:
        volumes = [f"{run_directory}:/results"]
        if experiment.dataset == "ton_iot":
            assert ton_iot_path is not None
            volumes.append(
                f"{ton_iot_path.resolve()}:/data/ton_iot/Train_Test_Network.csv:ro"
            )
        service: dict[str, Any] = {
            "image": image_name,
            "pull_policy": "never",
            "restart": "no",
            "ports": [f"{8000 + client.client_id}:8000"],
            "environment": _service_environment(
                experiment,
                client.client_id,
                run_id,
                run_directory,
                smoke,
                ton_iot_path,
            ),
            "volumes": volumes,
            "healthcheck": {
                "test": [
                    "CMD",
                    "python",
                    "-c",
                    (
                        "import urllib.request; "
                        "urllib.request.urlopen("
                        "'http://127.0.0.1:8000/health', timeout=2)"
                    ),
                ],
                "interval": "5s",
                "timeout": "3s",
                "start_period": "30s",
                "retries": 12,
            },
        }
        if client.client_id == 0:
            service["build"] = {
                "context": str(REPOSITORY_ROOT),
                "dockerfile": experiment.dockerfile,
            }
        services[f"client-{client.client_id}"] = service
    compose = {
        "name": f"dfl-{experiment.experiment_id}".replace("_", "-"),
        "services": services,
    }
    output_path.write_text(
        json.dumps(compose, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    spec_path = run_directory / "resolved_experiment.json"
    spec_path.write_text(
        json.dumps(resolved_spec(experiment, run_id, smoke), indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path, run_directory


def _load_final_records(run_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_directory.glob("client_*_metrics.jsonl")):
        final: dict[str, Any] | None = None
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        final = value
        if final is not None:
            records.append(final)
    if not records:
        raise ValueError(f"no client metrics found in {run_directory}")
    return records


def summarize_run(
    run_directory: Path, show_client_points: bool = False,
) -> tuple[Path, Path, list[Path]]:
    experiments = load_experiments([run_directory])
    records = _load_final_records(run_directory)
    summary_rows: list[dict[str, Any]] = []
    for record in records:
        test = record.get("test") or {}
        summary_rows.append(
            {
                "run_id": record.get("run_id", ""),
                "experiment_id": record.get("experiment_id", ""),
                "topology_name": record.get("topology_name", ""),
                "client_id": record["client_id"],
                "paper_role": record.get("paper_role", ""),
                "has_local_data": record.get("has_local_data", True),
                "is_connected": record.get("is_connected", False),
                "sample_count": record.get("sample_count", 0),
                "final_round": record.get("round", 0),
                "accuracy": test.get("accuracy", ""),
                "loss": test.get("loss", ""),
            }
        )
    eligible = [
        row
        for row in summary_rows
        if row["is_connected"] and row["paper_role"] in {"R", "E"}
    ]
    aggregate_accuracy = (
        sum(float(row["accuracy"]) for row in eligible) / len(eligible)
        if eligible
        else None
    )
    csv_path = run_directory / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    markdown_path = run_directory / "summary.md"
    aggregate_text = (
        f"{aggregate_accuracy:.6f}" if aggregate_accuracy is not None else "n/a"
    )
    lines = [
        "# Experiment summary",
        "",
        f"- Connected E/R average accuracy: {aggregate_text}",
        f"- Reported clients: {len(summary_rows)}",
        "",
        "| Client | Role | Connected | Samples | Accuracy | Loss |",
        "| ---: | :---: | :---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['client_id']} | {row['paper_role']} | "
            f"{row['is_connected']} | {row['sample_count']} | "
            f"{row['accuracy']} | {row['loss']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chart_paths: list[Path] = []
    accuracy_path = run_directory / "final_accuracy.png"
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.bar(
        [int(row["client_id"]) for row in summary_rows],
        [float(row["accuracy"]) for row in summary_rows],
    )
    axis.set_xlabel("Client ID")
    axis.set_ylabel("Final test accuracy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(accuracy_path, dpi=150)
    plt.close(figure)
    chart_paths.append(accuracy_path)

    loss_path = run_directory / "loss_evolution.png"
    figure, axis = plt.subplots(figsize=(10, 6))
    for metrics_path in sorted(run_directory.glob("client_*_metrics.jsonl")):
        rounds: list[int] = []
        losses: list[float] = []
        client_id = int(metrics_path.stem.split("_")[1])
        with metrics_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                loss = (record.get("test") or {}).get("loss")
                if loss is not None:
                    rounds.append(int(record["round"]))
                    losses.append(float(loss))
        if rounds:
            axis.plot(rounds, losses, label=f"Client {client_id}")
    axis.set_xlabel("Round")
    axis.set_ylabel("Test loss")
    axis.grid(alpha=0.3)
    if len(summary_rows) <= 20:
        axis.legend(ncol=2, fontsize="small")
    figure.tight_layout()
    figure.savefig(loss_path, dpi=150)
    plt.close(figure)
    chart_paths.append(loss_path)
    chart_paths.extend(plot_final_metrics(experiments, run_directory, show_client_points))
    return csv_path, markdown_path, chart_paths


def summarize_collection(root: Path, show_client_points: bool = False) -> list[Path]:
    run_directories = sorted(
        {
            path.parent
            for path in root.rglob("client_*_metrics.jsonl")
            if path.is_file()
        }
    )
    if not run_directories:
        raise ValueError(f"no client metrics found beneath {root}")

    rows: list[dict[str, Any]] = []
    for run_directory in run_directories:
        csv_path, _, _ = summarize_run(run_directory, show_client_points)
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            clients = list(csv.DictReader(stream))
        eligible = [
            client
            for client in clients
            if client["is_connected"] == "True"
            and client["paper_role"] in {"E", "R"}
        ]
        accuracy = (
            sum(float(client["accuracy"]) for client in eligible) / len(eligible)
            if eligible
            else None
        )
        spec_path = run_directory / "resolved_experiment.json"
        spec = (
            json.loads(spec_path.read_text(encoding="utf-8"))
            if spec_path.is_file()
            else {}
        )
        first = clients[0]
        rows.append(
            {
                "experiment_id": first["experiment_id"],
                "run_id": first["run_id"],
                "dataset": spec.get("dataset", ""),
                "topology_name": first["topology_name"],
                "rounds": spec.get("rounds", ""),
                "local_epochs": spec.get("local_epochs", ""),
                "eligible_clients": len(eligible),
                "aggregate_accuracy": "" if accuracy is None else accuracy,
                "run_directory": str(run_directory.resolve()),
            }
        )

    csv_path = root / "comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = root / "comparison.md"
    lines = [
        "# Paper experiment comparison",
        "",
        "| Dataset | Experiment | Run | Topology | Epochs | Accuracy |",
        "| :--- | :--- | :--- | :--- | ---: | ---: |",
    ]
    for row in rows:
        accuracy = row["aggregate_accuracy"]
        accuracy_text = f"{float(accuracy):.6f}" if accuracy != "" else "n/a"
        lines.append(
            f"| {row['dataset']} | {row['experiment_id']} | {row['run_id']} | "
            f"{row['topology_name']} | {row['local_epochs']} | {accuracy_text} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    chart_path = root / "experiment_comparison.png"
    figure, axis = plt.subplots(figsize=(max(10, len(rows) * 0.5), 6))
    labels = [f"{row['experiment_id']}\n{row['run_id']}" for row in rows]
    values = [
        float(row["aggregate_accuracy"])
        if row["aggregate_accuracy"] != ""
        else float("nan")
        for row in rows
    ]
    colors = [
        (
            "#4c78a8"
            if row["topology_name"] == "irregular-sparse"
            else "#f58518"
            if row["topology_name"] == "irregular-dense"
            else "#72b7b2"
        )
        for row in rows
    ]
    axis.bar(range(len(rows)), values, color=colors)
    axis.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
    axis.set_ylabel("Connected E/R average accuracy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return [csv_path, markdown_path, chart_path]


def _filtered_catalog(dataset: str | None) -> list[PaperExperiment]:
    return [
        item for item in build_catalog() if dataset is None or item.dataset == dataset
    ]

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


def _run_compose(
    compose_path: Path,
    experiment: PaperExperiment,
    run_directory: Path,
    show_client_points: bool = False,
) -> int:
    base_command = ["docker", "compose", "-f", str(compose_path)]
    try:
        started = subprocess.run(
            [*base_command, "up", "-d", "--build"],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        if started.returncode:
            return started.returncode
        completed = wait_for_completion(experiment.client_count)
        if completed:
            summarize_run(run_directory, show_client_points)
            return 0
        return 1
    except (OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        try:
            subprocess.run(
                [*base_command, "down"],
                cwd=REPOSITORY_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass


def _selected_experiments(args: argparse.Namespace) -> list[PaperExperiment]:
    catalog = list(build_catalog())
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
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="list experiment IDs")
    list_parser.add_argument("--dataset", choices=("mnist", "cifar10", "ton_iot"))

    render = subparsers.add_parser("render-compose")
    render.add_argument("experiment_id")
    render.add_argument("--run-id")
    render.add_argument("--output", type=Path)
    render.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    render.add_argument("--ton-iot-path", type=Path)

    for command in ("run", "smoke"):
        child = subparsers.add_parser(command)
        child.add_argument("experiment_ids", nargs="*")
        child.add_argument("--show-client-points", action="store_true")
        child.add_argument("--all", action="store_true")
        child.add_argument("--dataset", choices=("mnist", "cifar10", "ton_iot"))
        child.add_argument("--topology")
        child.add_argument("--local-epochs", type=int)
        child.add_argument("--run-id")
        child.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
        child.add_argument("--ton-iot-path", type=Path)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("run_directory", type=Path)
    summary.add_argument("--show-client-points", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list":
        for experiment in _filtered_catalog(args.dataset):
            print(experiment.experiment_id)
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
            experiment = experiment_by_id(args.experiment_id)
            run_id = args.run_id or default_run_id(experiment.experiment_id)
            output = args.output or (
                DEFAULT_GENERATED_ROOT / f"{experiment.experiment_id}-{run_id}.json"
            )
            compose_path, run_directory = render_compose(
                experiment,
                run_id,
                output,
                args.results_root,
                ton_iot_path=args.ton_iot_path,
            )
        except (KeyError, OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(compose_path)
        print(run_directory)
        return 0

    try:
        experiments = _selected_experiments(args)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for experiment in experiments:
        run_id = args.run_id or default_run_id(experiment.experiment_id)
        output = (
            DEFAULT_GENERATED_ROOT / f"{experiment.experiment_id}-{run_id}.json"
        )
        try:
            compose_path, run_directory = render_compose(
                experiment,
                run_id,
                output,
                args.results_root,
                smoke=args.command == "smoke",
                ton_iot_path=args.ton_iot_path,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(compose_path)
        print(run_directory)
        result = _run_compose(compose_path, experiment, run_directory, args.show_client_points)
        if result:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
