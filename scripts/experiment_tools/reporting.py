"""Reports for current and historical experiment metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from scripts.plot_metrics import load_experiments, plot_final_metrics


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
        spec = spec.get("experiment", spec)
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
        "# Experiment comparison",
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
    colors = ["#72b7b2"] * len(rows)
    axis.bar(range(len(rows)), values, color=colors)
    axis.set_xticks(range(len(rows)), labels, rotation=45, ha="right")
    axis.set_ylabel("Connected E/R average accuracy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=150)
    plt.close(figure)
    return [csv_path, markdown_path, chart_path]
