"""Plot per-client training and test metrics from one or more experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


CLIENT_FILE_PATTERN = re.compile(r"client_(\d+)_metrics\.jsonl$")
PHASES = ("train", "test")


class MetricsPlotError(ValueError):
    """Raised when experiment metrics cannot be loaded or plotted."""


@dataclass(frozen=True)
class MetricRecord:
    client_id: int
    model_version: int
    time: datetime
    values: dict[str, dict[str, float]]
    run_id: str | None = None
    started_at: datetime | None = None
    round_number: int | None = None


@dataclass(frozen=True)
class Experiment:
    path: Path
    label: str
    clients: dict[int, list[MetricRecord]]


@dataclass(frozen=True)
class PlotSeries:
    experiment_index: int
    label: str
    phase: str
    x: list[int | float]
    y: list[float]


def _parse_timestamp(value: Any, location: str) -> datetime:
    if not isinstance(value, str):
        raise MetricsPlotError(f"{location}: 'time' must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MetricsPlotError(
            f"{location}: invalid ISO-8601 timestamp {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MetricsPlotError(f"{location}: timestamp must include a timezone")
    return parsed


def _parse_integer(value: Any, field: str, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MetricsPlotError(f"{location}: '{field}' must be an integer")
    return value


def _parse_phase(
    record: dict[str, Any], phase: str, location: str
) -> dict[str, float]:
    raw_values = record.get(phase)
    if raw_values is None:
        return {}
    if not isinstance(raw_values, dict):
        raise MetricsPlotError(f"{location}: '{phase}' must be an object")

    values: dict[str, float] = {}
    for metric, raw_value in raw_values.items():
        if not isinstance(metric, str) or not metric:
            raise MetricsPlotError(
                f"{location}: {phase} metric names must be non-empty strings"
            )
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise MetricsPlotError(f"{location}: {phase}.{metric} must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise MetricsPlotError(f"{location}: {phase}.{metric} must be finite")
        values[metric] = value
    return values


def load_metrics_file(path: Path, expected_client_id: int) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise MetricsPlotError(f"cannot read metrics file {path}: {exc}") from exc

    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                raw_record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MetricsPlotError(
                    f"{location}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw_record, dict):
                raise MetricsPlotError(f"{location}: each line must contain an object")

            client_id = _parse_integer(
                raw_record.get("client_id"), "client_id", location
            )
            if client_id != expected_client_id:
                raise MetricsPlotError(
                    f"{location}: client_id {client_id} does not match "
                    f"filename client ID {expected_client_id}"
                )
            model_version = _parse_integer(
                raw_record.get("model_version"), "model_version", location
            )
            run_id = raw_record.get("run_id")
            if run_id is not None and (not isinstance(run_id, str) or not run_id):
                raise MetricsPlotError(f"{location}: run_id must be a non-empty string")
            timestamp = _parse_timestamp(raw_record.get("time"), location)
            values = {
                phase: _parse_phase(raw_record, phase, location) for phase in PHASES
            }
            records.append(
                MetricRecord(
                    client_id=client_id,
                    model_version=model_version,
                    time=timestamp,
                    values=values,
                    run_id=raw_record.get("run_id"),
                    round_number=(
                        _parse_integer(raw_record["round"], "round", location)
                        if "round" in raw_record else model_version
                    ),
                    started_at=(
                        _parse_timestamp(raw_record["started_at"], location)
                        if raw_record.get("started_at") is not None else None
                    ),
                )
            )

    if not records:
        raise MetricsPlotError(f"{path}: metrics file contains no records")
    return records


def _experiment_labels(paths: Sequence[Path]) -> list[str]:
    base_labels = [path.resolve().name or str(path.resolve()) for path in paths]
    totals = {label: base_labels.count(label) for label in set(base_labels)}
    occurrences: dict[str, int] = {}
    labels: list[str] = []
    for label in base_labels:
        occurrences[label] = occurrences.get(label, 0) + 1
        if totals[label] > 1:
            labels.append(f"{label} ({occurrences[label]})")
        else:
            labels.append(label)
    return labels


def load_experiments(paths: Sequence[Path]) -> list[Experiment]:
    labels = _experiment_labels(paths)
    experiments: list[Experiment] = []
    for path, label in zip(paths, labels):
        if not path.exists():
            raise MetricsPlotError(f"experiment directory does not exist: {path}")
        if not path.is_dir():
            raise MetricsPlotError(f"experiment path is not a directory: {path}")

        clients: dict[int, list[MetricRecord]] = {}
        for metrics_path in sorted(path.glob("client_*_metrics.jsonl")):
            match = CLIENT_FILE_PATTERN.fullmatch(metrics_path.name)
            if match is None:
                continue
            client_id = int(match.group(1))
            clients[client_id] = load_metrics_file(metrics_path, client_id)

        if not clients:
            raise MetricsPlotError(
                f"{path}: no client_<id>_metrics.jsonl files were found"
            )
        run_ids = {record.run_id for records in clients.values()
                   for record in records if record.run_id is not None}
        if len(run_ids) > 1:
            raise MetricsPlotError(f"{path}: metrics contain multiple run IDs")
        experiments.append(Experiment(path=path, label=label, clients=clients))
    return experiments


def selected_phases(phase: str) -> tuple[str, ...]:
    return PHASES if phase == "both" else (phase,)


def available_metrics(
    experiments: Sequence[Experiment], client_id: int, phases: Iterable[str]
) -> set[str]:
    metrics: set[str] = set()
    for experiment in experiments:
        for record in experiment.clients.get(client_id, []):
            for phase in phases:
                metrics.update(record.values[phase])
    return metrics


def build_series(
    experiments: Sequence[Experiment],
    client_id: int,
    metric: str,
    phase: str,
    x_axis: str,
) -> tuple[list[PlotSeries], list[str]]:
    series: list[PlotSeries] = []
    warnings: list[str] = []
    phases = selected_phases(phase)

    for experiment_index, experiment in enumerate(experiments):
        records = experiment.clients.get(client_id)
        if records is None:
            warnings.append(
                f"{experiment.label}: client {client_id} is unavailable; "
                f"omitting it from {metric}"
            )
            continue
        first_time = min(record.time for record in records)
        for selected_phase in phases:
            points: list[tuple[int | float, float]] = []
            for record in records:
                if metric not in record.values[selected_phase]:
                    continue
                if x_axis == "model_version":
                    x_value: int | float = record.model_version
                else:
                    x_value = (record.time - first_time).total_seconds()
                points.append((x_value, record.values[selected_phase][metric]))

            if not points:
                warnings.append(
                    f"{experiment.label}: client {client_id} has no "
                    f"{selected_phase}.{metric} values; omitting that series"
                )
                continue
            points.sort(key=lambda point: point[0])
            series.append(
                PlotSeries(
                    experiment_index=experiment_index,
                    label=experiment.label,
                    phase=selected_phase,
                    x=[point[0] for point in points],
                    y=[point[1] for point in points],
                )
            )
    return series, warnings


def _safe_metric_name(metric: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric).strip("._")
    return safe_name or "metric"


def plot_metrics(
    experiments: Sequence[Experiment],
    output_dir: Path,
    x_axis: str,
    phase: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client_ids = sorted(
        {client_id for experiment in experiments for client_id in experiment.clients}
    )
    colors = plt.get_cmap("tab10")
    line_styles = {"train": "-", "test": "--"}
    written: list[Path] = []
    emitted_warnings: set[str] = set()

    for client_id in client_ids:
        metrics = sorted(
            available_metrics(experiments, client_id, selected_phases(phase))
        )
        for metric in metrics:
            plot_series, warnings = build_series(
                experiments, client_id, metric, phase, x_axis
            )
            for warning in warnings:
                if warning not in emitted_warnings:
                    print(f"warning: {warning}", file=sys.stderr)
                    emitted_warnings.add(warning)
            if not plot_series:
                continue

            figure, axis = plt.subplots(figsize=(10, 6))
            for item in plot_series:
                axis.plot(
                    item.x,
                    item.y,
                    color=colors(item.experiment_index % colors.N),
                    linestyle=line_styles[item.phase],
                    marker="o",
                    markersize=3,
                    label=f"{item.label} — {item.phase}",
                )
            axis.set_title(f"Client {client_id} — {metric}")
            axis.set_xlabel(
                "Model version" if x_axis == "model_version" else "Elapsed time (s)"
            )
            axis.set_ylabel(metric)
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()

            output_path = (
                output_dir / f"client_{client_id}_{_safe_metric_name(metric)}.png"
            )
            figure.savefig(output_path, dpi=150)
            plt.close(figure)
            written.append(output_path)

    if not written:
        raise MetricsPlotError(
            f"no usable metrics were found for the selected phase '{phase}'"
        )
    return written



@dataclass(frozen=True)
class CombinedSeries:
    rounds: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    participants: np.ndarray
    points: dict[int, tuple[np.ndarray, np.ndarray]]


def build_combined_series(experiment: Experiment, metric: str) -> CombinedSeries:
    """Aggregate only observations recorded at the same round."""
    points = {}
    for client_id, items in sorted(experiment.clients.items()):
        deduplicated: dict[int, MetricRecord] = {}
        for record in items:
            round_number = (record.round_number if record.round_number is not None
                            else record.model_version)
            previous = deduplicated.get(round_number)
            if previous is None or record.model_version >= previous.model_version:
                deduplicated[round_number] = record
        usable = [(number, record) for number, record in sorted(deduplicated.items())
                  if metric in record.values["test"]]
        if not usable:
            print(f"warning: {experiment.label}: client {client_id} has no "
                  f"test.{metric} values", file=sys.stderr)
            continue
        points[client_id] = (
            np.array([number for number, _ in usable], dtype=np.int64),
            np.array([record.values["test"][metric] for _, record in usable]),
        )
    grid = np.array(sorted({number for x, _ in points.values() for number in x}),
                    dtype=np.int64)
    values = np.full((len(points), len(grid)), np.nan)
    for index, (x, y) in enumerate(points.values()):
        values[index, np.searchsorted(grid, x)] = y
    participants = np.sum(np.isfinite(values), axis=0)
    mean = np.full(len(grid), np.nan)
    std = np.full(len(grid), np.nan)
    for index in range(len(grid)):
        available = values[np.isfinite(values[:, index]), index]
        if available.size:
            mean[index] = available.mean()
            std[index] = available.std(ddof=0)
    return CombinedSeries(grid, mean, std, participants, points)


def plot_combined_metrics(
    experiment: Experiment, output_dir: Path, show_client_points: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    client_ids = sorted(experiment.clients)
    colors = plt.get_cmap("turbo")
    client_colors = {client: colors((i + 0.5) / len(client_ids))
                     for i, client in enumerate(client_ids)}
    for metric in ("accuracy", "loss"):
        series = build_combined_series(experiment, metric)
        if not series.points:
            continue
        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(series.rounds, series.mean, color="black", marker=".",
                  label="Mean", zorder=4)
        axis.fill_between(series.rounds, series.mean - series.std,
                          series.mean + series.std, color="gray", alpha=0.25,
                          label="\u00b11 standard deviation")
        if show_client_points:
            for client_id, (x, y) in series.points.items():
                axis.scatter(x, y, color=client_colors[client_id], s=22,
                             alpha=0.75, label=f"Client {client_id}", zorder=3)
        axis.set_title(f"{experiment.label} — test {metric}")
        axis.set_xlabel("Round")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylabel(f"Test {metric}")
        axis.grid(alpha=0.3)
        axis.legend(fontsize="small", ncol=2)
        figure.tight_layout()
        stem = f"combined_test_{metric}_round"
        path = output_dir / f"{stem}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        csv_path = output_dir / f"{stem}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["round", "mean", "std", "participants"])
            writer.writerows(zip(series.rounds, series.mean, series.std,
                                 series.participants))
        written.extend([path, csv_path])
    return written


def load_confusion(path: Path, client_id: int, records: list[MetricRecord]) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MetricsPlotError(f"{path}: cannot load confusion: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("client_id") != client_id:
        raise MetricsPlotError(f"{path}: invalid confusion client ID")
    run_ids = {r.run_id for r in records if r.run_id is not None}
    if run_ids and raw.get("run_id") not in run_ids:
        raise MetricsPlotError(f"{path}: confusion run ID does not match metrics")
    if raw.get("model_version") != max(r.model_version for r in records):
        raise MetricsPlotError(f"{path}: confusion model version does not match final metrics")
    kind = raw.get("classification_type")
    labels = raw.get("labels")
    if (kind not in ("multiclass", "binary", "multilabel")
            or not isinstance(labels, list) or not labels):
        raise MetricsPlotError(f"{path}: invalid classification type or labels")
    if not all(isinstance(label, str) for label in labels):
        raise MetricsPlotError(f"{path}: labels must be strings")
    if kind == "binary" and len(labels) != 2:
        raise MetricsPlotError(f"{path}: binary confusion requires two labels")
    try:
        counts = np.asarray(raw.get("counts"))
    except ValueError as exc:
        raise MetricsPlotError(f"{path}: invalid confusion counts") from exc
    expected = (
        (len(labels), 2, 2) if kind == "multilabel"
        else (len(labels), len(labels))
    )
    if (counts.shape != expected or counts.dtype.kind not in "iu"
            or np.any(counts < 0)):
        raise MetricsPlotError(f"{path}: invalid confusion counts")
    samples = raw.get("sample_count")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise MetricsPlotError(f"{path}: invalid sample count")
    totals = counts.sum(axis=(-2, -1))
    if np.any(totals != samples):
        raise MetricsPlotError(f"{path}: counts do not match sample count")
    return raw


def plot_confusions(experiment: Experiment, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for client_id, records in sorted(experiment.clients.items()):
        source = experiment.path / f"client_{client_id}_confusion.json"
        if not source.exists():
            print(f"warning: {experiment.label}: client {client_id} has no final "
                  "confusion artifact", file=sys.stderr)
            continue
        raw = load_confusion(source, client_id, records)
        counts = np.asarray(raw["counts"])
        multilabel = raw["classification_type"] == "multilabel"
        matrices = counts if multilabel else counts[np.newaxis, ...]
        columns = min(4, len(matrices))
        rows = math.ceil(len(matrices) / columns)
        figure, axes = plt.subplots(rows, columns,
                                    figsize=(5 * columns, 4.5 * rows),
                                    squeeze=False, layout="constrained")
        maximum = max(int(counts.max()), 1)
        for index, axis in enumerate(axes.flat):
            if index >= len(matrices):
                axis.set_visible(False)
                continue
            matrix = matrices[index]
            labels = ["0", "1"] if multilabel else raw["labels"]
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
            axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
            axis.set_yticks(range(len(labels)), labels)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("Actual class")
            if multilabel:
                axis.set_title(f"Diagnosis {raw['labels'][index]}")
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = int(matrix[row, col])
                    axis.text(col, row, str(value), ha="center", va="center",
                              color="white" if value > maximum / 2 else "black")
        figure.suptitle(f"{experiment.label} — Client {client_id} final confusion")
        figure.colorbar(image, ax=[a for a in axes.flat if a.get_visible()],
                        label="Count", shrink=0.8)
        path = output_dir / f"client_{client_id}_confusion.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def plot_final_metrics(
    experiments: Sequence[Experiment], output_dir: Path,
    show_client_points: bool = False,
) -> list[Path]:
    written = []
    for index, experiment in enumerate(experiments):
        directory = output_dir if len(experiments) == 1 else (
            output_dir / f"experiment_{index + 1}_{_safe_metric_name(experiment.label)}"
        )
        written.extend(plot_combined_metrics(experiment, directory, show_client_points))
        written.extend(plot_confusions(experiment, directory))
    return written

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot client metrics. The final path is the output directory and all "
            "preceding paths are experiment directories."
        )
    )
    parser.add_argument(
        "--x-axis",
        choices=("model_version", "time"),
        default="model_version",
        help="X axis to plot (default: model_version)",
    )
    parser.add_argument(
        "--phase",
        choices=("train", "test", "both"),
        default="both",
        help="metric phase to plot (default: both)",
    )
    parser.add_argument(
        "--show-client-points", action="store_true",
        help="overlay original client observations on combined round plots",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="one or more experiment directories followed by the output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if len(args.paths) < 2:
        parser.error(
            "provide at least one experiment directory and an output directory"
        )

    experiment_paths = [Path(path) for path in args.paths[:-1]]
    output_dir = Path(args.paths[-1])
    try:
        experiments = load_experiments(experiment_paths)
        written = plot_metrics(experiments, output_dir, args.x_axis, args.phase)
        written.extend(plot_final_metrics(experiments, output_dir,
                                          args.show_client_points))
    except (MetricsPlotError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {len(written)} artifact(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
