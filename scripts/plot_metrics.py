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
from matplotlib.ticker import MaxNLocator, PercentFormatter


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
    neighbor_count: int | None = None
    aggregation_weights: dict[int, float] | None = None
    aggregation_weight: float | None = None
    neighbor_versions: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class Experiment:
    path: Path
    label: str
    clients: dict[int, list[MetricRecord]]
    run_id: str | None = None


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


def load_metrics_file(path: Path, expected_client_id: int,
                      allow_empty: bool = False) -> list[MetricRecord]:
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
                    neighbor_count=_neighbor_count(raw_record, location),
                    aggregation_weights=_aggregation_weights(raw_record, location),
                    aggregation_weight=(
                        _positive_weight(raw_record["aggregation_weight"], location)
                        if "aggregation_weight" in raw_record else None
                    ),
                    neighbor_versions=(tuple(dict(peer) for peer in raw_record["neighbor_versions"])
                                       if "neighbor_versions" in raw_record else None),
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

    if not records and not allow_empty:
        raise MetricsPlotError(f"{path}: metrics file contains no records")
    return records


def _neighbor_count(record: dict, location: str) -> int | None:
    if "neighbor_versions" in record:
        peers = record["neighbor_versions"]
        if not isinstance(peers, list):
            raise MetricsPlotError(f"{location}: neighbor_versions must be a list")
        return len(peers)
    if "participants" in record:
        participants = _parse_integer(record["participants"], "participants", location)
        if participants < 1:
            raise MetricsPlotError(f"{location}: participants must include the local model")
        return participants - 1
    return None


def _positive_weight(value: Any, location: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise MetricsPlotError(f"{location}: invalid aggregation weight")
    return float(value)


def _aggregation_weights(record: dict, location: str) -> dict[int, float] | None:
    if "effective_aggregation_weights" in record:
        raw = record["effective_aggregation_weights"]
        # Supervised/KD adaptation is not a convex mixture of the inputs.
        if raw is None:
            return None
        if not isinstance(raw, dict) or not raw:
            raise MetricsPlotError(f"{location}: invalid effective aggregation weights")
        weights = {}
        for client, value in raw.items():
            if (not isinstance(client, str) or not client.isascii() or not client.isdigit()
                    or int(client) in weights or isinstance(value, bool)
                    or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
                raise MetricsPlotError(f"{location}: invalid effective aggregation weight/client")
            weights[int(client)] = float(value)
        if not math.isclose(sum(weights.values()), 1, abs_tol=1e-8):
            raise MetricsPlotError(f"{location}: effective aggregation weights must sum to one")
        return weights
    peers = record.get("neighbor_versions")
    if peers is None:
        return None
    entries = [record, *peers]
    weights = {}
    missing = False
    for entry in entries:
        if not isinstance(entry, dict):
            raise MetricsPlotError(f"{location}: aggregation participants must be objects")
        if "aggregation_weight" not in entry or "client_id" not in entry:
            missing = True
            continue
        client = _parse_integer(entry["client_id"], "client_id", location)
        weight = _positive_weight(entry["aggregation_weight"], location)
        if client < 0 or client in weights:
            raise MetricsPlotError(f"{location}: invalid or duplicate aggregation weight/client")
        weights[client] = float(weight)
    if missing:
        return None
    if "participants" in record and (
        _parse_integer(record["participants"], "participants", location) != len(entries)
    ):
        raise MetricsPlotError(f"{location}: participants disagree with neighbor_versions")
    return weights


def _read_artifact(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MetricsPlotError(f"{path}: cannot read artifact: {exc}") from exc
    if not isinstance(raw, dict):
        raise MetricsPlotError(f"{path}: artifact must be an object")
    return raw


def _check_identity(raw: dict, experiment: Experiment, path: Path,
                    client_id: int | None = None) -> None:
    runs = {r.run_id for items in experiment.clients.values() for r in items
            if r.run_id is not None}
    if experiment.run_id is not None:
        runs.add(experiment.run_id)
    spec_path = experiment.path / "resolved_experiment.json"
    if not runs and spec_path.is_file():
        run_id = _read_artifact(spec_path).get("run_id")
        if run_id is not None:
            runs.add(run_id)
    if runs and raw.get("run_id") not in runs:
        raise MetricsPlotError(f"{path}: artifact run ID does not match metrics")
    if client_id is not None and (type(raw.get("client_id")) is not int
                                  or raw["client_id"] != client_id):
        raise MetricsPlotError(f"{path}: artifact client ID does not match")


def load_topology(experiment: Experiment) -> tuple[list[int], int | None]:
    """Return the client registry and maximum configured degree, if recoverable."""
    ids = set(experiment.clients)
    degrees: dict[int, int] = {}
    counts = set()
    spec_path = experiment.path / "resolved_experiment.json"
    if spec_path.is_file():
        spec = _read_artifact(spec_path)
        if "run_id" in spec:
            _check_identity(spec, experiment, spec_path)
        count = spec.get("client_count")
        edges = spec.get("edges")
        clients = spec.get("clients")
        if count is not None:
            if type(count) is not int or count < 1:
                raise MetricsPlotError(f"{spec_path}: invalid client_count")
            counts.add(count)
        if clients is not None:
            if not isinstance(clients, list):
                raise MetricsPlotError(f"{spec_path}: invalid clients")
            for client in clients:
                if not isinstance(client, dict):
                    raise MetricsPlotError(f"{spec_path}: invalid client")
                client_id, neighbors = client.get("client_id"), client.get("neighbors")
                if (type(client_id) is not int or client_id < 0 or client_id in degrees
                        or not isinstance(neighbors, list)
                        or any(type(n) is not int or n < 0 or n == client_id for n in neighbors)
                        or len(set(neighbors)) != len(neighbors)):
                    raise MetricsPlotError(f"{spec_path}: invalid topology client/neighbors")
                degrees[client_id] = len(neighbors)
                ids.update([client_id, *neighbors])
        elif edges is not None:
            if not isinstance(edges, list):
                raise MetricsPlotError(f"{spec_path}: invalid edges")
            adjacency: dict[int, set[int]] = {i: set() for i in range(count or 0)}
            for edge in edges:
                if (not isinstance(edge, list) or len(edge) != 2
                        or any(type(n) is not int or n < 0 for n in edge) or edge[0] == edge[1]):
                    raise MetricsPlotError(f"{spec_path}: invalid edge")
                a, b = edge
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
            degrees.update({i: len(peers) for i, peers in adjacency.items()})
            ids.update(adjacency)
    for path in sorted(experiment.path.glob("client_*_topology.json")):
        match = re.fullmatch(r"client_(\d+)_topology.json", path.name)
        if match is None:
            continue
        client = int(match[1])
        raw = _read_artifact(path)
        _check_identity(raw, experiment, path, client)
        count, peers = raw.get("client_count"), raw.get("neighbors")
        if (type(count) is not int or count < 1 or not 0 <= client < count
                or not isinstance(peers, list)
                or any(not isinstance(p, str) or not p for p in peers)
                or len(set(peers)) != len(peers) or len(peers) >= count):
            raise MetricsPlotError(f"{path}: invalid topology")
        if client in degrees and degrees[client] != len(peers):
            raise MetricsPlotError(f"{path}: topology disagrees with resolved experiment")
        counts.add(count)
        degrees[client] = len(peers)
        ids.add(client)
    if len(counts) > 1:
        raise MetricsPlotError(f"{experiment.path}: inconsistent client counts")
    if counts:
        count = next(iter(counts))
        if any(i >= count or i < 0 for i in ids):
            raise MetricsPlotError(f"{experiment.path}: client outside topology")
        ids.update(range(count))
    for items in experiment.clients.values():
        for record in items:
            if record.aggregation_weights:
                if counts and any(i not in ids for i in record.aggregation_weights):
                    raise MetricsPlotError(f"{experiment.path}: contributor outside topology")
                ids.update(record.aggregation_weights)
            if record.client_id in degrees and record.neighbor_count is not None:
                if record.neighbor_count > degrees[record.client_id]:
                    raise MetricsPlotError(f"{experiment.path}: contributors exceed configured degree")
    complete = bool(degrees) and set(degrees) == ids
    if degrees and not complete:
        print(f"warning: {experiment.label}: incomplete topology; neighbor axis limit "
              "is estimated from known degrees and observed contributors", file=sys.stderr)
        observed = [r.neighbor_count or 0 for items in experiment.clients.values() for r in items]
        return sorted(ids), max([*degrees.values(), *observed])
    return sorted(ids), max(degrees.values()) if complete else None


def neighbor_axis_maximum(experiments: Sequence[Experiment]) -> int:
    maxima = []
    for experiment in experiments:
        _, maximum = load_topology(experiment)
        if maximum is None:
            print(f"warning: {experiment.label}: incomplete topology; neighbor axis limit "
                  "is estimated from observed contributors", file=sys.stderr)
            maximum = max((r.neighbor_count or 0 for items in experiment.clients.values()
                           for r in items), default=0)
        maxima.append(maximum)
    return max(maxima, default=0)


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
            clients[client_id] = load_metrics_file(
                metrics_path, client_id,
                allow_empty=(path / f"client_{client_id}_evaluations.jsonl").is_file(),
            )

        for evaluation_path in sorted(path.glob("client_*_evaluations.jsonl")):
            match = re.fullmatch(r"client_(\d+)_evaluations.jsonl", evaluation_path.name)
            if match is not None:
                clients.setdefault(int(match[1]), [])

        if not clients:
            raise MetricsPlotError(
                f"{path}: no client_<id>_metrics.jsonl files were found"
            )
        run_ids = {record.run_id for records in clients.values()
                   for record in records if record.run_id is not None}
        for client in clients:
            source = path / f"client_{client}_evaluations.jsonl"
            if not source.is_file():
                continue
            with source.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    location = f"{source}:{line_number}"
                    try:
                        raw = json.loads(line)
                    except ValueError as exc:
                        raise MetricsPlotError(f"{location}: malformed evaluation") from exc
                    if (not isinstance(raw, dict) or type(raw.get("client_id")) is not int
                            or raw["client_id"] != client
                            or not isinstance(raw.get("run_id"), str) or not raw["run_id"]):
                        raise MetricsPlotError(f"{location}: invalid evaluation identity")
                    run_ids.add(raw["run_id"])
        if len(run_ids) > 1:
            raise MetricsPlotError(f"{path}: metrics contain multiple run IDs")
        experiments.append(Experiment(path=path, label=label, clients=clients,
                                      run_id=next(iter(run_ids), None)))
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
    show_neighbor_count: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client_ids = sorted(
        {client_id for experiment in experiments for client_id in experiment.clients}
    )
    colors = plt.get_cmap("tab10")
    line_styles = {"train": "-", "test": "--"}
    written: list[Path] = []
    emitted_warnings: set[str] = set()
    peer_maximum = neighbor_axis_maximum(experiments) if show_neighbor_count else 0

    for client_id in client_ids:
        metrics = sorted(name for name in
                         available_metrics(experiments, client_id, selected_phases(phase))
                         if not re.fullmatch(r"(?:recall|support)_\d+", name))
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
            if show_neighbor_count and metric == "accuracy" and phase != "train":
                peer_axis = None
                for index, experiment in enumerate(experiments):
                    records = experiment.clients.get(client_id, [])
                    usable = [item for item in records if "accuracy" in item.values["test"]]
                    if any(item.neighbor_count is None for item in usable):
                        print(f"warning: {experiment.label}: client {client_id} has missing "
                              "neighbor counts; those observations are omitted", file=sys.stderr)
                    known = [item for item in usable if item.neighbor_count is not None]
                    if not known:
                        continue
                    if peer_axis is None:
                        peer_axis = axis.twinx()
                        peer_axis.set_ylabel("Contributing neighbors")
                        peer_axis.yaxis.set_major_locator(MaxNLocator(integer=True))
                    first_time = min(item.time for item in records)
                    points = sorted((
                        item.model_version if x_axis == "model_version" else
                        (item.time - first_time).total_seconds(), item.neighbor_count
                    ) for item in known)
                    peer_axis.plot(
                        [point[0] for point in points], [point[1] for point in points],
                        color=colors(index % colors.N), linestyle=":", marker="s",
                        markersize=3, label=f"{experiment.label} — neighbors",
                    )
                if peer_axis is not None:
                    peer_axis.set_ylim(0, peer_maximum if peer_maximum else 0.5)
                    if not peer_maximum:
                        peer_axis.set_yticks([0])
                    handles, labels = axis.get_legend_handles_labels()
                    peer_handles, peer_labels = peer_axis.get_legend_handles_labels()
                    axis.legend(handles + peer_handles, labels + peer_labels)
                else:
                    axis.legend()
            else:
                axis.legend()
            figure.tight_layout()

            output_path = (
                output_dir / f"client_{client_id}_{_safe_metric_name(metric)}.png"
            )
            figure.savefig(output_path, dpi=150)
            plt.close(figure)
            written.append(output_path)

    if not written:
        if any(list(experiment.path.glob("client_*_evaluations.jsonl"))
               for experiment in experiments):
            return []
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
    if raw.get("model_version") != max((r.model_version for r in records), default=0):
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


def plot_class_distribution(experiment: Experiment, output_dir: Path) -> list[Path]:
    rows = []
    missing = []
    labels = None
    classification_type = None
    for client_id, records in sorted(experiment.clients.items()):
        source = experiment.path / f"client_{client_id}_class_distribution.json"
        if not source.is_file():
            missing.append(client_id)
            continue
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MetricsPlotError(f"{source}: invalid class distribution: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("client_id") != client_id:
            raise MetricsPlotError(f"{source}: distribution client ID does not match")
        run_ids = {item.run_id for item in records if item.run_id is not None}
        if run_ids and run_ids != {raw.get("run_id")}:
            raise MetricsPlotError(f"{source}: distribution run ID does not match metrics")
        names, counts = raw.get("labels"), raw.get("counts")
        total, kind = raw.get("sample_count"), raw.get("classification_type")
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
                or not isinstance(counts, list) or len(counts) != len(names)
                or any(type(count) is not int or count < 0 for count in counts)
                or type(total) is not int or total < 0
                or kind not in {"multiclass", "multilabel"}):
            raise MetricsPlotError(f"{source}: invalid class labels or counts")
        if ((kind == "multiclass" and sum(counts) != total)
                or (kind == "multilabel" and any(count > total for count in counts))):
            raise MetricsPlotError(f"{source}: class counts disagree with sample_count")
        if labels is not None and (labels != names or classification_type != kind):
            raise MetricsPlotError(f"{source}: inconsistent class labels or classification type")
        labels, classification_type = names, kind
        rows.append((client_id, counts))
    if missing:
        print(f"warning: {experiment.label}: missing training class distributions for "
              f"clients {missing}; omitted from heatmap", file=sys.stderr)
    if not rows:
        return []
    matrix = np.asarray([counts for _, counts in rows])
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 0.65), max(4, len(rows) * 0.5)))
    display = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0,
                          vmax=max(1, int(matrix.max())))
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(rows)), [f"Client {client_id}" for client_id, _ in rows])
    axis.set_xlabel("Class")
    axis.set_ylabel("Client")
    axis.set_title(f"{experiment.label} — Training class distribution")
    for row in range(len(rows)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      fontsize=8, color="white" if matrix[row, column] > display.norm.vmax / 2 else "black")
    figure.colorbar(display, ax=axis, label=(
        "Positive training samples" if classification_type == "multilabel" else "Training samples"
    ))
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "class_distribution.png"
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return [destination]


def _round_records(records: Sequence[MetricRecord]) -> dict[int, MetricRecord]:
    selected: dict[int, MetricRecord] = {}
    for record in records:
        number = record.round_number if record.round_number is not None else record.model_version
        previous = selected.get(number)
        if previous is None or record.model_version >= previous.model_version:
            selected[number] = record
    return dict(sorted(selected.items()))


def build_contribution_matrix(
    records: Sequence[MetricRecord], client_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    selected = _round_records(records)
    if not selected:
        return np.array([], dtype=int), np.empty((len(client_ids), 0))
    rounds = np.arange(min(1, min(selected)), max(selected) + 1)
    matrix = np.full((len(client_ids), len(rounds)), np.nan)
    for column, number in enumerate(rounds):
        record = selected.get(int(number))
        if record is None or record.aggregation_weights is None:
            continue
        weights = record.aggregation_weights
        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0 or set(weights) - set(client_ids):
            raise MetricsPlotError("invalid contribution weights or client registry")
        matrix[:, column] = [weights.get(client, 0) / total for client in client_ids]
    return rounds, matrix


def plot_contributions(experiment: Experiment, output_dir: Path) -> list[Path]:
    client_ids, _ = load_topology(experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for client, records in sorted(experiment.clients.items()):
        if not any(r.aggregation_weights is not None for r in records):
            print(f"warning: {experiment.label}: client {client} has no aggregation weights; "
                  "contribution chart omitted", file=sys.stderr)
            continue
        if any(r.aggregation_weights is None for r in records):
            print(f"warning: {experiment.label}: client {client} has missing aggregation "
                  "weights; those rounds are masked", file=sys.stderr)
        rounds, matrix = build_contribution_matrix(records, client_ids)
        figure, axis = plt.subplots(figsize=(12, max(4, len(client_ids) * 0.4)))
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("lightgray")
        display = axis.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                              vmin=0, vmax=1, interpolation="nearest",
                              extent=(rounds[0] - .5, rounds[-1] + .5,
                                      len(client_ids) - .5, -.5))
        axis.set_yticks(range(len(client_ids)), [f"Client {i}" for i in client_ids])
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("Round")
        axis.set_ylabel("Contributing client")
        axis.set_title(f"{experiment.label} — Client {client} — Direct aggregation contribution")
        figure.colorbar(display, ax=axis, label="Effective parameter mixing weight",
                        format=PercentFormatter(xmax=1))
        figure.tight_layout()
        path = output_dir / f"client_{client}_aggregation_contributions.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["round", *[f"client_{i}" for i in client_ids]])
            for column, number in enumerate(rounds):
                writer.writerow([number, *[value if np.isfinite(value) else ""
                                            for value in matrix[:, column]]])
        written.extend([path, csv_path])
    return written


def _class_names(experiment: Experiment, client: int) -> list[str] | None:
    for suffix in ("class_distribution", "confusion"):
        path = experiment.path / f"client_{client}_{suffix}.json"
        if not path.is_file():
            continue
        raw = _read_artifact(path)
        _check_identity(raw, experiment, path, client)
        labels = raw.get("labels")
        if (not isinstance(labels, list) or not labels
                or any(not isinstance(label, str) or not label for label in labels)
                or len(set(labels)) != len(labels)):
            raise MetricsPlotError(f"{path}: invalid class labels")
        return labels
    return None


def _recall_observations(experiment: Experiment, client: int) -> tuple[list[tuple], bool]:
    source = experiment.path / f"client_{client}_evaluations.jsonl"
    if not source.is_file():
        return [(number, r.model_version, r.values["test"])
                for number, r in _round_records(experiment.clients[client]).items()], False
    selected = {}
    with source.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            location = f"{source}:{index}"
            try:
                raw = json.loads(line)
            except ValueError as exc:
                raise MetricsPlotError(f"{location}: malformed evaluation") from exc
            if not isinstance(raw, dict):
                raise MetricsPlotError(f"{location}: invalid evaluation")
            _check_identity(raw, experiment, source, client)
            if raw.get("kind") not in {"periodic", "final"}:
                continue
            at = raw.get("captured_seconds")
            if (isinstance(at, bool) or not isinstance(at, (int, float))
                    or not math.isfinite(at) or at < 0):
                raise MetricsPlotError(f"{location}: invalid evaluation time")
            version = _parse_integer(raw.get("model_version"), "model_version", location)
            prior = selected.get(at)
            if prior is None or version >= prior[0]:
                selected[at] = (version, _parse_phase(raw, "test", location))
    return [(at, version, values) for at, (version, values) in sorted(selected.items())], True


def plot_class_recall(experiment: Experiment, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for client in sorted(experiment.clients):
        observations, timed = _recall_observations(experiment, client)
        indices = sorted({int(name[7:]) for _, _, values in observations for name in values
                          if re.fullmatch(r"recall_\d+", name)})
        if not indices:
            print(f"warning: {experiment.label}: client {client} has no per-class recall; "
                  "recall chart omitted", file=sys.stderr)
            continue
        labels = _class_names(experiment, client)
        if labels is not None and indices[-1] >= len(labels):
            raise MetricsPlotError(f"{experiment.path}: recall index outside class labels")
        figure, axis = plt.subplots(figsize=(11, 6))
        colors = plt.get_cmap("tab10" if len(indices) <= 10 else "tab20")
        csv_rows = []
        missing_support = False
        for index in indices:
            name = labels[index] if labels is not None else str(index)
            points = []
            for at, version, values in observations:
                recall, support = values.get(f"recall_{index}"), values.get(f"support_{index}")
                if recall is not None and not 0 <= recall <= 1:
                    raise MetricsPlotError("recall must be between zero and one")
                if support is not None and (support < 0 or support != int(support)):
                    raise MetricsPlotError("class support must be a nonnegative integer")
                usable = recall is not None and support is not None and support > 0
                missing_support |= recall is not None and support is None
                points.append(recall if usable else np.nan)
                csv_rows.append([at, version, index, name, support if support is not None else "",
                                 recall if usable else ""])
            axis.plot([at for at, _, _ in observations], points, label=name,
                      color=colors(index % colors.N), marker=".")
        if missing_support:
            print(f"warning: {experiment.label}: client {client} has recall without support; "
                  "those points are masked", file=sys.stderr)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Elapsed time (s)" if timed else "Round")
        if not timed:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_ylabel("Recall")
        axis.set_title(f"{experiment.label} — Client {client} — Recall by class")
        axis.grid(alpha=.3)
        axis.legend(title="Class", ncol=2, fontsize="small")
        figure.tight_layout()
        path = output_dir / f"client_{client}_recall_by_class.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["elapsed_seconds" if timed else "round", "model_version",
                             "class_index", "class_label", "support", "recall"])
            writer.writerows(sorted(csv_rows, key=lambda row: (row[0], row[2])))
        written.extend([path, csv_path])
    return written


def plot_training_diagnostics(experiment: Experiment, output_dir: Path) -> list[Path]:
    return plot_class_recall(experiment, output_dir) + plot_contributions(experiment, output_dir)


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
        written.extend(plot_class_distribution(experiment, directory))
        written.extend(plot_training_diagnostics(experiment, directory))
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
        "--show-neighbor-count", action="store_true",
        help="overlay contributing neighbor counts on individual test accuracy plots",
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
        written = plot_metrics(experiments, output_dir, args.x_axis, args.phase,
                               args.show_neighbor_count)
        written.extend(plot_final_metrics(experiments, output_dir,
                                          args.show_client_points))
    except (MetricsPlotError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {len(written)} artifact(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
