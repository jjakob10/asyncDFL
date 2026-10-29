import csv
import json

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.mnist import trainer as mnist
from experiments.cifar10 import trainer as cifar
from experiments.ton_iot import trainer as ton
from experiments.ecg import trainer as ecg
from scripts.plot_metrics import (
    MetricsPlotError, build_contribution_matrix, load_experiments, load_topology,
    neighbor_axis_maximum, plot_class_recall, plot_contributions, plot_metrics,
)


@pytest.mark.parametrize("trainer,inputs,targets,expected", [
    (mnist, [[9., 0., 0.], [0., 9., 0.], [0., 9., 0.]], [0, 0, 1], [.5, 1., 0.]),
    (cifar, [[9., 0., 0.], [0., 9., 0.], [0., 9., 0.]], [0, 0, 1], [.5, 1., 0.]),
    (ton, [-9., 9., 0.], [0., 0., 1.], [.5, 1.]),
    (ecg, [[-9., 9.], [9., -9.], [0., -9.]], [[0., 1.], [0., 1.], [1., 0.]], [1., .5]),
])
@pytest.mark.parametrize("empty", [False, True])
def test_recall_uses_existing_evaluation_pass(monkeypatch, trainer, inputs, targets, expected, empty):
    class CountingModel(nn.Identity):
        calls = 0

        def forward(self, value):
            CountingModel.calls += 1
            return value

    monkeypatch.setattr(trainer, "create_model", CountingModel)
    if trainer is ecg:
        monkeypatch.setattr(trainer, "CLASS_COUNT", 2)
    x, y = torch.tensor(inputs), torch.tensor(targets)
    if empty:
        x, y = x[:0], y[:0]
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    metrics = trainer.evaluate({}, loader)
    assert CountingModel.calls == (0 if empty else 2)
    for i, recall in enumerate(expected):
        assert metrics[f"recall_{i}"] == (0 if empty else recall)
        assert metrics[f"support_{i}"] >= 0
    assert all(np.isfinite(value) for value in metrics.values())


def write_run(path, counts=(2, 3, 4)):
    path.mkdir(parents=True, exist_ok=True)
    for client in range(5):
        neighbors = [f"http://client-{i}" for i in range(5) if i != client][:counts[client % len(counts)]]
        (path / f"client_{client}_topology.json").write_text(json.dumps({
            "client_id": client, "run_id": "run", "client_count": 5, "neighbors": neighbors,
        }))
        records = []
        for number in (1, 3):
            peers = ([{"client_id": (client + 1) % 5, "aggregation_weight": 3}]
                     if number == 1 else [])
            records.append({"client_id": client, "run_id": "run", "round": number,
                            "model_version": number, "time": "2026-10-28T00:00:00+00:00",
                            "aggregation_weight": 1, "participants": len(peers) + 1,
                            "neighbor_versions": peers,
                            "test": {"accuracy": .5, "recall_0": .5, "support_0": 2,
                                     "recall_1": 0., "support_1": 0}})
        (path / f"client_{client}_metrics.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records))
    return load_experiments([path])[0]


def test_all_accuracy_axes_share_configured_maximum(tmp_path, monkeypatch):
    experiment = write_run(tmp_path / "run")
    assert neighbor_axis_maximum([experiment]) == 4
    observed = []
    original = Axes.set_ylim

    def limits(axis, *args, **kwargs):
        result = original(axis, *args, **kwargs)
        if axis.get_ylabel() == "Contributing neighbors":
            observed.append(axis.get_ylim())
        return result

    monkeypatch.setattr(Axes, "set_ylim", limits)
    plot_metrics([experiment], tmp_path / "plots", "model_version", "test", True)
    assert observed.count((0., 4.)) == 5


def test_contribution_weights_masks_and_csv(tmp_path):
    experiment = write_run(tmp_path / "run")
    ids, _ = load_topology(experiment)
    assert experiment.clients[0][0].aggregation_weight == 1
    assert experiment.clients[0][0].neighbor_versions[0]["client_id"] == 1
    rounds, matrix = build_contribution_matrix(experiment.clients[0], ids)
    assert rounds.tolist() == [1, 2, 3]
    np.testing.assert_allclose(matrix[:, 0], [.25, .75, 0, 0, 0])
    assert np.isnan(matrix[:, 1]).all()
    np.testing.assert_allclose(matrix[:, 2], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(matrix[:, [0, 2]].sum(0), [1, 1])
    paths = plot_contributions(experiment, tmp_path / "plots")
    assert len(paths) == 10
    with paths[1].open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["client_0"] == "0.25"
    assert rows[1]["client_0"] == ""
    assert rows[2]["client_1"] == "0.0"


def test_recall_labels_zero_support_and_round_deduplication(tmp_path):
    experiment = write_run(tmp_path / "run")
    (experiment.path / "client_0_class_distribution.json").write_text(json.dumps({
        "client_id": 0, "run_id": "run", "labels": ["cat", "dog"],
    }))
    path = experiment.path / "client_0_metrics.jsonl"
    row = json.loads(path.read_text().splitlines()[0])
    row["test"]["recall_0"] = .75
    with path.open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    experiment = load_experiments([experiment.path])[0]
    paths = plot_class_recall(experiment, tmp_path / "plots")
    with paths[1].open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["recall"] == "0.75"
    assert rows[0]["class_label"] == "cat"
    assert rows[1]["support"] == "0.0"
    assert rows[1]["recall"] == ""


def test_timed_recall_uses_periodic_and_final_only(tmp_path):
    experiment = write_run(tmp_path / "run")
    path = experiment.path / "client_0_evaluations.jsonl"
    path.write_text("".join(json.dumps({
        "client_id": 0, "run_id": "run", "kind": kind, "captured_seconds": at,
        "model_version": version, "test": {"recall_0": value, "support_0": 2},
    }) + "\n" for kind, at, version, value in [
        ("periodic", .25, 0, .2), ("before_first_mix", 1, 1, .9),
        ("periodic", 2, 2, .3), ("final", 2, 3, .4),
    ]))
    paths = plot_class_recall(experiment, tmp_path / "plots")
    with paths[1].open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["elapsed_seconds"]) for row in rows] == [.25, 2]
    assert [float(row["recall"]) for row in rows] == [.2, .4]


def test_old_records_warn_without_inventing_weights(tmp_path, capsys):
    experiment = write_run(tmp_path / "run")
    for path in experiment.path.glob("*_topology.json"):
        path.unlink()
    path = experiment.path / "client_0_metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        del record["aggregation_weight"]
        record["test"] = {"accuracy": .5}
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    experiment = load_experiments([experiment.path])[0]
    assert neighbor_axis_maximum([experiment]) == 1
    assert all("client_0_" not in path.name for path in plot_contributions(experiment, tmp_path / "plots"))
    assert "estimated" in capsys.readouterr().err


@pytest.mark.parametrize("weight", [0, -1, True, "1", float("nan")])
def test_invalid_weights_rejected(tmp_path, weight):
    experiment = write_run(tmp_path / "run")
    path = experiment.path / "client_0_metrics.jsonl"
    record = json.loads(path.read_text().splitlines()[0])
    record["aggregation_weight"] = weight
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(MetricsPlotError, match="weight"):
        load_experiments([experiment.path])


def test_stale_topology_rejected(tmp_path):
    experiment = write_run(tmp_path / "run")
    path = experiment.path / "client_0_topology.json"
    raw = json.loads(path.read_text())
    raw["run_id"] = "old"
    path.write_text(json.dumps(raw))
    with pytest.raises(MetricsPlotError, match="run ID"):
        load_topology(experiment)


def test_resolved_topology_and_comparison_maximum(tmp_path):
    first = write_run(tmp_path / "first", counts=(1,))
    second = write_run(tmp_path / "second", counts=(4,))
    for path in second.path.glob("*_topology.json"):
        path.unlink()
    (second.path / "resolved_experiment.json").write_text(json.dumps({
        "run_id": "run", "client_count": 5, "clients": [
            {"client_id": client, "neighbors": [i for i in range(5) if i != client]}
            for client in range(5)
        ],
    }))
    assert neighbor_axis_maximum([first, second]) == 4


def test_isolated_axes_and_partial_topology_never_shrink_known_degree(tmp_path, monkeypatch):
    experiment = write_run(tmp_path / "run", counts=(0,))
    for path in experiment.path.glob("*_metrics.jsonl"):
        row = json.loads(path.read_text().splitlines()[1])
        path.write_text(json.dumps(row) + "\n")
    experiment = load_experiments([experiment.path])[0]
    assert neighbor_axis_maximum([experiment]) == 0
    original = Axes.set_yticks
    tick_calls = []

    def ticks(axis, values, *args, **kwargs):
        if axis.get_ylabel() == "Contributing neighbors":
            tick_calls.append(values)
        return original(axis, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_yticks", ticks)
    plot_metrics([experiment], tmp_path / "plots", "model_version", "test", True)
    assert tick_calls == [[0]] * 5
    source = experiment.path / "client_0_topology.json"
    raw = json.loads(source.read_text())
    raw["neighbors"] = [f"http://client-{i}" for i in range(1, 5)]
    source.write_text(json.dumps(raw))
    for client in range(1, 5):
        (experiment.path / f"client_{client}_topology.json").unlink()
    assert neighbor_axis_maximum([experiment]) == 4


def test_evaluations_only_and_no_completed_cycles_remain_plottable(tmp_path):
    from scripts.plot_metrics import main

    path = tmp_path / "run"
    path.mkdir()
    (path / "client_0_metrics.jsonl").write_text("")
    (path / "client_0_evaluations.jsonl").write_text(json.dumps({
        "client_id": 0, "run_id": "run", "kind": "final", "captured_seconds": 1,
        "model_version": 0, "test": {"recall_0": 1., "support_0": 2},
    }) + "\n")
    (path / "resolved_experiment.json").write_text(json.dumps({"run_id": "run"}))
    assert main([str(path), str(tmp_path / "plots")]) == 0
    assert (tmp_path / "plots/client_0_recall_by_class.png").is_file()


def test_evaluation_only_mixed_runs_are_rejected(tmp_path):
    for client, run_id in enumerate(("first", "second")):
        (tmp_path / f"client_{client}_evaluations.jsonl").write_text(json.dumps({
            "client_id": client, "run_id": run_id, "kind": "final",
            "model_version": 0, "captured_seconds": 1,
            "test": {"recall_0": 1., "support_0": 2},
        }) + "\n")
    with pytest.raises(MetricsPlotError, match="multiple run IDs"):
        load_experiments([tmp_path])
