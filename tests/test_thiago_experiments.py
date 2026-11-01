import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Settings, parse_client_classes
from app.runtime import ClientRuntime
from app.training import DataBundle, class_distribution, make_partitions
from scripts.experiment_tools.catalog import build_catalog, load_manifest
from scripts.plot_metrics import (
    MetricsPlotError, load_experiments, plot_class_distribution, plot_metrics,
)
from scripts.run_experiments import main as runner_main
from scripts.experiment_tools.execution import render_compose
from scripts.experiment_tools.reporting import summarize_run


ROOT = Path(__file__).resolve().parents[1]
SHARED = (2, 5, 6, 7, 8, 9)
MAPPING = {item: SHARED for item in range(6)}
MAPPING[1] = (0, 1, *SHARED)
MAPPING[4] = (3, 4)


def test_manual_partition_preserves_data_and_requested_classes() -> None:
    labels = np.tile(np.arange(10), 101)
    parts = make_partitions(labels, 6, "manual", 42, min_size=1, client_classes=MAPPING)
    repeated = make_partitions(labels, 6, "manual", 42, min_size=1, client_classes=MAPPING)
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(labels)))
    for client, part in enumerate(parts):
        assert set(labels[part]) == set(MAPPING[client])
        assert np.array_equal(part, repeated[client])
    for label in range(10):
        counts = [np.count_nonzero(labels[parts[client]] == label)
                  for client in range(6) if label in MAPPING[client]]
        assert max(counts) - min(counts) <= 1
    changed = make_partitions(labels, 6, "manual", 43, min_size=1, client_classes=MAPPING)
    assert any(not np.array_equal(left, right) for left, right in zip(parts, changed))


@pytest.mark.parametrize("mapping,message", [
    ({0: (0,), 1: (1,)}, "every client"),
    ({**MAPPING, 1: (0, 0, 1, *SHARED)}, "duplicate classes"),
    ({**MAPPING, 6: (0,)}, "invalid or duplicate client"),
    ({**MAPPING, 1: (0, *SHARED)}, "available classes"),
    ({**MAPPING, 1: (0, 1, 10, *SHARED)}, "available classes"),
    ({**MAPPING, 0: ()}, "empty only"),
    ({**MAPPING, 0: (True,)}, "nonnegative integers"),
])
def test_manual_partition_rejects_invalid_map(mapping, message) -> None:
    with pytest.raises(ValueError, match=message):
        make_partitions(np.tile(np.arange(10), 100), 6, "manual", 42,
                        min_size=1, client_classes=mapping)


def test_manual_partition_rejects_impossible_class_presence_or_minimum() -> None:
    with pytest.raises(ValueError, match="too few samples"):
        make_partitions(np.arange(10), 6, "manual", 42, min_size=1, client_classes=MAPPING)
    with pytest.raises(ValueError, match="minimum partition size"):
        make_partitions(np.tile(np.arange(10), 100), 6, "manual", 42,
                        min_size=150, client_classes=MAPPING)


def test_manual_supports_explicit_empty_clients() -> None:
    parts = make_partitions(np.tile([0, 1], 10), 3, "manual", 42, min_size=1,
                            client_classes={0: (0,), 1: (), 2: (1,)}, empty_client_ids=(1,))
    assert list(map(len, parts)) == [10, 0, 10]


def test_manual_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CLIENT_COUNT", "6")
    monkeypatch.setenv("HEAD_CLIENT_IDS", "5")
    monkeypatch.setenv("PARTITION_MODE", "manual")
    monkeypatch.setenv("CLIENT_CLASSES", json.dumps(MAPPING))
    assert Settings.from_env().client_classes == MAPPING
    monkeypatch.setenv("CLIENT_CLASSES", "{}")
    with pytest.raises(ValueError, match="every client"):
        Settings.from_env()
    with pytest.raises(ValueError, match="JSON object"):
        parse_client_classes([], 6)


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_thiago_manifest_and_compose(dataset, tmp_path) -> None:
    manifest = ROOT / "experiments" / "thiago" / dataset / "manifest.json"
    catalog = build_catalog(load_manifest(manifest))
    assert len(catalog) == 6
    assert {(item.topology_name, item.partition_mode) for item in catalog} == {
        (topology, mode) for topology in ("ring", "star") for mode in ("iid", "dirichlet", "manual")
    }
    for experiment in catalog:
        assert experiment.client_count == 6
        assert experiment.rounds == 10 and experiment.local_epochs == 1
        assert experiment.batch_size == 64 and experiment.learning_rate == 0.01
        assert experiment.seed == 42 and experiment.momentum == 0.9
        assert experiment.dirichlet_alpha == 0.5 and experiment.min_partition_size == 100
        assert 4 not in experiment.clients[1].neighbors
        degrees = [len(client.neighbors) for client in experiment.clients]
        assert degrees == ([2] * 6 if experiment.topology_name == "ring" else [5, 1, 1, 1, 1, 1])
        assert all(client.has_local_data for client in experiment.clients)
        if experiment.partition_mode == "manual":
            assert experiment.client_classes == MAPPING
        compose_path, results = render_compose(
            experiment, "test", tmp_path / f"{experiment.experiment_id}.json", tmp_path / "results",
        )
        compose = json.loads(compose_path.read_text())
        assert len(compose["services"]) == 6
        for client_id, service in enumerate(compose["services"].values()):
            env = service["environment"]
            assert env["PARTITION_MODE"] == experiment.partition_mode
            assert env["DIRICHLET_ALPHA"] == "0.5"
            assert env["MIN_PARTITION_SIZE"] == "100"
            if experiment.partition_mode == "manual":
                assert parse_client_classes(json.loads(env["CLIENT_CLASSES"]), 6) == MAPPING
            assert service["ports"] == [f"{8000 + client_id}:9000"]
        resolved = json.loads((results / "resolved_experiment.json").read_text())
        assert resolved["partition_mode"] == experiment.partition_mode
    assert runner_main(["list", "--manifest", str(manifest)]) == 0
    assert runner_main(["render-compose", catalog[-1].experiment_id, "--manifest", str(manifest),
                        "--output", str(tmp_path / "cli.json"),
                        "--results-root", str(tmp_path / "cli-results")]) == 0


def write_result(directory: Path, peers=True) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for version, count in [(2, 0), (1, 2)]:
        record = {"client_id": 0, "run_id": "test", "round": version,
                  "model_version": version, "time": f"2026-10-18T12:00:0{version}+00:00",
                  "test": {"accuracy": 0.5 + version / 10, "loss": 0.3},
                  "train": {"accuracy": 0.8}}
        if peers:
            record["neighbor_versions"] = [{}] * count
            record["participants"] = 99  # neighbor_versions takes precedence
        records.append(record)
    (directory / "client_0_metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )


@pytest.mark.parametrize("x_axis,expected_x", [("model_version", [1, 2]), ("time", [0, 1])])
def test_neighbor_overlay_is_optional_and_aligned(tmp_path, monkeypatch, x_axis, expected_x) -> None:
    write_result(tmp_path)
    experiments = load_experiments([tmp_path])
    figures = []
    original = plt.subplots

    def capture(*args, **kwargs):
        figure, axis = original(*args, **kwargs)
        figures.append(figure)
        return figure, axis

    monkeypatch.setattr(plt, "subplots", capture)
    plot_metrics(experiments, tmp_path / "plain", x_axis, "both")
    assert all(len(figure.axes) == 1 for figure in figures)
    figures.clear()
    plot_metrics(experiments, tmp_path / "overlay", x_axis, "both", True)
    accuracy = next(figure for figure in figures if "accuracy" in figure.axes[0].get_title())
    assert len(accuracy.axes) == 2
    line = accuracy.axes[1].lines[0]
    assert list(line.get_xdata()) == expected_x
    assert list(line.get_ydata()) == [2, 0]
    assert accuracy.axes[1].get_ylim()[1] >= 2
    assert all(len(figure.axes) == 1 for figure in figures if "loss" in figure.axes[0].get_title())
    figures.clear()
    plot_metrics(experiments, tmp_path / "train", x_axis, "train", True)
    assert all(len(figure.axes) == 1 for figure in figures)


def test_neighbor_legacy_counts_and_missing_observations(tmp_path, capsys) -> None:
    write_result(tmp_path, peers=False)
    path = tmp_path / "client_0_metrics.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["participants"] = 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    experiments = load_experiments([tmp_path])
    assert [record.neighbor_count for record in experiments[0].clients[0]] == [0, None]
    plot_metrics(experiments, tmp_path / "charts", "model_version", "test", True)
    assert "missing neighbor counts" in capsys.readouterr().err


def test_class_distribution_heatmap_and_summary(tmp_path) -> None:
    write_result(tmp_path)
    distribution = class_distribution(np.array([0, 1, 1]), ["zero", "one"])
    distribution.update(client_id=0, run_id="test")
    source = tmp_path / "client_0_class_distribution.json"
    source.write_text(json.dumps(distribution))
    experiment = load_experiments([tmp_path])[0]
    charts = plot_class_distribution(experiment, tmp_path / "charts")
    assert charts[0].name == "class_distribution.png"
    assert charts[0].stat().st_size > 0
    assert any(path.name == "class_distribution.png" for path in summarize_run(tmp_path)[2])
    distribution["run_id"] = "stale"
    source.write_text(json.dumps(distribution))
    with pytest.raises(MetricsPlotError, match="run ID"):
        plot_class_distribution(experiment, tmp_path / "charts")


def test_class_distribution_multilabel_and_empty() -> None:
    distribution = class_distribution(np.array([[1, 1], [0, 1]]), ["a", "b"], True)
    assert distribution["counts"] == [1, 2]
    assert distribution["sample_count"] == 2
    assert class_distribution(np.array([], dtype=int), ["a", "b"])["counts"] == [0, 0]


def test_runtime_writes_training_distribution_before_rounds(tmp_path) -> None:
    class Trainer:
        def create_model(self):
            return nn.Linear(1, 2)

        def evaluate(self, state, loader):
            assert (tmp_path / "client_0_class_distribution.json").is_file()
            return {"accuracy": 0.5}

    loader = DataLoader(TensorDataset(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)))
    bundle = DataBundle(None, loader, 0, class_distribution(np.array([], dtype=int), ["0", "1"]))
    config = Settings(client_id=0, client_name="client-0", rounds=1, results_dir=str(tmp_path))
    runtime = ClientRuntime(config, bundle, Trainer())
    runtime.start("test")
    runtime.wait(timeout=5)
    assert runtime.status()["state"] == "completed"
    raw = json.loads((tmp_path / "client_0_class_distribution.json").read_text())
    assert raw["counts"] == [0, 0] and raw["sample_count"] == 0
    assert raw["client_id"] == 0 and raw["run_id"] == "test"


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_image_manual_metadata_matches_capped_training_data(dataset, monkeypatch) -> None:
    from importlib import import_module

    trainer = import_module(f"experiments.{dataset}.trainer")
    dataset_name = "MNIST" if dataset == "mnist" else "CIFAR10"

    class SyntheticDataset:
        def __init__(self, *args, **kwargs):
            labels = np.tile(np.arange(10), 100)
            self.targets = torch.tensor(labels) if dataset == "mnist" else labels.tolist()

        def __len__(self):
            return len(self.targets)

    monkeypatch.setattr(trainer.datasets, dataset_name, SyntheticDataset)
    config = Settings(
        client_id=0, client_name="client-0", client_count=6, head_client_ids=(5,),
        partition_mode="manual", client_classes=MAPPING, max_train_samples=400,
    )
    selected = []
    for client in range(6):
        bundle = trainer.load_data(replace(config, client_id=client))
        indices = bundle.train_loader.dataset.indices
        targets = np.asarray(bundle.train_loader.dataset.dataset.targets)[indices]
        assert set(targets) == set(MAPPING[client])
        assert bundle.class_distribution["counts"] == np.bincount(targets, minlength=10).tolist()
        assert bundle.class_distribution["sample_count"] == len(indices) == bundle.sample_count
        selected.extend(indices)
    assert len(selected) == len(set(selected)) == 400
    empty = trainer.load_data(replace(
        config, empty_client_ids=(0,), client_classes={**MAPPING, 0: ()},
    ))
    assert empty.train_loader is None and empty.sample_count == 0
    assert empty.class_distribution["counts"] == [0] * 10


@pytest.mark.parametrize("command", ["run", "smoke"])
def test_manifest_execution_commands_use_results_root_without_starting_containers(
    command, tmp_path, monkeypatch,
) -> None:
    from scripts import run_experiments as runner

    data = load_manifest(ROOT / "experiments/thiago/mnist/manifest.json")
    data["results_root"] = str(tmp_path / "results")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    calls = []

    def fake_run(compose_path, experiment, run_directory, show_client_points):
        compose = json.loads(compose_path.read_text())
        env = compose["services"]["client-0"]["environment"]
        assert env["ROUNDS"] == ("1" if command == "smoke" else "10")
        assert env["MAX_TRAIN_SAMPLES"] == ("400" if command == "smoke" else "0")
        assert env["PARTITION_MODE"] == experiment.partition_mode
        assert run_directory.is_relative_to(tmp_path / "results")
        assert compose_path.is_relative_to(tmp_path / "results" / ".generated")
        calls.append(experiment.experiment_id)
        return 0

    monkeypatch.setattr(runner, "run_compose", fake_run)
    assert runner.main([command, "--all", "--manifest", str(manifest), "--run-id", "test"]) == 0
    assert len(calls) == len(set(calls)) == 6


@pytest.mark.parametrize("changes", [
    {"counts": [1, -1]}, {"counts": [1]}, {"sample_count": 5},
    {"labels": ["one", "one"]}, {"client_id": 1},
])
def test_class_distribution_rejects_inconsistent_artifacts(tmp_path, changes) -> None:
    write_result(tmp_path)
    raw = class_distribution(np.array([0, 1, 1]), ["zero", "one"])
    raw.update(client_id=0, run_id="test")
    raw.update(changes)
    (tmp_path / "client_0_class_distribution.json").write_text(json.dumps(raw))
    with pytest.raises(MetricsPlotError):
        plot_class_distribution(load_experiments([tmp_path])[0], tmp_path / "charts")
