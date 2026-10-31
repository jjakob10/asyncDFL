import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from app.collection import CollectionSource, resolve_plan
from app.training import DataBundle
from experiments.simulated_collection.catalog import (
    ROOT, client_settings, collection_compose, load_catalog,
)
from experiments.simulated_collection.images import load_collection_data
from experiments.simulated_collection.validate import validate_stocks
from scripts.run_collection_experiments import main, render
from scripts.experiment_compose import build_compose
from experiments.paper.catalog import build_catalog, load_manifest


@pytest.mark.parametrize("smoke,rounds", [(False, 10), (True, 5)])
def test_six_examples_same_pair_stock_schedule_and_six_client_ring(tmp_path, smoke, rounds):
    catalog = load_catalog(smoke=smoke)
    assert len(catalog) == 6
    assert {item.spec.dataset for item in catalog} == {"mnist", "cifar10", "harth"}
    for left, right in zip(catalog[::2], catalog[1::2]):
        assert left.mode == "accumulative" and right.mode == "sliding_window"
        assert left.schedule == right.schedule
        assert left.spec.seed == right.spec.seed == 42
        for item in (left, right):
            assert item.spec.rounds == rounds
            assert item.window_rounds == 3
            assert item.spec.client_count == 6
            compose = collection_compose(item, "test", tmp_path)
            assert len(compose["services"]) == 6
            for client in item.spec.clients:
                assert set(client.neighbors) == {(client.client_id-1) % 6, (client.client_id+1) % 6}
                env = compose["services"][f"client-{client.client_id}"]["environment"]
                assert env["ROUNDS"] == str(rounds)
                assert env["MAX_TRAIN_SAMPLES"] == "0"
                assert env["COLLECTION_MODE"] == item.mode
                client_settings(item, client.client_id).validate()


def test_stock_preflight_all_clients_and_no_partial_run(tmp_path, monkeypatch):
    experiment = load_catalog(smoke=True)[0]
    directory = render(experiment, "check", tmp_path)
    resolved = json.loads((directory / "resolved_experiment.json").read_text())
    class FakeTrainer:
        def load_data(self, settings):
            labels = np.tile(np.arange(10), 10)
            data = TensorDataset(torch.zeros(100, 1), torch.tensor(labels))
            source = CollectionSource(data, labels, tuple(f"{settings.client_id}:{i}" for i in range(100)), tuple(range(10)))
            return DataBundle(None, DataLoader(data), 100, collection_source=source)
    monkeypatch.setattr("experiments.simulated_collection.validate.load_trainer", lambda _: FakeTrainer())
    result = validate_stocks(resolved, directory)
    assert len(result["clients"]) == 6
    assert len(set(result["clients"].values())) == 6
    class TooSmall(FakeTrainer):
        def load_data(self, settings):
            data = super().load_data(settings)
            if settings.client_id == 5:
                data.collection_source.labels[:] = 0
            return data
    monkeypatch.setattr("experiments.simulated_collection.validate.load_trainer", lambda _: TooSmall())
    with pytest.raises(ValueError, match="client 5 round"):
        validate_stocks(resolved, directory)
    with pytest.raises(FileExistsError):
        render(experiment, "check", tmp_path)


def test_preflight_rejects_overlapping_stocks(tmp_path, monkeypatch):
    item = load_catalog(smoke=True)[0]
    directory = render(item, "overlap", tmp_path)
    resolved = json.loads((directory / "resolved_experiment.json").read_text())
    labels = np.tile(np.arange(10), 10)
    data = TensorDataset(torch.zeros(100, 1), torch.tensor(labels))
    source = CollectionSource(data, labels, tuple(map(str, range(100))), tuple(range(10)))
    class FakeTrainer:
        def load_data(self, settings):
            return DataBundle(None, DataLoader(data), 100, collection_source=source)
    monkeypatch.setattr("experiments.simulated_collection.validate.load_trainer", lambda _: FakeTrainer())
    with pytest.raises(ValueError, match="overlap"):
        validate_stocks(resolved, directory)


def test_render_and_cli_selection(tmp_path, capsys):
    assert main(["list", "--dataset", "harth"]) == 0
    assert capsys.readouterr().out.count("rounds=10") == 2
    assert main(["render-compose", "--all", "--results-root", str(tmp_path), "--run-id", "render"]) == 0
    assert len(list(tmp_path.glob("*/render/compose.json"))) == 6
    assert main(["run", "unknown"]) == 1
    assert main(["render-compose", "--all", "--results-root", str(tmp_path), "--run-id", "../escape"]) == 1


def test_image_adapter_preserves_original_loader_and_stable_original_indices(tmp_path):
    dataset = TensorDataset(torch.zeros(10, 1), torch.arange(10))
    dataset.targets = torch.arange(10)
    subset = Subset(dataset, [7, 0, 3])
    loader = DataLoader(subset)
    bundle = DataBundle(loader, loader, 3)
    class FakeTrainer:
        def load_data(self, settings):
            return bundle
    config = client_settings(load_catalog()[0], 0)
    result = load_collection_data(FakeTrainer(), "mnist", config)
    assert result.train_loader is loader and result.test_loader is loader
    assert result.collection_source.labels.tolist() == [7, 0, 3]
    assert result.collection_source.sample_ids == ("mnist:train:7", "mnist:train:0", "mnist:train:3")


def test_existing_catalogs_and_compose_have_no_collection_configuration(tmp_path):
    manifests = [None, ROOT / "experiments/thiago/mnist/manifest.json",
                 ROOT / "experiments/thiago/cifar10/manifest.json"]
    for path in manifests:
        catalog = build_catalog(load_manifest(path) if path else None)
        for item in catalog:
            assert "coleta" not in item.experiment_id
            compose = build_compose(item, "regression", tmp_path,
                                    ton_iot_path=tmp_path / "ton.csv")
            for service in compose["services"].values():
                assert not any(key.startswith("COLLECTION_") for key in service["environment"])
