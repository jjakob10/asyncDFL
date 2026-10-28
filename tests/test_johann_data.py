from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from app.config import Settings
from experiments.johann.catalog import SCENARIOS, compose_for, load_spec, topology, validate_topology
from experiments.johann.data import index_hash, quantitative_partitions, split_training


@pytest.mark.parametrize("seed", [7, 42, 43])
def test_partition_concentration_and_capacity(seed):
    labels = np.repeat(np.arange(10), [541, 604, 552, 613, 525, 540, 592, 626, 585, 622])
    training, validation = split_training(labels, seed)
    assert not set(training) & set(validation)
    assert sorted([*training, *validation]) == list(range(len(labels)))
    assert abs(len(validation) / len(labels) - .1) < .001
    assert np.array_equal(training, split_training(labels, seed)[0])
    for distribution in ("distributed", "concentrated"):
        parts = quantitative_partitions(labels[training], distribution, seed)
        assert np.ptp([len(p) for p in parts]) <= 1
        assert sorted(np.concatenate(parts)) == list(range(len(training)))
        repeated = quantitative_partitions(labels[training], distribution, seed)
        assert all(np.array_equal(a, b) for a, b in zip(parts, repeated))
        for label in (3, 8):
            count = np.count_nonzero(labels[training] == label)
            shares = [np.count_nonzero(labels[training][parts[c]] == label) for c in (0, 4)]
            target = .4 if distribution == "concentrated" else 1 / 8
            assert all(abs(n / count - target) < .01 for n in shares)


def test_partitions_reject_infeasible_or_missing_classes():
    labels = np.r_[np.arange(10), np.full(1000, 3), np.full(1000, 8)]
    with pytest.raises(ValueError, match="infeasible"):
        quantitative_partitions(labels, "concentrated", 7)
    with pytest.raises(ValueError, match="all ten"):
        quantitative_partitions(np.arange(9), "distributed", 7)
    with pytest.raises(ValueError):
        split_training(np.arange(10), 7)


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_adapter_uses_same_global_data_with_separate_validation(dataset, monkeypatch):
    from experiments.johann import trainer as shared

    class SyntheticDataset(Dataset):
        classes = [str(i) for i in range(10)]

        def __init__(self, *args, train=True, **kwargs):
            self.targets = np.tile(np.arange(10), 100 if train else 20)

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.zeros(1, 28, 28), int(self.targets[index])

    monkeypatch.setattr(shared.datasets, "MNIST" if dataset == "mnist" else "CIFAR10", SyntheticDataset)
    trainer = import_module(f"experiments.johann.{dataset}.trainer")
    settings = Settings(client_id=0, client_name="client-0", max_train_samples=800, max_test_samples=50)
    global_hashes, validation_hashes = set(), set()
    selected_sets = []
    for distribution in ("distributed", "concentrated"):
        indices = []
        for client in range(8):
            data = trainer.load_data(replace(settings, client_id=client, johann_distribution=distribution))
            info = data.class_distribution
            global_hashes.add(info["training_indices_sha256"])
            validation_hashes.add(info["validation_indices_sha256"])
            indices.extend(info["local_indices"])
            assert data.sample_count == 90
            assert len(data.test_loader.dataset) == 50
            assert sum(info["counts"]) == data.sample_count
        assert len(set(indices)) == 720
        selected_sets.append(set(indices))
    assert len(global_hashes) == len(validation_hashes) == 1
    assert selected_sets[0] == selected_sets[1]
    # Test mode reads the official test dataset; validation hash remains unchanged.
    test = trainer.load_data(replace(settings, evaluation_split="test"))
    assert test.class_distribution["validation_indices_sha256"] in validation_hashes


def test_topology_and_compose_defaults(tmp_path):
    edges = topology()
    assert len(edges) == 12
    assert all(sum(client in edge for edge in edges) == 3 for client in range(8))
    validate_topology(edges)
    with pytest.raises(ValueError, match="disconnected"):
        validate_topology(tuple((i, (i + 1) % 8) for i in range(8)))
    for dataset in ("mnist", "cifar10"):
        for scenario in SCENARIOS:
            spec = load_spec(dataset, scenario)
            compose = compose_for(spec, "test", tmp_path)
            assert len(compose["services"]) == 8
            for service in compose["services"].values():
                env = service["environment"]
                assert env["AUTO_START_HEAD"] == "false"
                assert env["EVALUATION_SPLIT"] == "validation"
                assert len(env["NEIGHBORS"].split(",")) == 3
    assert load_spec("mnist", "concentrated-r", mode="collection")["evaluation_split"] == "test"


@pytest.mark.parametrize("changes", [dict(duration=0), dict(duration=float("nan")), dict(interval=100)])
def test_invalid_budget_is_rejected(changes):
    with pytest.raises(ValueError):
        load_spec("mnist", "concentrated-r", **changes)
