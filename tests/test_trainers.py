from types import ModuleType

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from app.config import Settings
from app.training import load_trainer, make_partitions
from experiments.cifar10 import trainer as cifar10_trainer
from experiments.mnist import trainer as mnist_trainer


def settings() -> Settings:
    return Settings(
        client_id=0,
        client_name="client-0",
        client_count=1,
        head_client_ids=(0,),
        neighbors=(),
        local_epochs=1,
        batch_size=2,
    )


def test_load_trainer_imports_built_in_module() -> None:
    loaded = load_trainer("experiments.mnist.trainer")

    assert loaded.create_model().__class__ is mnist_trainer.MnistCNN


def test_load_trainer_reports_import_failure() -> None:
    with pytest.raises(RuntimeError, match="could not import trainer module"):
        load_trainer("experiments.does_not_exist.trainer")


def test_load_trainer_reports_missing_contract(monkeypatch) -> None:
    incomplete = ModuleType("incomplete_trainer")
    incomplete.create_model = lambda: None
    monkeypatch.setattr(
        "app.training.importlib.import_module",
        lambda module_name: incomplete,
    )

    with pytest.raises(RuntimeError, match="load_data, train_local, evaluate"):
        load_trainer("incomplete_trainer")


@pytest.mark.parametrize(
    ("trainer", "image_shape"),
    [
        (mnist_trainer, (1, 28, 28)),
        (cifar10_trainer, (3, 32, 32)),
    ],
)
def test_classifier_train_and_evaluate_smoke(trainer, image_shape) -> None:
    config = settings()
    images = torch.zeros((2, *image_shape))
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    model = trainer.create_model()

    assert model(images).shape == (2, 10)

    state, train_metrics = trainer.train_local(
        model.state_dict(),
        loader,
        config,
    )
    test_metrics = trainer.evaluate(state, loader)

    assert set(train_metrics) == {"loss", "accuracy"}
    assert set(test_metrics) == {"loss", "accuracy"} | {
        f"{metric}_{i}" for metric in ("recall", "support") for i in range(10)
    }
    assert all(tensor.device.type == "cpu" for tensor in state.values())


def test_settings_reads_trainer_module(monkeypatch) -> None:
    monkeypatch.setenv("TRAINER_MODULE", "experiments.cifar10.trainer")
    monkeypatch.setenv("CLIENT_COUNT", "1")
    monkeypatch.setenv("HEAD_CLIENT_IDS", "0")

    assert Settings.from_env().trainer_module == "experiments.cifar10.trainer"


@pytest.mark.parametrize("trainer,dataset_name", [
    (mnist_trainer, "MNIST"), (cifar10_trainer, "CIFAR10"),
])
@pytest.mark.parametrize("client_count", [4, 8])
@pytest.mark.parametrize("capped", [False, True])
def test_load_data_selects_reproducible_dirichlet_partitions(
    monkeypatch, trainer, dataset_name, client_count, capped
) -> None:
    class SyntheticDataset(Dataset):
        def __init__(self, root, train, download, transform):
            labels = np.tile(np.arange(10), 240 if train else 20)
            self.targets = (
                torch.tensor(labels) if dataset_name == "MNIST" else labels.tolist()
            )

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.zeros(1), int(self.targets[index])

    monkeypatch.setattr(trainer.datasets, dataset_name, SyntheticDataset)
    labels = np.tile(np.arange(10), 240)
    candidates = np.arange(len(labels))
    if capped:
        candidates = np.random.default_rng(42).permutation(candidates)[:800]
    minimum = min(100, max(1, len(candidates) // client_count // 2))
    expected = make_partitions(
        labels[candidates], client_count, "dirichlet", 42,
        alpha=0.5, min_size=minimum,
    )
    test_indices = np.random.default_rng(42).permutation(200)[:40]
    selected = []
    for client_id in range(client_count):
        config = Settings(
            client_id=client_id,
            client_name=f"client-{client_id}",
            client_count=client_count,
            head_client_ids=(client_count - 1,),
            partition_mode="dirichlet",
            max_train_samples=800 if capped else 0,
            max_test_samples=40 if capped else 0,
        )
        bundle = trainer.load_data(config)
        repeated = trainer.load_data(config)
        indices = bundle.train_loader.dataset.indices
        assert indices == candidates[expected[client_id]].tolist()
        assert repeated.train_loader.dataset.indices == indices
        assert bundle.sample_count == len(indices) >= minimum
        distribution = bundle.class_distribution
        assert distribution["sample_count"] == len(indices)
        assert distribution["counts"] == np.bincount(labels[indices], minlength=10).tolist()
        selected.extend(indices)
        if capped:
            assert bundle.test_loader.dataset.indices == test_indices.tolist()
        else:
            assert len(bundle.test_loader.dataset) == 200
    assert len(selected) == len(set(selected)) == len(candidates)
    assert set(selected) == set(candidates.tolist())
