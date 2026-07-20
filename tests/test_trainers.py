from types import ModuleType

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import Settings
from app.training import load_trainer
from experiments.cifar10 import trainer as cifar10_trainer
from experiments.mnist import trainer as mnist_trainer


def settings() -> Settings:
    return Settings(
        client_id=0,
        client_name="client-0",
        client_count=1,
        head_client_id=0,
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
    assert set(test_metrics) == {"loss", "accuracy"}
    assert all(tensor.device.type == "cpu" for tensor in state.values())


def test_settings_reads_trainer_module(monkeypatch) -> None:
    monkeypatch.setenv("TRAINER_MODULE", "experiments.cifar10.trainer")
    monkeypatch.setenv("CLIENT_COUNT", "1")
    monkeypatch.setenv("HEAD_CLIENT_ID", "0")

    assert Settings.from_env().trainer_module == "experiments.cifar10.trainer"
