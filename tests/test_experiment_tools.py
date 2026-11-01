import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app.config import Settings
from experiments.mnist import trainer as mnist
from experiments.cifar10 import trainer as cifar
from scripts.experiment_tools import ROOT, execution, prepare_data
from scripts.experiment_tools.catalog import build_catalog, load_manifest
from scripts.run_experiments import main


@pytest.mark.parametrize("trainer,name,factory", [(mnist, "mnist", "MNIST"), (cifar, "cifar10", "CIFAR10")])
def test_collection_opt_in_keeps_original_partition_and_test(trainer, name, factory, monkeypatch):
    class SyntheticDataset:
        def __init__(self, *args, **kwargs):
            self.targets = torch.tensor(np.tile(np.arange(10), 100))

        def __len__(self):
            return len(self.targets)

    monkeypatch.setattr(trainer.datasets, factory, SyntheticDataset)
    config = Settings(client_id=2, client_name="client-2", client_count=6, head_client_ids=(5,),
                      max_train_samples=400, max_test_samples=40)
    fixed = trainer.load_data(config)
    collected = trainer.load_data(replace(config, collection_mode="sliding_window"))
    assert fixed.collection_source is None
    assert fixed.train_loader.dataset.indices == collected.train_loader.dataset.indices
    assert fixed.test_loader.dataset.indices == collected.test_loader.dataset.indices
    source = collected.collection_source
    indices = fixed.train_loader.dataset.indices
    assert source.sample_ids == tuple(f"{name}:train:{i}" for i in indices)
    assert source.labels.tolist() == [int(SyntheticDataset().targets[i]) for i in indices]
    assert source.class_codes == trainer.COLLECTION_CLASS_CODES


@pytest.mark.parametrize("trainer,factory", [(mnist, "MNIST"), (cifar, "CIFAR10")])
def test_optional_preparation_downloads_both_splits(trainer, factory, tmp_path, monkeypatch):
    calls = []
    def dataset(*args, **kwargs):
        calls.append((args, kwargs))
    monkeypatch.setattr(trainer.datasets, factory, dataset)
    trainer.prepare_data(tmp_path)
    assert [kw for _, kw in calls] == [{"train": True, "download": True}, {"train": False, "download": True}]


def test_preparation_dispatch_has_no_dataset_registry(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prepare_data, "load_trainer", lambda module: SimpleNamespace(prepare_data=calls.append))
    directory = tmp_path / "new_dataset"
    prepare_data.prepare_dataset("external.dataset.trainer", directory)
    assert directory.is_dir() and calls == [directory]
    monkeypatch.setattr(prepare_data, "load_trainer", lambda module: SimpleNamespace())
    with pytest.raises(ValueError, match="must expose prepare_data"):
        prepare_data.prepare_dataset("external.dataset.trainer", directory)


def test_fixed_command_requires_manifest_and_rejects_collection(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["list"])
    assert error.value.code == 2
    manifest = ROOT / "experiments/thiago-coleta-simulada/manifest.json"
    assert main(["render-compose", "anything", "--manifest", str(manifest),
                 "--results-root", str(tmp_path / "results")]) == 2
    assert not (tmp_path / "results").exists()


def test_shared_lifecycle_saves_logs_and_cleans_up_after_start_failure(tmp_path, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "up" in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0, stdout="test logs", stderr="")
    monkeypatch.setattr(execution.subprocess, "run", run)
    spec = build_catalog(load_manifest(ROOT / "experiments/thiago/mnist/manifest.json"))[0]
    assert execution.run_compose(tmp_path / "compose.json", spec, tmp_path) == 2
    assert (tmp_path / "container.log").read_text() == "test logs"
    assert not json.loads((tmp_path / "outcome.json").read_text())["completed"]
    assert next(i for i, call in enumerate(calls) if "logs" in call) < next(i for i, call in enumerate(calls) if "down" in call)
