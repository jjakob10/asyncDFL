import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Settings
from app.runtime import ClientRuntime
from app.training import classification_confusion
from experiments.mnist import trainer as mnist
from experiments.cifar10 import trainer as cifar
from tests.test_runtime import TinyTrainer, tiny_data


@pytest.mark.parametrize("trainer,kind,inputs,targets,expected", [
    (mnist, "multiclass", [[9, 0, 0], [0, 9, 0], [0, 9, 0]],
     [0, 0, 1], [[1, 1, 0], [0, 1, 0], [0, 0, 0]]),
    (cifar, "multiclass", [[9, 0, 0], [0, 9, 0], [0, 9, 0]],
     [0, 0, 1], [[1, 1, 0], [0, 1, 0], [0, 0, 0]]),
])
def test_trainer_confusion_counts(monkeypatch, trainer, kind, inputs, targets, expected):
    monkeypatch.setattr(trainer, "create_model", nn.Identity)
    loader = DataLoader(TensorDataset(torch.tensor(inputs, dtype=torch.float32),
                                    torch.tensor(targets)), batch_size=2)
    result = trainer.evaluate_confusion({}, loader)
    assert result["classification_type"] == kind
    assert result["sample_count"] == 3
    counts = np.asarray(result["counts"])
    if kind == "multiclass":
        assert counts.shape == (10, 10)
        assert counts[:3, :3].tolist() == expected
        assert counts.sum() == 3
    else:
        assert result["counts"] == expected


class ConfusionTinyTrainer(TinyTrainer):
    def __init__(self):
        super().__init__()
        self.final_state = None
        self.confusion_calls = 0

    def evaluate_confusion(self, state, loader):
        self.final_state = state
        self.confusion_calls += 1
        return {"classification_type": "binary", "labels": ["0", "1"],
                "sample_count": 4, "counts": [[3, 1], [0, 0]]}


@pytest.mark.parametrize("empty", [False, True])
def test_final_confusion_is_atomic_and_matches_checkpoint(tmp_path, empty):
    trainer = ConfusionTinyTrainer()
    config = Settings(client_id=0, client_name="client-0", neighbors=(),
                      rounds=2, results_dir=str(tmp_path))
    data = tiny_data()
    if empty:
        data.train_loader = None
        data.sample_count = 0
    runtime = ClientRuntime(config, data=data, trainer=trainer)
    runtime.start("run")
    runtime.wait(5)
    assert runtime.status()["state"] == "completed"
    artifact = tmp_path / "client_0_confusion.json"
    raw = json.loads(artifact.read_text())
    assert raw["run_id"] == "run"
    assert raw["model_version"] == 2
    saved = torch.load(tmp_path / "client_0_final.pt", weights_only=True)
    assert all(torch.equal(saved[key], value) for key, value in trainer.final_state.items())
    original = artifact.read_bytes()
    assert runtime.start("run") == "already_started"
    assert trainer.confusion_calls == 1
    assert artifact.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))
    rows = [json.loads(line) for line in (tmp_path / "client_0_metrics.jsonl").read_text().splitlines()]
    assert all(row["started_at"] == runtime.status()["started_at"] for row in rows)


def test_confusion_write_failure_marks_run_failed(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr("app.runtime.json.dump", fail)
    runtime = ClientRuntime(Settings(client_id=0, client_name="client-0", neighbors=(),
                                    rounds=1, results_dir=str(tmp_path)),
                            trainer=ConfusionTinyTrainer())
    runtime.start("run")
    runtime.wait(5)
    assert runtime.status()["state"] == "failed"
    assert not (tmp_path / "client_0_confusion.json").exists()


def test_legacy_trainer_warns_and_completes(tmp_path, caplog):
    runtime = ClientRuntime(Settings(client_id=0, client_name="client-0", neighbors=(),
                                    rounds=1, results_dir=str(tmp_path)), trainer=TinyTrainer())
    runtime.start("run")
    runtime.wait(5)
    assert runtime.status()["state"] == "completed"
    assert "no evaluate_confusion" in caplog.text


def test_confusion_uses_last_neighbor_aggregated_model(tmp_path, monkeypatch):
    trainer = ConfusionTinyTrainer()
    runtime = ClientRuntime(
        Settings(client_id=0, client_name="client-0", neighbors=("http://peer",),
                 rounds=2, results_dir=str(tmp_path)), trainer=trainer,
    )
    initial = {key: value.clone() for key, value in runtime._model.items()}
    monkeypatch.setattr(runtime, "_forward_start", lambda: None)
    monkeypatch.setattr(runtime, "_fetch_neighbor", lambda neighbor, state: (
        {key: value + 2 for key, value in state.items()}, 4, {}
    ))
    runtime.start("run")
    runtime.wait(5)
    assert runtime.status()["state"] == "completed"
    for key, value in initial.items():
        torch.testing.assert_close(trainer.final_state[key], value + 2)


def test_confusion_evaluation_failure_marks_run_failed(tmp_path):
    trainer = ConfusionTinyTrainer()
    def fail(state, loader):
        raise ValueError("invalid predictions")
    trainer.evaluate_confusion = fail
    runtime = ClientRuntime(
        Settings(client_id=0, client_name="client-0", neighbors=(),
                 rounds=1, results_dir=str(tmp_path)), trainer=trainer,
    )
    runtime.start("run")
    runtime.wait(5)
    assert runtime.status()["state"] == "failed"
    assert "invalid predictions" in runtime.status()["recent_errors"][-1]["message"]


def test_no_gradient_tracking_and_empty_test_set():
    class CheckIdentity(nn.Module):
        def forward(self, inputs):
            assert not torch.is_grad_enabled()
            assert not self.training
            return inputs
    loader = DataLoader(TensorDataset(torch.tensor([0.0]), torch.tensor([1.0])))
    assert classification_confusion(CheckIdentity(), {}, loader, "binary", ["0", "1"])["counts"] == [[0, 0], [0, 1]]
    empty = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)))
    assert classification_confusion(CheckIdentity(), {}, empty, "binary", ["0", "1"])["counts"] == [[0, 0], [0, 0]]
