import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Settings
from app.runtime import ClientRuntime
from experiments.mnist import trainer as mnist
from experiments.cifar10 import trainer as cifar
from experiments.ecg import trainer as ecg
from experiments.ton_iot import trainer as ton
from scripts.plot_metrics import (
    MetricsPlotError, build_combined_series, load_experiments,
    plot_final_metrics,
)
from tests.test_runtime import TinyTrainer, tiny_data


@pytest.mark.parametrize("trainer,kind,inputs,targets,expected", [
    (mnist, "multiclass", [[9, 0, 0], [0, 9, 0], [0, 9, 0]],
     [0, 0, 1], [[1, 1, 0], [0, 1, 0], [0, 0, 0]]),
    (cifar, "multiclass", [[9, 0, 0], [0, 9, 0], [0, 9, 0]],
     [0, 0, 1], [[1, 1, 0], [0, 1, 0], [0, 0, 0]]),
    (ton, "binary", [-9, 9, 0], [0, 0, 1], [[1, 1], [0, 1]]),
    (ecg, "multilabel", [[-9, 9], [9, -9], [0, -9]],
     [[0, 1], [0, 1], [1, 0]], [[[1, 1], [0, 1]], [[1, 0], [1, 1]]]),
])
def test_trainer_confusion_counts(monkeypatch, trainer, kind, inputs, targets, expected):
    monkeypatch.setattr(trainer, "create_model", nn.Identity)
    if trainer is ecg:
        monkeypatch.setattr(trainer, "CLASS_COUNT", 2)
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


def write_client(path: Path, client: int, observations, run="run", start=True):
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for version, second, value in observations:
        row = {"client_id": client, "model_version": version,
               "time": f"2026-10-14T00:00:{second:02d}+00:00", "run_id": run,
               "test": {"accuracy": value, "loss": 1 - value}}
        if start:
            row["started_at"] = "2026-10-14T00:00:00+00:00"
        rows.append(json.dumps(row))
    (path / f"client_{client}_metrics.jsonl").write_text("\n".join(rows) + "\n")


def test_interpolation_population_std_and_variable_participants(tmp_path):
    write_client(tmp_path, 0, [(1, 10, 0), (2, 30, 1)])
    write_client(tmp_path, 1, [(1, 20, 1), (2, 40, 0)])
    write_client(tmp_path, 2, [(1, 25, 0)])
    series = build_combined_series(load_experiments([tmp_path])[0], "accuracy")
    assert series.time.tolist() == [10, 20, 25, 30, 40]
    assert series.participants.tolist() == [1, 2, 3, 2, 1]
    np.testing.assert_allclose(series.mean, [0, 0.75, 0.5, 0.75, 0])
    np.testing.assert_allclose(series.std, [0, 0.25, np.sqrt(0.125), 0.25, 0])


def test_duplicate_timestamps_choose_highest_version_then_last_line(tmp_path, capsys):
    write_client(tmp_path, 0, [(3, 10, 0.4), (1, 10, 0.1),
                              (3, 10, 0.8), (4, 20, 1)], start=False)
    series = build_combined_series(load_experiments([tmp_path])[0], "accuracy")
    assert series.time.tolist() == [0, 10]
    assert series.mean.tolist() == [0.8, 1]
    assert "fallback" in capsys.readouterr().err


def test_rejects_mixed_runs_and_invalid_run_id(tmp_path):
    write_client(tmp_path, 0, [(1, 10, 0.5)])
    write_client(tmp_path, 1, [(1, 10, 0.5)], run="another")
    with pytest.raises(MetricsPlotError, match="multiple run IDs"):
        load_experiments([tmp_path])
    write_client(tmp_path, 1, [(1, 10, 0.5)], run=[])
    with pytest.raises(MetricsPlotError, match="run_id"):
        load_experiments([tmp_path])


def test_combined_outputs_confusions_and_client_legend(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_client(source, 0, [(1, 10, 0.5), (2, 30, 0.8)])
    write_client(source, 1, [(1, 20, 0.4), (2, 40, 0.7)])
    (source / "client_0_confusion.json").write_text(json.dumps({
        "client_id": 0, "run_id": "run", "model_version": 2,
        "classification_type": "binary", "labels": ["0", "1"],
        "sample_count": 4, "counts": [[2, 1], [0, 1]],
    }))
    from matplotlib.axes import Axes
    original = Axes.legend
    legends = []
    def legend(axis, *args, **kwargs):
        legends.append(axis.get_legend_handles_labels()[1])
        return original(axis, *args, **kwargs)
    monkeypatch.setattr(Axes, "legend", legend)
    written = plot_final_metrics(load_experiments([source]), tmp_path / "charts", True)
    assert len(written) == 5
    assert all(p.stat().st_size > 0 for p in written)
    assert all("Client 0" in labels and "Client 1" in labels for labels in legends)
    with (tmp_path / "charts/combined_test_accuracy_time.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["participants"]) for r in rows] == [1, 2, 2, 1]


def test_multiple_experiments_have_separate_output_directories(tmp_path):
    sources = [tmp_path / "a/run", tmp_path / "b/run"]
    for source in sources:
        write_client(source, 0, [(1, 10, 0.5)])
    written = plot_final_metrics(load_experiments(sources), tmp_path / "plots")
    assert len({p.parent for p in written}) == 2
    assert len(written) == 8


def test_multilabel_panel_and_stale_artifact_rejection(tmp_path):
    write_client(tmp_path, 0, [(1, 10, 0.5)])
    raw = {"client_id": 0, "run_id": "run", "model_version": 1,
           "classification_type": "multilabel", "labels": [str(i) for i in range(20)],
           "sample_count": 4, "counts": [[[2, 1], [0, 1]]] * 20}
    path = tmp_path / "client_0_confusion.json"
    path.write_text(json.dumps(raw))
    experiment = load_experiments([tmp_path])[0]
    from scripts.plot_metrics import plot_confusions
    assert len(plot_confusions(experiment, tmp_path / "charts")) == 1
    raw["run_id"] = "stale"
    path.write_text(json.dumps(raw))
    with pytest.raises(MetricsPlotError, match="run ID"):
        plot_confusions(experiment, tmp_path / "charts")


def test_no_gradient_tracking_and_empty_test_set(monkeypatch):
    class CheckIdentity(nn.Module):
        def forward(self, inputs):
            assert not torch.is_grad_enabled()
            assert not self.training
            return inputs
    monkeypatch.setattr(ton, "create_model", CheckIdentity)
    loader = DataLoader(TensorDataset(torch.tensor([0.0]), torch.tensor([1.0])))
    assert ton.evaluate_confusion({}, loader)["counts"] == [[0, 0], [0, 1]]
    empty = DataLoader(TensorDataset(torch.empty(0), torch.empty(0)))
    assert ton.evaluate_confusion({}, empty)["counts"] == [[0, 0], [0, 0]]
