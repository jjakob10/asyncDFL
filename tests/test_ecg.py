from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from app.config import Settings
from app.federation import validate_compatible
from app.training import load_trainer
from experiments.ecg import trainer


def _settings(
    data_dir: Path,
    *,
    client_id: int = 0,
    partition_mode: str = "iid",
    empty_client_ids: tuple[int, ...] = (),
    max_train_samples: int = 0,
    max_test_samples: int = 0,
) -> Settings:
    return Settings(
        client_id=client_id,
        client_name=f"client-{client_id}",
        client_count=2,
        head_client_ids=(1,),
        empty_client_ids=empty_client_ids,
        neighbors=(),
        local_epochs=1,
        batch_size=2,
        learning_rate=0.01,
        momentum=0.9,
        seed=17,
        partition_mode=partition_mode,
        dirichlet_alpha=10.0,
        min_partition_size=1,
        data_dir=str(data_dir),
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
    )


def _write_ecg_data(
    data_dir: Path,
    *,
    locations: int = 2,
    train_rows: int = 12,
    test_rows: int = 2,
    sample_shape: tuple[int, int] = (12, 5_000),
    nonfinite: bool = False,
    omit_first_test_record: bool = False,
) -> Path:
    root = data_dir / "ECG" / "preprocessed"
    for location_index in range(locations):
        location = f"client{location_index + 1}"
        location_dir = root / location
        location_dir.mkdir(parents=True)
        train_metadata: list[dict[str, object]] = []
        test_metadata: list[dict[str, object]] = []
        with h5py.File(location_dir / "records.h5", "w") as records:
            for split, row_count, metadata in (
                ("train", train_rows, train_metadata),
                ("test", test_rows, test_metadata),
            ):
                for row_index in range(row_count):
                    ecg_id = (
                        f"{location_index + 1}{0 if split == 'train' else 1}"
                        f"{row_index:04d}"
                    )
                    metadata.append(
                        {
                            "ECG_ID": ecg_id,
                            "Code_Label": (
                                "0;1" if row_index % 2 == 0 else "2"
                            ),
                            "Location": location,
                        }
                    )
                    if (
                        omit_first_test_record
                        and location_index == 0
                        and split == "test"
                        and row_index == 0
                    ):
                        continue
                    sample = np.zeros(sample_shape, dtype=np.float16)
                    if (
                        nonfinite
                        and location_index == 0
                        and split == "test"
                        and row_index == 0
                    ):
                        sample[0, 0] = np.nan
                    records.create_dataset(ecg_id, data=sample)
        pd.DataFrame(train_metadata).to_csv(
            location_dir / "train.csv",
            index=False,
        )
        pd.DataFrame(test_metadata).to_csv(
            location_dir / "test.csv",
            index=False,
        )
    return root


class TinyECGModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(12, 20)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(inputs).squeeze(-1))


def test_load_trainer_imports_ecg_module() -> None:
    loaded = load_trainer("experiments.ecg.trainer")

    assert isinstance(loaded.create_model(), trainer.ECGResNet34)


def test_ecg_model_produces_raw_multilabel_logits() -> None:
    model = trainer.create_model().eval()
    inputs = torch.zeros((2, 12, 64))

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (2, 20)
    assert not any(isinstance(module, nn.Sigmoid) for module in model.modules())


def test_train_and_evaluate_return_compatible_cpu_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trainer, "create_model", TinyECGModel)
    config = _settings(tmp_path)
    inputs = torch.zeros((2, 12, 64))
    targets = torch.zeros((2, 20))
    targets[0, [0, 1]] = 1
    targets[1, 2] = 1
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2)
    initial_state = trainer.create_model().state_dict()

    state, train_metrics = trainer.train_local(initial_state, loader, config)
    test_metrics = trainer.evaluate(state, loader)

    validate_compatible(initial_state, state)
    assert all(tensor.device.type == "cpu" for tensor in state.values())
    assert set(train_metrics) == {"loss", "accuracy"}
    assert set(test_metrics) == {"loss", "accuracy"}
    assert all(math.isfinite(value) for value in train_metrics.values())
    assert all(math.isfinite(value) for value in test_metrics.values())


@pytest.mark.parametrize("partition_mode", ["iid", "dirichlet"])
def test_load_data_is_deterministic_and_uses_common_test_set(
    tmp_path: Path,
    partition_mode: str,
) -> None:
    _write_ecg_data(tmp_path)
    config = _settings(
        tmp_path,
        partition_mode=partition_mode,
        max_train_samples=12,
        max_test_samples=3,
    )

    first = trainer.load_data(config)
    second = trainer.load_data(config)

    assert first.train_loader is not None
    assert second.train_loader is not None
    assert isinstance(first.train_loader.dataset, Subset)
    assert first.train_loader.dataset.indices == second.train_loader.dataset.indices
    assert first.sample_count == len(first.train_loader.dataset)
    assert len(first.test_loader.dataset) == 3
    inputs, targets = next(iter(first.test_loader))
    assert inputs.shape[1:] == (12, 5_000)
    assert targets.shape[1:] == (20,)
    assert targets.dtype == torch.float32


def test_empty_client_has_no_training_loader(tmp_path: Path) -> None:
    _write_ecg_data(tmp_path)

    bundle = trainer.load_data(
        _settings(
            tmp_path,
            client_id=0,
            empty_client_ids=(0,),
            max_train_samples=12,
        )
    )

    assert bundle.train_loader is None
    assert bundle.sample_count == 0
    assert len(bundle.test_loader.dataset) == 4


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("train.csv", "missing ECG metadata file"),
        ("records.h5", "missing ECG records file"),
    ],
)
def test_load_data_rejects_missing_source_file(
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    root = _write_ecg_data(tmp_path)
    (root / "client1" / filename).unlink()

    with pytest.raises(FileNotFoundError, match=message):
        trainer.load_data(_settings(tmp_path))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("Code_Label", "invalid", "invalid Code_Label"),
        ("Code_Label", "20", "outside 0..19"),
        ("Code_Label", "1;1", "duplicate diagnosis codes"),
        ("Location", "unknown", "every Location value"),
    ],
)
def test_load_data_rejects_invalid_metadata(
    tmp_path: Path,
    column: str,
    value: str,
    message: str,
) -> None:
    root = _write_ecg_data(tmp_path)
    csv_path = root / "client1" / "train.csv"
    frame = pd.read_csv(csv_path, dtype={"ECG_ID": str})
    frame.loc[0, column] = value
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=message):
        trainer.load_data(_settings(tmp_path))


def test_load_data_rejects_missing_required_column(tmp_path: Path) -> None:
    root = _write_ecg_data(tmp_path)
    csv_path = root / "client1" / "train.csv"
    frame = pd.read_csv(csv_path, dtype={"ECG_ID": str}).drop(
        columns=["Code_Label"]
    )
    frame.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required column.*Code_Label"):
        trainer.load_data(_settings(tmp_path))


@pytest.mark.parametrize(
    ("fixture_options", "message"),
    [
        ({"sample_shape": (12, 4_999)}, "has shape"),
        ({"nonfinite": True}, "contains non-finite values"),
        ({"omit_first_test_record": True}, "missing ECG record"),
    ],
)
def test_dataset_rejects_invalid_hdf5_records(
    tmp_path: Path,
    fixture_options: dict[str, object],
    message: str,
) -> None:
    _write_ecg_data(tmp_path, **fixture_options)
    bundle = trainer.load_data(_settings(tmp_path))

    with pytest.raises((KeyError, ValueError), match=message):
        next(iter(bundle.test_loader))
