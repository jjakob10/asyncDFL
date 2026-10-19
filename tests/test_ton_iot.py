from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from experiments.ton_iot import trainer


def write_dataset(path: Path) -> None:
    rows = []
    for index in range(100):
        rows.append(
            {
                "ts": index,
                "type": "normal" if index % 2 == 0 else "scan",
                "label": index % 2,
                "duration": float(index),
                "bytes": float(index * 2),
                "service": ["http", "dns", "ssh"][index % 3],
                "protocol": ["tcp", "udp"][index % 2],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def settings(path: Path, tmp_path: Path) -> Settings:
    return Settings(
        client_id=0,
        client_name="client-0",
        client_count=2,
        head_client_ids=(1,),
        ton_iot_path=str(path),
        results_dir=str(tmp_path),
        batch_size=8,
        min_partition_size=1,
        max_train_samples=40,
        max_test_samples=20,
    )


def test_ton_iot_preprocessing_and_model_smoke(tmp_path: Path) -> None:
    path = tmp_path / "Train_Test_Network.csv"
    write_dataset(path)

    data = trainer.load_data(settings(path, tmp_path))
    model = trainer.create_model()
    inputs, _ = next(iter(data.test_loader))

    assert data.class_distribution["sample_count"] == 20
    assert sum(data.class_distribution["counts"]) == 20
    assert data.sample_count == 20
    assert model(inputs).shape == (len(inputs),)
    assert (tmp_path / "preprocessing_client_0.json").is_file()


def test_ton_iot_rejects_missing_schema(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"label": [0, 1], "value": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        trainer.load_data(settings(path, tmp_path))
