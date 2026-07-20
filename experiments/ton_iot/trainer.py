from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import Settings
from app.federation import StateDict
from app.training import DataBundle, make_partitions


_FEATURE_COUNT = 42


class TonIotNetwork(nn.Module):
    def __init__(self, feature_count: int | None = None) -> None:
        super().__init__()
        size = feature_count or _FEATURE_COUNT
        self.network = nn.Sequential(
            nn.Linear(size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)


def create_model() -> nn.Module:
    return TonIotNetwork()


def _read_dataset(settings: Settings) -> pd.DataFrame:
    if not settings.ton_iot_path:
        raise ValueError("TON_IOT_PATH must point to Train_Test_Network.csv")
    path = Path(settings.ton_iot_path)
    if not path.is_file():
        raise FileNotFoundError(f"TON_IOT dataset does not exist: {path}")
    frame = pd.read_csv(path)
    required = {"ts", "type", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"TON_IOT dataset is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("TON_IOT dataset is empty")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
        raise ValueError("TON_IOT label must contain only binary values 0 and 1")
    frame = frame.copy()
    frame["label"] = labels.astype(np.int64)
    return frame


def _cap_frame(
    features: pd.DataFrame,
    labels: pd.Series,
    limit: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if not limit or limit >= len(features):
        return features, labels
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(features), size=limit, replace=False))
    return features.iloc[selected].reset_index(drop=True), labels.iloc[
        selected
    ].reset_index(drop=True)


def _prepare_arrays(
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    frame = _read_dataset(settings)
    labels = frame.pop("label")
    frame = frame.drop(columns=["ts", "type"])
    train_features, test_features, train_labels, test_labels = train_test_split(
        frame,
        labels,
        test_size=0.2,
        random_state=settings.seed,
        stratify=labels,
    )
    train_features, train_labels = _cap_frame(
        train_features,
        train_labels,
        settings.max_train_samples,
        settings.seed,
    )
    test_features, test_labels = _cap_frame(
        test_features,
        test_labels,
        settings.max_test_samples,
        settings.seed + 1,
    )
    numeric_columns = list(
        train_features.select_dtypes(include=[np.number, "bool"]).columns
    )
    categorical_columns = [
        column for column in train_features.columns if column not in numeric_columns
    ]
    if not numeric_columns or not categorical_columns:
        raise ValueError(
            "TON_IOT preprocessing requires numeric and categorical features"
        )
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_columns),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            ),
        ]
    )
    train_array = transformer.fit_transform(train_features).astype(np.float32)
    test_array = transformer.transform(test_features).astype(np.float32)
    if not np.isfinite(train_array).all() or not np.isfinite(test_array).all():
        raise ValueError("TON_IOT preprocessing produced non-finite values")
    metadata: dict[str, object] = {
        "source": str(Path(settings.ton_iot_path).resolve()),
        "seed": settings.seed,
        "train_rows": len(train_array),
        "test_rows": len(test_array),
        "input_features": int(train_array.shape[1]),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
    return (
        train_array,
        train_labels.to_numpy(dtype=np.float32),
        test_array,
        test_labels.to_numpy(dtype=np.float32),
        metadata,
    )


def load_data(settings: Settings) -> DataBundle:
    global _FEATURE_COUNT
    train_x, train_y, test_x, test_y, metadata = _prepare_arrays(settings)
    _FEATURE_COUNT = int(train_x.shape[1])
    partitions = make_partitions(
        train_y.astype(np.int64),
        settings.client_count,
        settings.partition_mode,
        settings.seed,
        settings.dirichlet_alpha,
        min(
            settings.min_partition_size,
            max(1, len(train_y) // max(1, settings.client_count) // 2),
        ),
        empty_client_ids=settings.empty_client_ids,
    )
    local_indices = partitions[settings.client_id]
    train_loader = None
    if len(local_indices):
        train_dataset = TensorDataset(
            torch.from_numpy(train_x[local_indices]),
            torch.from_numpy(train_y[local_indices]),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(
                settings.seed + settings.client_id
            ),
        )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
        batch_size=max(settings.batch_size, 256),
        shuffle=False,
        num_workers=0,
    )
    metadata_path = (
        Path(settings.results_dir)
        / f"preprocessing_client_{settings.client_id}.json"
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return DataBundle(train_loader, test_loader, len(local_indices))


def train_local(
    base_state: StateDict,
    train_loader: DataLoader,
    settings: Settings,
) -> tuple[StateDict, dict[str, float]]:
    model = create_model()
    model.load_state_dict(base_state)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for _ in range(settings.local_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                (torch.sigmoid(logits) >= 0.5) == targets.bool()
            ).sum().item()
            total_seen += batch_size
    return (
        {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        {
            "loss": total_loss / max(total_seen, 1),
            "accuracy": total_correct / max(total_seen, 1),
        },
    )


@torch.inference_mode()
def evaluate(state: StateDict, test_loader: DataLoader) -> dict[str, float]:
    model = create_model()
    model.load_state_dict(state)
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for inputs, targets in test_loader:
        logits = model(inputs)
        total_loss += criterion(logits, targets).item()
        total_correct += (
            (torch.sigmoid(logits) >= 0.5) == targets.bool()
        ).sum().item()
        total_seen += targets.size(0)
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }
