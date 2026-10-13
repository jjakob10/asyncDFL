from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyfive
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from app.config import Settings
from app.federation import StateDict
from app.training import DataBundle, make_partitions


INPUT_CHANNELS = 12
INPUT_LENGTH = 5_000
CLASS_COUNT = 20
REQUIRED_COLUMNS = {"ECG_ID", "Code_Label", "Location"}
INTERNAL_RECORDS_COLUMN = "_records_path"
INTERNAL_LABELS_COLUMN = "_label_indices"
INTERNAL_PARTITION_COLUMN = "_partition_label"


def _canonical_labels(value: object, source: Path, row_number: int) -> tuple[int, ...]:
    if pd.isna(value):
        raise ValueError(
            f"{source}: row {row_number} has an empty Code_Label value"
        )
    text = str(value).strip()
    if not text:
        raise ValueError(
            f"{source}: row {row_number} has an empty Code_Label value"
        )
    try:
        labels = tuple(sorted(int(item.strip()) for item in text.split(";")))
    except ValueError as exc:
        raise ValueError(
            f"{source}: row {row_number} has invalid Code_Label {text!r}"
        ) from exc
    if len(set(labels)) != len(labels):
        raise ValueError(
            f"{source}: row {row_number} has duplicate diagnosis codes"
        )
    if any(label < 0 or label >= CLASS_COUNT for label in labels):
        raise ValueError(
            f"{source}: row {row_number} has a diagnosis code outside "
            f"0..{CLASS_COUNT - 1}"
        )
    return labels


def _read_split(root: Path, split: str) -> pd.DataFrame:
    if not root.is_dir():
        raise FileNotFoundError(f"ECG data directory does not exist: {root}")
    location_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not location_dirs:
        raise FileNotFoundError(f"no ECG location directories found under {root}")

    frames: list[pd.DataFrame] = []
    for location_dir in location_dirs:
        csv_path = location_dir / f"{split}.csv"
        records_path = location_dir / "records.h5"
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing ECG metadata file: {csv_path}")
        if not records_path.is_file():
            raise FileNotFoundError(f"missing ECG records file: {records_path}")

        frame = pd.read_csv(csv_path, dtype={"ECG_ID": str})
        missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{csv_path}: missing required column(s): {missing}")
        if frame["ECG_ID"].isna().any():
            raise ValueError(f"{csv_path}: ECG_ID must not contain missing values")

        locations = frame["Location"].astype(str)
        if not locations.eq(location_dir.name).all():
            raise ValueError(
                f"{csv_path}: every Location value must equal "
                f"{location_dir.name!r}"
            )

        parsed_labels = [
            _canonical_labels(value, csv_path, row_number)
            for row_number, value in enumerate(frame["Code_Label"], start=2)
        ]
        prepared = frame.copy()
        prepared["ECG_ID"] = prepared["ECG_ID"].astype(str)
        prepared[INTERNAL_RECORDS_COLUMN] = str(records_path)
        prepared[INTERNAL_LABELS_COLUMN] = parsed_labels
        prepared[INTERNAL_PARTITION_COLUMN] = [
            ";".join(str(label) for label in labels) for labels in parsed_labels
        ]
        frames.append(prepared)

    return pd.concat(frames, ignore_index=True)


class ECGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy reader for FedCVD's preprocessed ECG records."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._files: dict[str, pyfive.File] = {}
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        records_path = str(row[INTERNAL_RECORDS_COLUMN])
        ecg_id = str(row["ECG_ID"])
        records = self._files.get(records_path)
        if records is None:
            try:
                records = pyfive.File(records_path, "r")
            except Exception as exc:
                raise OSError(
                    f"could not open ECG records file: {records_path} "
                    f"for ECG record {ecg_id!r}"
                ) from exc
            self._files[records_path] = records

        try:
            inputs = np.asarray(records[ecg_id][...], dtype=np.float32)
        except KeyError as exc:
            raise KeyError(
                f"{records_path}: missing ECG record {ecg_id!r}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"{records_path}: could not read ECG record {ecg_id!r}"
            ) from exc
        if inputs.shape != (INPUT_CHANNELS, INPUT_LENGTH):
            raise ValueError(
                f"{records_path}: ECG record {ecg_id!r} has shape "
                f"{inputs.shape}, expected {(INPUT_CHANNELS, INPUT_LENGTH)}"
            )
        if not np.isfinite(inputs).all():
            raise ValueError(
                f"{records_path}: ECG record {ecg_id!r} contains non-finite values"
            )

        targets = torch.zeros(CLASS_COUNT, dtype=torch.float32)
        targets[list(row[INTERNAL_LABELS_COLUMN])] = 1.0
        return torch.from_numpy(inputs), targets

    def close(self) -> None:
        files = getattr(self, "_files", {})
        opened = list(files.values())
        files.clear()
        for records in opened:
            records.close()

    def __del__(self) -> None:
        self.close()


def _conv7(
    input_channels: int,
    output_channels: int,
    stride: int = 1,
) -> nn.Conv1d:
    return nn.Conv1d(
        input_channels,
        output_channels,
        kernel_size=7,
        stride=stride,
        padding=3,
        bias=False,
    )


def _conv1(
    input_channels: int,
    output_channels: int,
    stride: int = 1,
) -> nn.Conv1d:
    return nn.Conv1d(
        input_channels,
        output_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv7(input_channels, output_channels, stride)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv7(output_channels, output_channels)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(outputs + residual)


class ECGResNet34(nn.Module):
    """FedCVD's 1D ResNet-34 without its final sigmoid."""

    def __init__(self) -> None:
        super().__init__()
        self._current_channels = 64
        self.conv1 = nn.Conv1d(
            INPUT_CHANNELS,
            64,
            kernel_size=15,
            stride=2,
            padding=7,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, CLASS_COUNT)

    def _make_layer(
        self,
        output_channels: int,
        block_count: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self._current_channels != output_channels:
            downsample = nn.Sequential(
                _conv1(self._current_channels, output_channels, stride),
                nn.BatchNorm1d(output_channels),
            )
        blocks = [
            BasicBlock(
                self._current_channels,
                output_channels,
                stride,
                downsample,
            )
        ]
        self._current_channels = output_channels
        blocks.extend(
            BasicBlock(self._current_channels, output_channels)
            for _ in range(1, block_count)
        )
        return nn.Sequential(*blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.maxpool(self.relu(self.bn1(self.conv1(inputs))))
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs)
        return self.fc(torch.flatten(outputs, 1))


def create_model() -> nn.Module:
    return ECGResNet34()


def _capped_indices(size: int, limit: int, seed: int) -> np.ndarray:
    indices = np.arange(size, dtype=np.int64)
    if limit and limit < size:
        return np.random.default_rng(seed).permutation(indices)[:limit]
    return indices


def load_data(settings: Settings) -> DataBundle:
    root = Path(settings.data_dir) / "ECG" / "preprocessed"
    train_frame = _read_split(root, "train")
    test_frame = _read_split(root, "test")
    train_dataset = ECGDataset(train_frame)
    test_dataset: Dataset = ECGDataset(test_frame)

    candidate_indices = _capped_indices(
        len(train_frame),
        settings.max_train_samples,
        settings.seed,
    )
    partition_keys = train_frame.iloc[candidate_indices][
        INTERNAL_PARTITION_COLUMN
    ].to_numpy(dtype=str)
    _, partition_labels = np.unique(partition_keys, return_inverse=True)
    partitions = make_partitions(
        partition_labels,
        settings.client_count,
        settings.partition_mode,
        settings.seed,
        settings.dirichlet_alpha,
        min(
            settings.min_partition_size,
            max(1, len(partition_labels) // settings.client_count // 2),
        ),
        empty_client_ids=settings.empty_client_ids,
    )
    local_indices = candidate_indices[partitions[settings.client_id]]
    train_subset = Subset(train_dataset, local_indices.tolist())

    test_indices = _capped_indices(
        len(test_frame),
        settings.max_test_samples,
        settings.seed,
    )
    if len(test_indices) != len(test_frame):
        test_dataset = Subset(test_dataset, test_indices.tolist())

    train_loader = None
    if len(train_subset):
        train_loader = DataLoader(
            train_subset,
            batch_size=settings.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(
                settings.seed + settings.client_id
            ),
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=max(settings.batch_size, 256),
        shuffle=False,
        num_workers=0,
    )
    return DataBundle(train_loader, test_loader, len(train_subset))


def train_local(
    base_state: StateDict,
    train_loader: DataLoader,
    settings: Settings,
) -> tuple[StateDict, dict[str, float]]:
    model = create_model()
    model.load_state_dict(base_state)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=settings.learning_rate,
        momentum=settings.momentum,
    )
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    total_elements = 0

    for _ in range(settings.local_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * targets.numel()
            total_correct += (
                (torch.sigmoid(logits) >= 0.5)
                .eq(targets.bool())
                .all(dim=1)
                .sum()
                .item()
            )
            total_seen += targets.size(0)
            total_elements += targets.numel()

    return (
        {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        {
            "loss": float(total_loss / max(total_elements, 1)),
            "accuracy": float(total_correct / max(total_seen, 1)),
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
    total_elements = 0

    for inputs, targets in test_loader:
        logits = model(inputs)
        total_loss += criterion(logits, targets).item()
        total_correct += (
            (torch.sigmoid(logits) >= 0.5)
            .eq(targets.bool())
            .all(dim=1)
            .sum()
            .item()
        )
        total_seen += targets.size(0)
        total_elements += targets.numel()

    return {
        "loss": float(total_loss / max(total_elements, 1)),
        "accuracy": float(total_correct / max(total_seen, 1)),
    }
