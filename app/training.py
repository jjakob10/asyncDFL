from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.config import Settings, parse_client_classes
from app.federation import StateDict


@dataclass
class DataBundle:
    train_loader: DataLoader | None
    test_loader: DataLoader
    sample_count: int
    class_distribution: dict[str, Any] | None = None


def class_distribution(
    labels: np.ndarray, class_names: list[str], multilabel: bool = False,
) -> dict[str, Any]:
    """Describe local training targets without loading input samples."""
    labels = np.asarray(labels)
    if multilabel:
        counts = labels.sum(axis=0).astype(np.int64)
    else:
        counts = np.bincount(labels.astype(np.int64), minlength=len(class_names))
    return {
        "classification_type": "multilabel" if multilabel else "multiclass",
        "labels": class_names,
        "counts": counts.tolist(),
        "sample_count": len(labels),
    }


class Trainer(Protocol):
    def create_model(self) -> nn.Module: ...

    def load_data(self, settings: Settings) -> DataBundle: ...

    def train_local(
        self,
        base_state: StateDict,
        train_loader: DataLoader,
        settings: Settings,
    ) -> tuple[StateDict, dict[str, float]]: ...

    def evaluate(
        self,
        state: StateDict,
        test_loader: DataLoader,
    ) -> dict[str, float]: ...


REQUIRED_TRAINER_FUNCTIONS = (
    "create_model",
    "load_data",
    "train_local",
    "evaluate",
)


def load_trainer(module_name: str) -> Trainer:
    """Import and validate a trainer module."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"could not import trainer module {module_name!r}: {exc}"
        ) from exc
    _validate_trainer(module, module_name)
    return cast(Trainer, module)


def _validate_trainer(module: ModuleType, module_name: str) -> None:
    missing = [
        function_name
        for function_name in REQUIRED_TRAINER_FUNCTIONS
        if not callable(getattr(module, function_name, None))
    ]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"trainer module {module_name!r} is missing required callable(s): {names}"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_partitions(
    labels: np.ndarray,
    client_count: int,
    mode: str,
    seed: int,
    alpha: float = 0.5,
    min_size: int = 100,
    max_attempts: int = 1_000,
    empty_client_ids: tuple[int, ...] = (),
    client_classes: dict[int, tuple[int, ...]] | None = None,
) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if client_count < 1:
        raise ValueError("client_count must be positive")
    empty_ids = set(empty_client_ids)
    if any(not 0 <= item < client_count for item in empty_ids):
        raise ValueError("empty client IDs must be valid client IDs")
    data_client_ids = [item for item in range(client_count) if item not in empty_ids]
    if not data_client_ids:
        raise ValueError("at least one client must have local data")
    data_client_count = len(data_client_ids)
    if labels.size < data_client_count * min_size:
        raise ValueError("dataset is too small for the requested minimum partition size")

    if mode == "manual":
        mapping = parse_client_classes(client_classes, client_count)
        if set(mapping) != set(range(client_count)):
            raise ValueError("CLIENT_CLASSES must specify every client")
        for client_id, classes in mapping.items():
            if bool(classes) == (client_id in empty_ids):
                raise ValueError("CLIENT_CLASSES must be empty only for empty clients")
        assigned = {label for classes in mapping.values() for label in classes}
        if assigned != set(np.unique(labels).tolist()):
            raise ValueError("CLIENT_CLASSES must cover exactly the available classes")
        rng = np.random.default_rng(seed)
        partitions: list[list[int]] = [[] for _ in range(client_count)]
        for label in sorted(assigned):
            owners = [item for item in data_client_ids if label in mapping[item]]
            indices = rng.permutation(np.flatnonzero(labels == label))
            if len(indices) < len(owners):
                raise ValueError(f"class {label} has too few samples for its assigned clients")
            for owner, part in zip(owners, np.array_split(indices, len(owners))):
                partitions[owner].extend(part.tolist())
        result = [rng.permutation(values).astype(np.int64) for values in partitions]
        if any(len(result[item]) < min_size for item in data_client_ids):
            raise ValueError("manual partition does not meet minimum partition size")
        return result

    if mode == "iid":
        rng = np.random.default_rng(seed)
        data_parts = [
            part.astype(np.int64)
            for part in np.array_split(rng.permutation(labels.size), data_client_count)
        ]
        result = [np.empty(0, dtype=np.int64) for _ in range(client_count)]
        for client_id, part in zip(data_client_ids, data_parts):
            result[client_id] = part
        return result
    if mode != "dirichlet":
        raise ValueError(f"unknown partition mode: {mode}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    classes = np.unique(labels)
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        partitions: list[list[int]] = [[] for _ in range(data_client_count)]
        for label in classes:
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            probabilities = rng.dirichlet(np.full(data_client_count, alpha))
            counts = rng.multinomial(len(indices), probabilities)
            cursor = 0
            for client_id, count in enumerate(counts):
                partitions[client_id].extend(
                    indices[cursor : cursor + count].tolist()
                )
                cursor += count
        if min(map(len, partitions)) >= min_size:
            data_parts = []
            for values in partitions:
                array = np.asarray(values, dtype=np.int64)
                rng.shuffle(array)
                data_parts.append(array)
            result = [np.empty(0, dtype=np.int64) for _ in range(client_count)]
            for client_id, part in zip(data_client_ids, data_parts):
                result[client_id] = part
            return result
    raise RuntimeError(
        f"could not create a Dirichlet partition with min_size={min_size}"
    )


class ConfusionTrainer(Protocol):
    """Optional trainer capability; not required by load_trainer."""

    def evaluate_confusion(
        self, state: StateDict, test_loader: DataLoader
    ) -> dict[str, Any]: ...


@torch.no_grad()
def classification_confusion(
    model: nn.Module,
    state: StateDict,
    test_loader: DataLoader,
    kind: str,
    labels: list[str],
) -> dict[str, Any]:
    """Count actual rows and predicted columns without retaining predictions."""
    model.load_state_dict(state)
    model.eval()
    count = len(labels)
    shape = (count, 2, 2) if kind == "multilabel" else (count, count)
    counts = torch.zeros(shape, dtype=torch.int64)
    total_seen = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        total_seen += targets.size(0)
        if kind == "multiclass":
            predicted = outputs.argmax(dim=1)
            indices = targets.long().reshape(-1) * count + predicted.reshape(-1)
            counts += torch.bincount(indices, minlength=count * count).reshape(shape)
        else:
            predicted = (torch.sigmoid(outputs) >= 0.5).long()
            actual = targets.long().reshape(predicted.shape)
            if kind == "multilabel":
                offsets = torch.arange(count).reshape(1, -1) * 4
                indices = (offsets + actual * 2 + predicted).reshape(-1)
                counts += torch.bincount(indices, minlength=count * 4).reshape(shape)
            else:
                indices = (actual * 2 + predicted).reshape(-1)
                counts += torch.bincount(indices, minlength=4).reshape(shape)
    return {
        "classification_type": kind,
        "labels": labels,
        "sample_count": total_seen,
        "counts": counts.tolist(),
    }
