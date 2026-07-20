from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from app.config import Settings
from app.federation import StateDict


@dataclass
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    sample_count: int


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
) -> list[np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if client_count < 1:
        raise ValueError("client_count must be positive")
    if labels.size < client_count * min_size:
        raise ValueError("dataset is too small for the requested minimum partition size")

    if mode == "iid":
        rng = np.random.default_rng(seed)
        return [
            part.astype(np.int64)
            for part in np.array_split(rng.permutation(labels.size), client_count)
        ]
    if mode != "dirichlet":
        raise ValueError(f"unknown partition mode: {mode}")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    classes = np.unique(labels)
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        partitions: list[list[int]] = [[] for _ in range(client_count)]
        for label in classes:
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            probabilities = rng.dirichlet(np.full(client_count, alpha))
            counts = rng.multinomial(len(indices), probabilities)
            cursor = 0
            for client_id, count in enumerate(counts):
                partitions[client_id].extend(
                    indices[cursor : cursor + count].tolist()
                )
                cursor += count
        if min(map(len, partitions)) >= min_size:
            result = []
            for values in partitions:
                array = np.asarray(values, dtype=np.int64)
                rng.shuffle(array)
                result.append(array)
            return result
    raise RuntimeError(
        f"could not create a Dirichlet partition with min_size={min_size}"
    )
