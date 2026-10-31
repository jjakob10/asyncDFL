"""Adapt the existing image trainers without changing their fixed-data behavior."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from torch.utils.data import Subset

from app.collection import CollectionSource
from app.config import Settings
from app.training import DataBundle


def load_collection_data(trainer: Any, name: str, settings: Settings) -> DataBundle:
    data = trainer.load_data(settings)
    if data.train_loader is None:
        raise ValueError("image collection adapter requires a nonempty local stock")
    subset = data.train_loader.dataset
    if not isinstance(subset, Subset):
        raise ValueError("image trainer must return a training Subset")
    targets = np.asarray(subset.dataset.targets, dtype=np.int64)
    indices = np.asarray(subset.indices, dtype=np.int64)
    data.collection_source = CollectionSource(
        subset, targets[indices], tuple(f"{name}:train:{i}" for i in indices),
        tuple(range(10)), metadata={
            "dataset": name, "partition_mode": settings.partition_mode,
            "targets_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
        })
    return data
