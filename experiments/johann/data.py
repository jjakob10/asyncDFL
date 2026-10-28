"""Deterministic stratification and capacity-constrained class allocation."""
from __future__ import annotations

import hashlib

import numpy as np


def index_hash(indices: np.ndarray | list[int]) -> str:
    return hashlib.sha256(np.sort(indices).astype("<i8").tobytes()).hexdigest()


def apportion(total: int, weights: np.ndarray) -> np.ndarray:
    """Largest remainder, resolving ties by client/class index."""
    if total < 0 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("invalid allocation weights")
    quotas = total * weights / weights.sum()
    counts = np.floor(quotas).astype(np.int64)
    order = np.argsort(-(quotas - counts), kind="stable")
    counts[order[:total - int(counts.sum())]] += 1
    return counts


def stratified_subset(labels: np.ndarray, size: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    if not 0 < size <= len(labels):
        raise ValueError("stratified subset size is out of range")
    classes, counts = np.unique(labels, return_counts=True)
    allocation = apportion(size, counts)
    if np.any(allocation == 0):
        raise ValueError("sample cap is too small to retain every class")
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.permutation(np.flatnonzero(labels == label))[:count]
        for label, count in zip(classes, allocation)
    ]))


def split_training(labels: np.ndarray, seed: int, cap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split after an optional stratified global cap; never read official test labels."""
    labels = np.asarray(labels)
    candidate = (stratified_subset(labels, min(cap, len(labels)), seed)
                 if cap else np.arange(len(labels)))
    validation_local = stratified_subset(labels[candidate], round(len(candidate) * 0.1), seed + 10000)
    validation = candidate[validation_local]
    training = np.setdiff1d(candidate, validation)
    for label in np.unique(labels):
        if not np.any(labels[training] == label):
            raise ValueError("each class needs training and validation examples")
    return training, validation


def quantitative_partitions(labels: np.ndarray, distribution: str, seed: int) -> list[np.ndarray]:
    """Allocate eight balanced clients, reserving 80% of classes 3/8 for 0/4."""
    labels = np.asarray(labels, dtype=np.int64)
    if distribution not in {"distributed", "concentrated"}:
        raise ValueError("unknown Johann distribution")
    if set(np.unique(labels)) != set(range(10)):
        raise ValueError("Johann requires all ten dataset classes")
    capacity = np.full(8, len(labels) // 8, dtype=np.int64)
    capacity[:len(labels) % 8] += 1
    remaining = capacity.copy()
    rng = np.random.default_rng(seed)
    partitions = [[] for _ in range(8)]
    order = [3, 8] + [i for i in range(10) if i not in {3, 8}]
    for label in order:
        indices = rng.permutation(np.flatnonzero(labels == label))
        if distribution == "concentrated" and label in {3, 8}:
            counts = apportion(len(indices), np.array([.4, .2 / 6, .2 / 6, .2 / 6,
                                                        .4, .2 / 6, .2 / 6, .2 / 6]))
        else:
            counts = apportion(len(indices), remaining)
        if np.any(counts > remaining):
            raise ValueError("80% concentration is infeasible with balanced client capacities")
        offset = 0
        for client, count in enumerate(counts):
            partitions[client].extend(indices[offset:offset + count].tolist())
            offset += count
        remaining -= counts
    if np.any(remaining) or np.any(capacity == 0):
        raise ValueError("allocation did not fill every client capacity")
    return [rng.permutation(part).astype(np.int64) for part in partitions]
