import numpy as np
import pytest

from app.training import make_partitions


def assert_valid_partition(parts: list[np.ndarray], size: int, clients: int) -> None:
    assert len(parts) == clients
    combined = np.concatenate(parts)
    assert len(combined) == size
    assert len(np.unique(combined)) == size
    assert set(combined.tolist()) == set(range(size))


def test_iid_partition_is_complete_disjoint_and_reproducible() -> None:
    labels = np.tile(np.arange(10), 80)
    first = make_partitions(labels, 8, "iid", seed=42, min_size=50)
    second = make_partitions(labels, 8, "iid", seed=42, min_size=50)

    assert_valid_partition(first, len(labels), 8)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert {len(part) for part in first} == {100}


def test_dirichlet_partition_is_complete_disjoint_and_reproducible() -> None:
    labels = np.tile(np.arange(10), 300)
    first = make_partitions(
        labels, 8, "dirichlet", seed=42, alpha=0.5, min_size=100
    )
    second = make_partitions(
        labels, 8, "dirichlet", seed=42, alpha=0.5, min_size=100
    )

    assert_valid_partition(first, len(labels), 8)
    assert min(map(len, first)) >= 100
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


def test_partition_rejects_impossible_minimum() -> None:
    with pytest.raises(ValueError, match="too small"):
        make_partitions(np.arange(20) % 2, 8, "iid", seed=42, min_size=3)
