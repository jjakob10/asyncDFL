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


@pytest.mark.parametrize("clients", [4, 8])
def test_dirichlet_partition_is_complete_disjoint_and_reproducible(clients) -> None:
    labels = np.tile(np.arange(10), 300)
    first = make_partitions(
        labels, clients, "dirichlet", seed=42, alpha=0.5, min_size=100
    )
    second = make_partitions(
        labels, clients, "dirichlet", seed=42, alpha=0.5, min_size=100
    )

    assert_valid_partition(first, len(labels), clients)
    assert min(map(len, first)) >= 100
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    proportions = np.array([
        np.bincount(labels[part], minlength=10) / len(part) for part in first
    ])
    assert np.max(np.ptp(proportions, axis=0)) > 0.1


@pytest.mark.parametrize("clients", [4, 8])
def test_dirichlet_alpha_changes_class_concentration(clients) -> None:
    labels = np.tile(np.arange(10), 1000)
    concentrations = []
    for alpha in (0.1, 100.0):
        parts = make_partitions(
            labels, clients, "dirichlet", seed=42, alpha=alpha, min_size=1
        )
        proportions = np.array([
            np.bincount(labels[part], minlength=10) / len(part) for part in parts
        ])
        concentrations.append(np.mean(np.sum(proportions ** 2, axis=1)))
    assert concentrations[0] > concentrations[1] + 0.05


def test_partition_rejects_impossible_minimum() -> None:
    with pytest.raises(ValueError, match="too small"):
        make_partitions(np.arange(20) % 2, 8, "iid", seed=42, min_size=3)


@pytest.mark.parametrize("alpha", [0, -0.5])
def test_dirichlet_rejects_nonpositive_alpha(alpha) -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        make_partitions(np.arange(800) % 10, 8, "dirichlet", 42, alpha=alpha)


def test_dirichlet_reports_exhausted_attempts() -> None:
    with pytest.raises(RuntimeError, match="could not create a Dirichlet partition"):
        make_partitions(
            np.arange(800) % 10, 8, "dirichlet", 42, max_attempts=0
        )


def test_empty_clients_receive_no_data_and_remaining_partition_is_complete() -> None:
    labels = np.tile(np.arange(10), 70)

    parts = make_partitions(
        labels,
        10,
        "iid",
        seed=1999,
        min_size=50,
        empty_client_ids=(0, 4, 9),
    )

    assert [len(parts[item]) for item in (0, 4, 9)] == [0, 0, 0]
    combined = np.concatenate([part for part in parts if len(part)])
    assert len(combined) == len(labels)
    assert len(set(combined.tolist())) == len(labels)
