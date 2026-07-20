from __future__ import annotations

from experiments.paper.catalog import build_catalog, experiment_by_id, ring_edges


def test_catalog_contains_complete_decentralized_matrix() -> None:
    catalog = build_catalog()

    assert len(catalog) == 158
    assert len({item.experiment_id for item in catalog}) == 158
    assert {item.dataset for item in catalog} == {"mnist", "cifar10", "ton_iot"}


def test_ring_topology_degrees_and_component_heads() -> None:
    experiment = experiment_by_id("mnist-8p2-top4-e32")

    assert experiment.head_client_ids == (7, 8, 9)
    assert [len(item.neighbors) for item in experiment.clients[:8]] == [4] * 8
    assert experiment.clients[8].neighbors == ()
    assert experiment.clients[9].neighbors == ()
    assert experiment.clients[0].paper_role == "E"
    assert experiment.clients[8].paper_role == "D"
    assert experiment.clients[9].paper_role == "ED"


def test_disconnected_topology_makes_every_client_a_head() -> None:
    experiment = experiment_by_id("cifar10-16p4-top0-e2")

    assert experiment.head_client_ids == tuple(range(20))
    assert all(item.component_size == 1 for item in experiment.clients)


def test_complete_ring_has_expected_edge_count() -> None:
    assert len(ring_edges(8, 7)) == 28
    assert len(ring_edges(16, 15)) == 120


def test_irregular_topologies_are_fixed_connected_fixtures() -> None:
    sparse = experiment_by_id("mnist-irregular-sparse-e32")
    dense = experiment_by_id("mnist-irregular-dense-e32")

    assert len(sparse.edges) == 7
    assert len(dense.edges) == 16
    assert sparse.head_client_ids == (7,)
    assert dense.head_client_ids == (7,)
    assert all(item.component_size == 8 for item in sparse.clients)
