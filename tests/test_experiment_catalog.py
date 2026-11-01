from pathlib import Path

import pytest

from scripts.experiment_tools.catalog import build_catalog, experiment_by_id, load_manifest
from scripts.experiment_tools.topology import resolve_clients, ring_edges

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_catalog_preserves_ring_star_and_class_partitions():
    for dataset in ("mnist", "cifar10"):
        manifest = load_manifest(ROOT / "experiments" / "thiago" / dataset / "manifest.json")
        catalog = build_catalog(manifest)
        assert len(catalog) == 6
        assert {item.topology_name for item in catalog} == {"ring", "star"}
        assert all(item.client_count == 6 for item in catalog)
        assert experiment_by_id(catalog[0].experiment_id, manifest) == catalog[0]
        with pytest.raises(KeyError, match="unknown experiment"):
            experiment_by_id("unknown", manifest)


def test_explicit_manifest_is_required():
    for value in ({}, {"regular_setups": {}}, {"experiments": {}}):
        with pytest.raises(ValueError, match="explicit experiments list"):
            build_catalog(value)


def test_graph_components_heads_and_roles_are_dataset_independent():
    clients = resolve_clients(5, ((0, 1), (1, 2)), (0, 4), (3, 4))
    assert tuple(c.client_id for c in clients if c.is_head) == (2, 3, 4)
    assert [c.component_size for c in clients] == [3, 3, 3, 1, 1]
    assert [c.paper_role for c in clients] == ["E", "R", "R", "D", "ED"]
    assert all(c.is_head for c in resolve_clients(3, (), (), ()))


def test_ring_construction_and_invalid_edges():
    assert len(ring_edges(6, 2)) == 6
    assert (0, 5) in ring_edges(6, 2)
    assert len(ring_edges(8, 7)) == 28
    for count, degree in ((0, 0), (6, 6), (6, 3)):
        with pytest.raises(ValueError):
            ring_edges(count, degree)
    with pytest.raises(ValueError, match="invalid edge"):
        resolve_clients(3, ((0, 0),), (), ())
