"""Explicit experiment manifests and reusable resolved specifications."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, parse_client_classes
from .topology import ClientTopology, resolve_clients, ring_edges


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset: str
    trainer_module: str
    dockerfile: str
    topology_name: str
    client_count: int
    edges: tuple[tuple[int, int], ...]
    empty_client_ids: tuple[int, ...]
    disconnected_client_ids: tuple[int, ...]
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    seed: int
    clients: tuple[ClientTopology, ...]
    partition_mode: str = "iid"
    dirichlet_alpha: float = 0.5
    min_partition_size: int = 100
    client_classes: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @property
    def head_client_ids(self) -> tuple[int, ...]:
        return tuple(item.client_id for item in self.clients if item.is_head)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("experiment manifest must contain a JSON object")
    return value


def build_catalog(manifest: dict[str, Any]) -> tuple[ExperimentSpec, ...]:
    if not isinstance(manifest, dict) or not isinstance(manifest.get("experiments"), list):
        raise ValueError("manifest must contain an explicit experiments list")
    return _explicit_catalog(manifest)


def reject_collection_manifest(manifest: dict[str, Any]) -> None:
    """Prevent fixed-data tools from silently ignoring a collection schedule."""
    defaults = manifest.get("defaults", {})
    for entry in manifest.get("experiments", []):
        config = {**defaults, **manifest.get("datasets", {}).get(entry.get("dataset"), {}), **entry}
        if config.get("collection_mode", "disabled") != "disabled" or any(
            key in config for key in ("schedule", "smoke_schedule")
        ):
            raise ValueError("collection manifests require run_collection_experiments")


def _explicit_catalog(data: dict[str, Any]) -> tuple[ExperimentSpec, ...]:
    experiments = []
    for entry in data["experiments"]:
        dataset = entry["dataset"]
        config = {**data.get("defaults", {}), **data["datasets"][dataset], **entry}
        count = config["client_count"]
        if type(count) is not int or count < 3:
            raise ValueError("ring/star experiments require at least three clients")
        topology = config["topology_name"]
        if topology == "ring":
            edges = ring_edges(count, 2)
        elif topology == "star":
            edges = tuple((0, item) for item in range(1, count))
        else:
            raise ValueError("explicit topology_name must be ring or star")
        mapping = parse_client_classes(config.get("client_classes", {}), count)
        settings = Settings(
            client_id=0, client_name="client-0", client_count=count,
            head_client_ids=(count - 1,), partition_mode=config["partition_mode"],
            client_classes=mapping, dirichlet_alpha=config.get("dirichlet_alpha", 0.5),
            rounds=config["rounds"], local_epochs=config["local_epochs"],
            batch_size=config["batch_size"],
        )
        settings.validate()
        experiments.append(ExperimentSpec(
            experiment_id=config["experiment_id"], dataset=dataset,
            trainer_module=config["trainer_module"], dockerfile=config["dockerfile"],
            topology_name=topology, client_count=count, edges=edges,
            empty_client_ids=(), disconnected_client_ids=(),
            rounds=settings.rounds, local_epochs=settings.local_epochs,
            batch_size=settings.batch_size, learning_rate=config["learning_rate"],
            momentum=config["momentum"], seed=config["seed"],
            clients=resolve_clients(count, edges, (), ()),
            partition_mode=settings.partition_mode,
            dirichlet_alpha=settings.dirichlet_alpha,
            min_partition_size=config.get("min_partition_size", 100),
            client_classes=mapping,
        ))
    if len({item.experiment_id for item in experiments}) != len(experiments):
        raise ValueError("manifest contains duplicate experiment IDs")
    return tuple(experiments)


def experiment_by_id(
    experiment_id: str, manifest: dict[str, Any],
) -> ExperimentSpec:
    matches = [item for item in build_catalog(manifest) if item.experiment_id == experiment_id]
    if not matches:
        raise KeyError(f"unknown experiment: {experiment_id}")
    return matches[0]
