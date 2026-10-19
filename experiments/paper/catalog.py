from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.config import Settings, parse_client_classes


MANIFEST_PATH = Path(__file__).with_name("manifest.json")


@dataclass(frozen=True)
class ClientTopology:
    client_id: int
    neighbors: tuple[int, ...]
    component_id: int
    component_size: int
    is_head: bool
    has_local_data: bool
    paper_role: str


@dataclass(frozen=True)
class PaperExperiment:
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


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("paper manifest must contain a JSON object")
    return value


def ring_edges(client_count: int, degree: int) -> tuple[tuple[int, int], ...]:
    if client_count < 1 or not 0 <= degree < client_count:
        raise ValueError("degree must be between zero and client_count - 1")
    if degree not in {client_count - 1} and degree % 2:
        raise ValueError("non-complete ring degree must be even")
    edges: set[tuple[int, int]] = set()
    if degree == client_count - 1:
        for left in range(client_count):
            for right in range(left + 1, client_count):
                edges.add((left, right))
    else:
        for distance in range(1, degree // 2 + 1):
            for left in range(client_count):
                right = (left + distance) % client_count
                edges.add(tuple(sorted((left, right))))
    return tuple(sorted(edges))


def _components(
    client_count: int, edges: Iterable[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(client_count)]
    for left, right in edges:
        if left == right or not 0 <= left < client_count or not 0 <= right < client_count:
            raise ValueError("topology contains an invalid edge")
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(client_count))
    components: list[tuple[int, ...]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        found: set[int] = set()
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            unseen.discard(current)
            stack.extend(adjacency[current] - found)
        components.append(tuple(sorted(found)))
    return tuple(components)


def resolve_clients(
    client_count: int,
    edges: tuple[tuple[int, int], ...],
    empty_client_ids: tuple[int, ...],
    disconnected_client_ids: tuple[int, ...],
) -> tuple[ClientTopology, ...]:
    adjacency = [set() for _ in range(client_count)]
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = _components(client_count, edges)
    component_lookup = {
        client_id: (component_id, component)
        for component_id, component in enumerate(components)
        for client_id in component
    }
    empty = set(empty_client_ids)
    disconnected = set(disconnected_client_ids)
    result = []
    for client_id in range(client_count):
        component_id, component = component_lookup[client_id]
        is_empty = client_id in empty
        is_disconnected = client_id in disconnected
        role = (
            "ED"
            if is_empty and is_disconnected
            else "E"
            if is_empty
            else "D"
            if is_disconnected
            else "R"
        )
        result.append(
            ClientTopology(
                client_id=client_id,
                neighbors=tuple(sorted(adjacency[client_id])),
                component_id=component_id,
                component_size=len(component),
                is_head=client_id == max(component),
                has_local_data=not is_empty,
                paper_role=role,
            )
        )
    return tuple(result)


def build_catalog(manifest: dict[str, Any] | None = None) -> tuple[PaperExperiment, ...]:
    data = manifest or load_manifest()
    if "experiments" in data:
        return _explicit_catalog(data)
    defaults = data["defaults"]
    experiments: list[PaperExperiment] = []
    for dataset, dataset_config in data["datasets"].items():
        for setup_name, setup in data["regular_setups"].items():
            connected = int(setup["connected_clients"])
            total = int(setup["total_clients"])
            empty = tuple(int(item) for item in setup["empty_clients"])
            disconnected = tuple(int(item) for item in setup["disconnected_clients"])
            for degree in setup["degrees"]:
                edges = ring_edges(connected, int(degree))
                for local_epochs in dataset_config["local_epochs"]:
                    experiment_id = (
                        f"{dataset}-{setup_name}-top{degree}-e{local_epochs}"
                    )
                    clients = resolve_clients(total, edges, empty, disconnected)
                    experiments.append(
                        PaperExperiment(
                            experiment_id=experiment_id,
                            dataset=dataset,
                            trainer_module=dataset_config["trainer_module"],
                            dockerfile=dataset_config["dockerfile"],
                            topology_name=f"topology-{degree}",
                            client_count=total,
                            edges=edges,
                            empty_client_ids=empty,
                            disconnected_client_ids=disconnected,
                            rounds=int(dataset_config["rounds"]),
                            local_epochs=int(local_epochs),
                            batch_size=int(defaults["batch_size"]),
                            learning_rate=float(defaults["learning_rate"]),
                            momentum=float(defaults["momentum"]),
                            seed=int(defaults["seed"]),
                            clients=clients,
                        )
                    )
        if dataset_config["irregular"]:
            for topology_name, topology in data["irregular_topologies"].items():
                client_count = int(topology["client_count"])
                edges = tuple(tuple(map(int, edge)) for edge in topology["edges"])
                clients = resolve_clients(client_count, edges, (), ())
                experiments.append(
                    PaperExperiment(
                        experiment_id=f"{dataset}-{topology_name}-e32",
                        dataset=dataset,
                        trainer_module=dataset_config["trainer_module"],
                        dockerfile=dataset_config["dockerfile"],
                        topology_name=topology_name,
                        client_count=client_count,
                        edges=edges,
                        empty_client_ids=(),
                        disconnected_client_ids=(),
                        rounds=int(dataset_config["rounds"]),
                        local_epochs=32,
                        batch_size=int(defaults["batch_size"]),
                        learning_rate=float(defaults["learning_rate"]),
                        momentum=float(defaults["momentum"]),
                        seed=int(defaults["seed"]),
                        clients=clients,
                    )
                )
    return tuple(experiments)


def _explicit_catalog(data: dict[str, Any]) -> tuple[PaperExperiment, ...]:
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
        experiments.append(PaperExperiment(
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
    experiment_id: str, manifest: dict[str, Any] | None = None,
) -> PaperExperiment:
    matches = [item for item in build_catalog(manifest) if item.experiment_id == experiment_id]
    if not matches:
        raise KeyError(f"unknown paper experiment: {experiment_id}")
    return matches[0]
