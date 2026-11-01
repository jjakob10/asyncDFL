"""Dataset-independent graph construction and client roles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class ClientTopology:
    client_id: int
    neighbors: tuple[int, ...]
    component_id: int
    component_size: int
    is_head: bool
    has_local_data: bool
    paper_role: str


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
