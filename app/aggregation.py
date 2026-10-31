"""Published age-weighted/limited pairwise rules, adapted to full CPU snapshots.

Sources: Hegedus et al. (JPDC 2021), Danner et al. (ICCCI 2023), Alg. 3.
Age counts example visits, not distinct data or local publication versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.federation import StateDict, clone_state_dict, fedavg, validate_compatible

AGE_UNIT = "example_visits_v1"
METHODS = ("M0", "M1", "M2", "M3", "M4")


@dataclass
class ModelInput:
    state: StateDict
    client_id: int
    age: int | None
    sample_count: int
    metadata: dict = field(default_factory=dict)


@dataclass
class AggregationResult:
    state: StateDict
    age: int | None
    weights: dict[int, float]
    trace: list[dict]


def aggregate_models(method: str, local: ModelInput, neighbors: list[ModelInput],
                     limit: int = 0) -> AggregationResult:
    if method not in METHODS:
        raise ValueError("unsupported aggregation method")
    inputs = [local, *sorted(neighbors, key=lambda item: item.client_id)]
    if len({item.client_id for item in inputs}) != len(inputs):
        raise ValueError("duplicate aggregation client")
    for item in inputs:
        validate_compatible(local.state, item.state)
        if item.sample_count <= 0:
            raise ValueError("sample counts must be positive")
        if item.age is not None and (type(item.age) is not int or item.age < 0):
            raise ValueError("training ages must be nonnegative integers")
    if method in {"M0", "M3", "M4"}:
        total = sum(item.sample_count for item in inputs)
        # Legacy peers may omit age; do not fabricate a complete lineage from it.
        age = None if any(item.age is None for item in inputs) else max(item.age for item in inputs)
        return AggregationResult(
            fedavg([(item.state, item.sample_count) for item in inputs]), age,
            {item.client_id: item.sample_count / total for item in inputs}, [])
    if any(item.age is None for item in inputs):
        raise ValueError("age-based aggregation requires complete training ages")
    if type(limit) is not int or limit < 0:
        raise ValueError("merge limit must be a nonnegative integer")
    state, age = clone_state_dict(local.state), local.age
    weights = {item.client_id: float(item is local) for item in inputs}
    trace = []
    for peer in inputs[1:]:
        before = age
        if method == "M2" and abs(age - peer.age) > limit:
            incoming = float(peer.age > age)
            action = "replace" if incoming else "keep"
        elif age + peer.age == 0:
            incoming, action = 0.0, "both_zero_keep"
        else:
            incoming, action = peer.age / (age + peer.age), "age_weighted"
        if incoming == 1:
            state = clone_state_dict(peer.state)
        elif incoming > 0:
            state = fedavg([(state, 1 - incoming), (peer.state, incoming)])
        weights = {client: value * (1 - incoming) for client, value in weights.items()}
        weights[peer.client_id] += incoming
        age = max(age, peer.age)
        trace.append({"client_id": peer.client_id, "local_age_before": before,
                      "incoming_age": peer.age, "incoming_weight": incoming,
                      "action": action, "age_after": age})
    return AggregationResult(state, age, weights, trace)


def select_rejoin_neighbor(neighbors: list[ModelInput]) -> ModelInput | None:
    eligible = [item for item in neighbors if item.age is not None]
    return min(eligible, key=lambda item: (-item.age, item.client_id)) if eligible else None
