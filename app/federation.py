from __future__ import annotations

import io
from collections.abc import Mapping, Sequence

import torch

StateDict = dict[str, torch.Tensor]


def clone_state_dict(state: Mapping[str, torch.Tensor]) -> StateDict:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def validate_compatible(reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]) -> None:
    if reference.keys() != candidate.keys():
        raise ValueError("model state dictionaries have different parameter names")
    for name, reference_tensor in reference.items():
        candidate_tensor = candidate[name]
        if reference_tensor.shape != candidate_tensor.shape:
            raise ValueError(f"shape mismatch for parameter {name}")
        if reference_tensor.dtype != candidate_tensor.dtype:
            raise ValueError(f"dtype mismatch for parameter {name}")


def fedavg(weighted_states: Sequence[tuple[Mapping[str, torch.Tensor], float]]) -> StateDict:
    if not weighted_states:
        raise ValueError("FedAvg requires at least one model")
    if any(weight <= 0 for _, weight in weighted_states):
        raise ValueError("FedAvg weights must be positive")

    reference = weighted_states[0][0]
    for candidate, _ in weighted_states[1:]:
        validate_compatible(reference, candidate)

    total_weight = float(sum(weight for _, weight in weighted_states))
    averaged: StateDict = {}
    for name, reference_tensor in reference.items():
        accumulator = torch.zeros_like(reference_tensor, dtype=torch.float64, device="cpu")
        for state, weight in weighted_states:
            accumulator.add_(state[name].detach().cpu().to(torch.float64), alpha=weight / total_weight)

        if reference_tensor.is_floating_point() or reference_tensor.is_complex():
            averaged[name] = accumulator.to(reference_tensor.dtype)
        elif reference_tensor.dtype == torch.bool:
            averaged[name] = (accumulator >= 0.5).to(torch.bool)
        else:
            averaged[name] = accumulator.round().to(reference_tensor.dtype)
    return averaged


def serialize_state_dict(state: Mapping[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(clone_state_dict(state), buffer)
    return buffer.getvalue()


def deserialize_state_dict(
    payload: bytes,
    reference: Mapping[str, torch.Tensor] | None = None,
) -> StateDict:
    try:
        loaded = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid PyTorch model payload") from exc
    if not isinstance(loaded, dict) or not all(
        isinstance(name, str) and isinstance(tensor, torch.Tensor)
        for name, tensor in loaded.items()
    ):
        raise ValueError("payload is not a state dictionary")
    state = clone_state_dict(loaded)
    if reference is not None:
        validate_compatible(reference, state)
    return state

