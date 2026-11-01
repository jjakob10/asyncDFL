"""Deterministic, without-replacement arrivals and round-based data retention."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from app.config import Settings


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


@dataclass
class CollectionSource:
    """Local stock; labels use original class codes, not encoded model targets."""

    dataset: Dataset
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    class_codes: tuple[int, ...]
    chronological: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels)

    def validate(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 1 or labels.dtype.kind not in "iu":
            raise ValueError("collection labels must be a one-dimensional integer array")
        if len(labels) != len(self.dataset) or len(labels) != len(self.sample_ids):
            raise ValueError("collection stock arrays have different lengths")
        if len(set(self.sample_ids)) != len(self.sample_ids) or any(
            not isinstance(item, str) or not item for item in self.sample_ids
        ):
            raise ValueError("collection sample IDs must be unique nonempty strings")
        if (not self.class_codes or any(type(code) is not int or code < 0 for code in self.class_codes)
                or len(set(self.class_codes)) != len(self.class_codes)
                or not set(labels.tolist()).issubset(self.class_codes)):
            raise ValueError("invalid collection class codes")


def validate_schedule(schedule: dict, rounds: int, client_count: int,
                      class_codes: tuple[int, ...]) -> dict[int, dict[int, dict]]:
    if set(schedule) != {"version", "selection", "clients"}:
        raise ValueError("schedule requires only version, selection and clients")
    if type(schedule["version"]) is not int or schedule["version"] != 1:
        raise ValueError("unsupported schedule version")
    selection = schedule["selection"]
    if not isinstance(selection, str) or selection not in {"by_class", "chronological"}:
        raise ValueError("selection must be by_class or chronological")
    clients = schedule["clients"]
    if not isinstance(clients, dict) or set(clients) != {str(i) for i in range(client_count)}:
        raise ValueError("schedule must declare every client exactly once")
    result = {}
    for key, rules in clients.items():
        if not isinstance(rules, list):
            raise ValueError(f"client {key}: rules must be a list")
        expanded = {}
        for rule in rules:
            quota_key = "counts_by_class" if selection == "by_class" else "count"
            if not isinstance(rule, dict) or set(rule) != {"rounds", quota_key}:
                raise ValueError(f"client {key}: rule requires rounds and {quota_key}")
            interval = rule["rounds"]
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(type(r) is not int for r in interval)
                    or not 1 <= interval[0] <= interval[1] <= rounds):
                raise ValueError(f"client {key}: rounds must be an inclusive interval within 1..{rounds}")
            quotas = rule[quota_key]
            if selection == "by_class":
                if not isinstance(quotas, dict) or any(
                    code not in {str(c) for c in class_codes} for code in quotas
                ):
                    raise ValueError(f"client {key}: unknown class or invalid counts_by_class")
                values = list(quotas.values())
            else:
                values = [quotas]
            if any(type(q) is not int or q < 0 for q in values):
                raise ValueError(f"client {key}: quantities must be nonnegative integers")
            for round_number in range(interval[0], interval[1] + 1):
                if round_number in expanded:
                    raise ValueError(f"client {key}: overlapping rules at round {round_number}")
                expanded[round_number] = quotas
        result[int(key)] = expanded
    return result


@dataclass
class CollectionPlan:
    source: CollectionSource
    releases: tuple[np.ndarray, ...]
    resolved: dict
    plan_hash: str

    def round_data(self, round_number: int, settings: Settings) -> tuple[DataLoader | None, dict, dict]:
        if not 1 <= round_number <= len(self.releases):
            raise ValueError("round is outside the collection plan")
        start = 0 if settings.collection_mode == "accumulative" else max(
            0, round_number - settings.collection_window_rounds)
        active = np.concatenate(self.releases[start:round_number])
        cumulative = np.concatenate(self.releases[:round_number])
        new = self.releases[round_number - 1]
        expired = (self.releases[start - 1] if start else np.empty(0, dtype=np.int64))
        expired_total = np.concatenate(self.releases[:start]) if start else expired

        def describe(indices):
            return {"count": len(indices), "by_class": {
                str(code): int(np.count_nonzero(self.source.labels[indices] == code))
                for code in self.source.class_codes}}

        metrics = {
            "mode": settings.collection_mode,
            "window_rounds": settings.collection_window_rounds,
            "plan_hash": self.plan_hash,
            "stock_count": len(self.source.dataset),
            "stock_remaining": len(self.source.dataset) - len(cumulative),
            "trained": bool(len(active)),
            **{name: describe(indices) for name, indices in (
                ("new", new), ("cumulative", cumulative), ("expired", expired),
                ("expired_total", expired_total), ("active", active))},
        }
        audit = {"round": round_number, "plan_hash": self.plan_hash,
                 "released_ids": [self.source.sample_ids[i] for i in new],
                 "expired_ids": [self.source.sample_ids[i] for i in expired],
                 "mode": settings.collection_mode,
                 "window_rounds": settings.collection_window_rounds}
        loader = None
        if len(active):
            generator = torch.Generator().manual_seed(
                settings.seed + settings.client_id + round_number * 1_000_003)
            loader = DataLoader(Subset(self.source.dataset, active.tolist()),
                                batch_size=settings.batch_size, shuffle=True,
                                num_workers=0, generator=generator)
        return loader, metrics, audit


def resolve_plan(schedule: dict, source: CollectionSource, settings: Settings) -> CollectionPlan:
    source.validate()
    rules = validate_schedule(schedule, settings.rounds, settings.client_count, source.class_codes)
    if schedule["selection"] == "chronological" and not source.chronological:
        raise ValueError("chronological schedule requires a chronologically ordered source")
    if source.chronological and schedule["selection"] != "chronological":
        raise ValueError("chronological stock cannot be reordered by class")
    rng = np.random.default_rng(np.random.SeedSequence([settings.seed, settings.client_id]))
    queues = {code: rng.permutation(np.flatnonzero(source.labels == code))
              for code in sorted(source.class_codes)}
    cursors = dict.fromkeys(source.class_codes, 0)
    chronological_cursor = 0
    releases = []
    for r in range(1, settings.rounds + 1):
        quotas = rules[settings.client_id].get(r)
        selected = []
        if quotas is not None:
            requests = (sorted(((int(k), v) for k, v in quotas.items()))
                        if schedule["selection"] == "by_class" else [(None, quotas)])
            for code, requested in requests:
                queue = np.arange(len(source.dataset)) if code is None else queues[code]
                cursor = chronological_cursor if code is None else cursors[code]
                available = len(queue) - cursor
                if requested > available:
                    label = "" if code is None else f" class {code}"
                    raise ValueError(f"client {settings.client_id} round {r}{label}: "
                                     f"requested {requested}, available {available}")
                selected.extend(queue[cursor:cursor + requested].tolist())
                if code is None:
                    chronological_cursor += requested
                else:
                    cursors[code] += requested
        releases.append(np.asarray(selected, dtype=np.int64))
    resolved = {
        "version": 1, "client_id": settings.client_id, "seed": settings.seed,
        "schedule": schedule, "source_metadata": source.metadata,
        "stock_hash": digest(list(zip(source.sample_ids, source.labels.tolist()))),
        "releases": [[source.sample_ids[i] for i in batch] for batch in releases],
    }
    return CollectionPlan(source, tuple(releases), resolved, digest(resolved))
