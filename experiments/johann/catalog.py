"""Resolved Johann specifications and shared Compose generation."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from app.aggregation import AGE_UNIT, METHODS

from scripts.experiment_tools.catalog import ExperimentSpec
from scripts.experiment_tools.topology import resolve_clients
from scripts.experiment_tools.compose import build_compose

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = tuple(f"{d}-{a}" for d in ("distributed", "concentrated") for a in ("r", "c", "none"))
GROUPS = {"r": (0, 4), "c": (2, 6), "none": ()}


def topology() -> tuple[tuple[int, int], ...]:
    edges = {tuple(sorted((i, (i + 1) % 8))) for i in range(8)}
    edges.update((i, i + 4) for i in range(4))
    return tuple(sorted(edges))


def validate_topology(edges: tuple[tuple[int, int], ...]) -> None:
    for removed in GROUPS.values():
        active = set(range(8)) - set(removed)
        visited, stack = set(), [min(active)]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(b if a == node else a for a, b in edges
                         if node in (a, b) and a in active and b in active)
        if visited != active:
            raise ValueError(f"active graph is disconnected after removing {removed}")


def load_spec(dataset: str, scenario: str, seed: int = 7, duration: float | None = None,
              interval: float | None = None, mode: str = "pilot", method: str = "M0",
              method_parameters: dict | None = None, *,
              base_manifest: dict | None = None) -> dict[str, Any]:
    if dataset not in {"mnist", "cifar10"} or scenario not in SCENARIOS:
        raise ValueError("unsupported Johann dataset or scenario")
    spec = (deepcopy(base_manifest) if base_manifest is not None else
            json.loads((Path(__file__).parent / dataset / "manifest.json").read_text(encoding="utf-8")))
    required = {"dataset": dataset, "client_count": 8, "tracked_classes": [3, 8],
                "group_r": [0, 4], "group_c": [2, 6], "validation_fraction": .1,
                "concentration_fraction": .8, "scenarios": list(SCENARIOS)}
    for key, expected in required.items():
        if spec[key] != expected:
            raise ValueError(f"this M0 implementation requires {key}={expected}")
    spec.update(scenario=scenario, seed=seed, mode=mode,
                duration_seconds=duration if duration is not None else spec["duration_seconds"],
                evaluation_interval_seconds=interval if interval is not None else spec["evaluation_interval_seconds"],
                evaluation_split="test" if mode == "collection" else "validation",
                offline_client_ids=list(GROUPS[scenario.rsplit("-", 1)[1]]), method=method)
    if method not in METHODS:
        raise ValueError("unsupported Johann method")
    n = math.ceil((720 if mode == "smoke" else {"mnist": 54000, "cifar10": 45000}[dataset]) / 8)
    configured = dict(spec.get("methods", {}).get(method, {}))
    overrides = method_parameters or {}
    allowed = {"M0": set(), "M1": set(), "M2": {"merge_limit_examples", "merge_limit_multiplier"},
               "M3": {"rejoin_batches"},
               "M4": {"rejoin_batches", "distillation_weight", "distillation_temperature"}}[method]
    if set(overrides) - allowed:
        raise ValueError(f"parameters do not apply to {method}: {sorted(set(overrides) - allowed)}")
    if "merge_limit_multiplier" in overrides:
        configured.pop("merge_limit_examples", None)
    configured.update(overrides)
    if method == "M2":
        multiplier = configured.pop("merge_limit_multiplier", 1)
        if type(multiplier) is not int or multiplier < 1:
            raise ValueError("merge limit multiplier must be a positive integer")
        configured.setdefault("merge_limit_examples", n * multiplier)
    if method in {"M3", "M4"}:
        configured.setdefault("rejoin_batches", 16)
    if method == "M4":
        configured.setdefault("distillation_weight", .5)
        configured.setdefault("distillation_temperature", 2.0)
    if "merge_limit_examples" in configured and (type(configured["merge_limit_examples"]) is not int or configured["merge_limit_examples"] < 0):
        raise ValueError("merge_limit_examples must be a nonnegative integer")
    if "rejoin_batches" in configured and (type(configured["rejoin_batches"]) is not int or configured["rejoin_batches"] < 1):
        raise ValueError("rejoin_batches must be a positive integer")
    for key, minimum in (("distillation_weight", 0), ("distillation_temperature", 0)):
        if key in configured and (isinstance(configured[key], bool) or not isinstance(configured[key], (int, float))
                                  or not math.isfinite(configured[key]) or configured[key] < minimum
                                  or key == "distillation_temperature" and configured[key] == 0):
            raise ValueError(f"invalid {key}")
    if set(configured) - allowed:
        raise ValueError(f"unsupported manifest parameters for {method}")
    spec.update(method_parameters=configured, training_age_unit=AGE_UNIT,
                max_local_training_samples=n, execution_backend="docker")
    if mode not in {"pilot", "collection", "smoke"} or type(seed) is not int or seed < 0:
        raise ValueError("invalid mode or seed")
    for key in ("duration_seconds", "evaluation_interval_seconds"):
        if not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if spec["evaluation_interval_seconds"] > spec["duration_seconds"] / 10:
        raise ValueError("evaluation interval must allow at least ten observations per run")
    if not 0 < spec["offline_start_fraction"] < spec["offline_end_fraction"] < 1:
        raise ValueError("invalid offline fractions")
    validate_topology(topology())
    spec["edges"] = [list(edge) for edge in topology()]
    return spec


def compose_for(spec: dict[str, Any], run_id: str, directory: Path) -> dict[str, Any]:
    dataset = spec["dataset"]
    edges = topology()
    experiment = ExperimentSpec(
        experiment_id=f"johann-{dataset}-{spec['method']}-{spec['scenario']}", dataset=dataset,
        trainer_module=f"experiments.johann.{dataset}.trainer",
        dockerfile=f"experiments/{dataset}/Dockerfile", topology_name="ring-opposites",
        client_count=8, edges=edges, empty_client_ids=(), disconnected_client_ids=(),
        rounds=1, local_epochs=spec["local_epochs"], batch_size=spec["batch_size"],
        learning_rate=spec["learning_rate"], momentum=spec["momentum"], seed=spec["seed"],
        clients=resolve_clients(8, edges, (), ()),
    )
    compose = build_compose(experiment, run_id, directory)
    if spec.get("frozen_bundle"):
        # Compose normalizes project names; record the same name used in engine labels.
        compose["name"] = compose["name"].lower()
    # Same Dockerfiles and build cache; separate tags prevent replacing legacy images.
    for service in compose["services"].values():
        service["image"] = f"dfl-johann-{dataset}:latest"
        if spec.get("frozen_bundle"):
            service["image"] = spec["expected_image_id"]
            service.pop("build", None)
            service["pull_policy"] = "never"
        service["environment"].update({
            "RUN_DURATION_SECONDS": str(spec["duration_seconds"]),
            "EVALUATION_INTERVAL_SECONDS": str(spec["evaluation_interval_seconds"]),
            "OFFLINE_CLIENT_IDS": ",".join(map(str, spec["offline_client_ids"])),
            "OFFLINE_START_FRACTION": str(spec["offline_start_fraction"]),
            "OFFLINE_END_FRACTION": str(spec["offline_end_fraction"]),
            "AUTO_START_HEAD": "false", "REQUEST_TIMEOUT": "1", "REQUEST_RETRIES": "1",
            "JOHANN_DISTRIBUTION": spec["scenario"].rsplit("-", 1)[0],
            "EVALUATION_SPLIT": spec["evaluation_split"],
            "MAX_TRAIN_SAMPLES": "800" if spec["mode"] == "smoke" else "0",
            "MAX_TEST_SAMPLES": "100" if spec["mode"] == "smoke" else "0",
            "METHOD": spec["method"],
            "MERGE_LIMIT_EXAMPLES": str(spec["method_parameters"].get("merge_limit_examples", 0)),
            "REJOIN_BATCHES": str(spec["method_parameters"].get("rejoin_batches", 16)),
            "DISTILLATION_WEIGHT": str(spec["method_parameters"].get("distillation_weight", 0)),
            "DISTILLATION_TEMPERATURE": str(spec["method_parameters"].get("distillation_temperature", 2)),
        })
    return compose
