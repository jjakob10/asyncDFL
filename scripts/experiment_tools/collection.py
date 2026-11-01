"""Collection specifications and Compose augmentation for explicit manifests."""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

from app.collection import digest, read_json, validate_schedule
from app.config import Settings
from .catalog import ExperimentSpec, build_catalog
from .compose import build_compose

from . import ROOT
from app.training import load_trainer



@dataclass(frozen=True)
class CollectionExperiment:
    spec: ExperimentSpec
    mode: str
    window_rounds: int
    schedule: dict
    smoke: bool


def load_catalog(path: Path, smoke: bool = False) -> tuple[CollectionExperiment, ...]:
    manifest = read_json(path)
    bases = build_catalog(manifest)
    result = []
    for base, entry in zip(bases, manifest["experiments"]):
        if smoke:
            base = replace(base, rounds=5, local_epochs=1)
        schedule = read_json(path.parent / entry["smoke_schedule" if smoke else "schedule"])
        codes = getattr(load_trainer(base.trainer_module), "COLLECTION_CLASS_CODES", None)
        if not isinstance(codes, tuple) or not codes:
            raise ValueError("collection trainer must expose COLLECTION_CLASS_CODES")
        validate_schedule(schedule, base.rounds, base.client_count, codes)
        experiment = CollectionExperiment(base, entry["collection_mode"],
                                          entry.get("window_rounds", 3), schedule, smoke)
        client_settings(experiment, 0).validate()
        result.append(experiment)
    return tuple(result)


def client_settings(experiment: CollectionExperiment, client_id: int,
                    data_dir: str = "/data") -> Settings:
    base = experiment.spec
    common = {f.name: getattr(base, f.name) for f in fields(Settings)
              if hasattr(base, f.name) and f.name not in {"clients", "head_client_ids"}}
    client = base.clients[client_id]
    return Settings(**common, client_id=client_id, client_name=f"client-{client_id}",
                    head_client_ids=base.head_client_ids,
                    neighbors=tuple(f"http://client-{i}:9000" for i in client.neighbors),
                    component_id=client.component_id, component_size=client.component_size,
                    paper_role=client.paper_role, data_dir=data_dir,
                    collection_mode=experiment.mode,
                    collection_window_rounds=experiment.window_rounds,
                    collection_schedule_path="/results/schedule.json",
                    collection_preflight_path="/results/preflight.json",
                    max_test_samples=200 if experiment.smoke else 0)


def collection_compose(experiment: CollectionExperiment, run_id: str, directory: Path) -> dict:
    compose = build_compose(experiment.spec, run_id, directory)
    compose["name"] = f"dfl-collection-{digest([experiment.spec.experiment_id, run_id])[:16]}"
    for service in compose["services"].values():
        service["image"] = f"dfl-thiago-coleta-simulada-{experiment.spec.dataset}:latest"
        if "build" in service:
            service["build"]["args"] = {"DATASET": experiment.spec.dataset}
        service["environment"].update({
            "DATA_DIR": "/data", "DOWNLOAD_DATA": "false",
            "COLLECTION_MODE": experiment.mode,
            "COLLECTION_WINDOW_ROUNDS": str(experiment.window_rounds),
            "COLLECTION_SCHEDULE_PATH": "/results/schedule.json",
            "COLLECTION_PREFLIGHT_PATH": "/results/preflight.json",
            "MAX_TEST_SAMPLES": "200" if experiment.smoke else "0",
        })
    return compose
