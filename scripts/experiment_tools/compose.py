"""Shared, side-effect-free Docker Compose configuration for experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .catalog import ExperimentSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(item) for item in values)


def service_environment(
    experiment: ExperimentSpec,
    client_id: int,
    run_id: str,
    smoke: bool,
) -> dict[str, str]:
    client = experiment.clients[client_id]
    neighbors = ",".join(
        f"http://client-{neighbor}:9000" for neighbor in client.neighbors
    )
    environment = {
        "TRAINER_MODULE": experiment.trainer_module,
        "API_PORT": "9000",
        "CLIENT_COUNT": str(experiment.client_count),
        "CLIENT_ID": str(client_id),
        "CLIENT_NAME": f"client-{client_id}",
        "HEAD_CLIENT_IDS": _csv(experiment.head_client_ids),
        "RUN_ID": run_id,
        "EMPTY_CLIENT_IDS": _csv(experiment.empty_client_ids),
        "NEIGHBORS": neighbors,
        "ROUNDS": "1" if smoke else str(experiment.rounds),
        "LOCAL_EPOCHS": "1" if smoke else str(experiment.local_epochs),
        "BATCH_SIZE": str(experiment.batch_size),
        "LEARNING_RATE": str(experiment.learning_rate),
        "MOMENTUM": str(experiment.momentum),
        "SEED": str(experiment.seed),
        "PARTITION_MODE": experiment.partition_mode,
        "DIRICHLET_ALPHA": str(experiment.dirichlet_alpha),
        "MIN_PARTITION_SIZE": str(experiment.min_partition_size),
        "CLIENT_CLASSES": json.dumps(experiment.client_classes),
        "HEAD_START_DELAY": "8",
        "REQUEST_TIMEOUT": "5",
        "REQUEST_RETRIES": "6",
        "RETRY_DELAY": "1",
        "RESULTS_DIR": "/results",
        "EXPERIMENT_ID": experiment.experiment_id,
        "TOPOLOGY_NAME": experiment.topology_name,
        "COMPONENT_ID": str(client.component_id),
        "COMPONENT_SIZE": str(client.component_size),
        "PAPER_ROLE": client.paper_role,
        "MAX_TRAIN_SAMPLES": "400" if smoke else "0",
        "MAX_TEST_SAMPLES": "200" if smoke else "0",
        "TORCH_NUM_THREADS": "1",
    }
    return environment



def build_compose(
    experiment: ExperimentSpec,
    run_id: str,
    run_directory: Path,
    smoke: bool = False,
) -> dict[str, Any]:
    """Resolve all client services without creating files or running Docker."""
    image_name = f"dfl-paper-{experiment.dataset}:latest"
    services: dict[str, Any] = {}
    for client in experiment.clients:
        volumes = [f"{run_directory}:/results"]
        service: dict[str, Any] = {
            "image": image_name,
            "pull_policy": "never",
            "restart": "no",
            "ports": [f"{8000 + client.client_id}:9000"],
            "environment": service_environment(
                experiment,
                client.client_id,
                run_id,
                smoke,
            ),
            "volumes": volumes,
            "healthcheck": {
                "test": [
                    "CMD",
                    "python",
                    "-c",
                    (
                        "import os, urllib.request; "
                        "urllib.request.urlopen("
                        "'http://127.0.0.1:' + os.environ['API_PORT'] "
                        "+ '/health', timeout=2)"
                    ),
                ],
                "interval": "5s",
                "timeout": "3s",
                "start_period": "30s",
                "retries": 12,
            },
        }
        if client.client_id == 0:
            service["build"] = {
                "context": str(REPOSITORY_ROOT),
                "dockerfile": experiment.dockerfile,
            }
        services[f"client-{client.client_id}"] = service
    compose = {
        "name": f"dfl-{experiment.experiment_id}".replace("_", "-"),
        "services": services,
    }
    return compose
