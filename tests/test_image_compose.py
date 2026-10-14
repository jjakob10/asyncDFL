"""Validate standalone image experiments without starting containers."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    ("ring", 8, "compose.ring.8.yaml"),
    ("ring", 4, "compose.ring.4.yaml"),
    ("star", 4, "compose.star.4.yaml"),
    ("star", 8, "compose.star.8.yaml"),
]


@pytest.fixture(scope="module")
def docker_compose() -> str:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable")
    version = subprocess.run(
        [docker, "compose", "version"], capture_output=True, text=True, timeout=30
    )
    if version.returncode:
        pytest.skip("Docker Compose is unavailable")
    return docker


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
@pytest.mark.parametrize("topology,count,filename", VARIANTS)
@pytest.mark.parametrize("iid", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_image_compose_configuration(
    docker_compose, tmp_path, dataset, topology, count, filename, iid, override
) -> None:
    identity = f"{dataset}-{topology}-{count}" + ("-iid" if iid else "")
    if iid:
        filename = filename.replace(".yaml", ".iid.yaml")
    env_file = tmp_path / "experiment.env"
    overrides = {
        "PARTITION_MODE": "dirichlet" if iid else "iid", "DIRICHLET_ALPHA": "0.2",
        "MIN_PARTITION_SIZE": "7", "SEED": "123", "ROUNDS": "1",
        "MAX_TRAIN_SAMPLES": "800", "MAX_TEST_SAMPLES": "40",
        "RUN_ID": "smoke-run",
    }
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in overrides.items())
        if override else ""
    )
    # Shell values take precedence over --env-file; isolate all interpolated keys.
    defaults = {
        "PARTITION_MODE": "iid" if iid else "dirichlet", "DIRICHLET_ALPHA": "0.5",
        "MIN_PARTITION_SIZE": "100", "SEED": "42", "ROUNDS": "10",
        "MAX_TRAIN_SAMPLES": "0", "MAX_TEST_SAMPLES": "0",
        "RUN_ID": identity,
    }
    excluded = set(defaults) | {
        "TRAINER_MODULE", "LOCAL_EPOCHS", "BATCH_SIZE", "LEARNING_RATE",
        "MOMENTUM", "REQUEST_TIMEOUT", "REQUEST_RETRIES", "RETRY_DELAY",
        "HEAD_START_DELAY", "TORCH_NUM_THREADS",
    }
    environment = {
        key: value for key, value in os.environ.items()
        if key.upper() not in excluded and not key.upper().startswith("COMPOSE_")
    }
    result = subprocess.run(
        [docker_compose, "compose", "--env-file", str(env_file), "-f",
         str(ROOT / "experiments" / dataset / filename),
         "config", "--format", "json"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["name"] == f"dfl-{identity}"
    services = rendered["services"]
    assert set(services) == {f"client-{item}" for item in range(count)}
    head = count - 1 if topology == "ring" else 0
    for client_id in range(count):
        service = services[f"client-{client_id}"]
        env = service["environment"]
        expected = dict(overrides if override else defaults)
        if iid:
            expected["PARTITION_MODE"] = "iid"
        assert {key: env[key] for key in expected} == expected
        assert env["API_PORT"] == "9000"
        assert env["CLIENT_COUNT"] == str(count)
        assert env["CLIENT_ID"] == str(client_id)
        assert env["HEAD_CLIENT_IDS"] == str(head)
        assert env["TRAINER_MODULE"] == f"experiments.{dataset}.trainer"
        assert env["EXPERIMENT_ID"] == identity
        assert env["TOPOLOGY_NAME"] == topology
        assert env["COMPONENT_ID"] == "0"
        assert env["COMPONENT_SIZE"] == str(count)
        assert not env.get("EMPTY_CLIENT_IDS")
        if topology == "ring":
            neighbors = {(client_id - 1) % count, (client_id + 1) % count}
        else:
            neighbors = set(range(1, count)) if client_id == 0 else {0}
        assert set(env["NEIGHBORS"].split(",")) == {
            f"http://client-{item}:9000" for item in neighbors
        }
        dependencies = service.get("depends_on", {})
        if client_id == head:
            assert set(dependencies) == set(services) - {f"client-{head}"}
            assert all(
                item["condition"] == "service_healthy"
                for item in dependencies.values()
            )
        else:
            assert not dependencies
        assert str(service["ports"][0]["published"]) == str(8000 + client_id)
        assert service["ports"][0]["target"] == 9000
        volume = service["volumes"][0]
        assert Path(volume["source"]).resolve() == ROOT / "results" / identity
        assert volume["target"] == "/results"
        assert service["healthcheck"]["test"][0] == "CMD"
        assert "os.environ['API_PORT']" in service["healthcheck"]["test"][3]


@pytest.mark.parametrize("count", [4, 8])
def test_ecg_compose_ports(docker_compose, tmp_path, count) -> None:
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    result = subprocess.run(
        [docker_compose, "compose", "--env-file", str(env_file), "-f",
         str(ROOT / "experiments/ecg" / f"compose.ring.{count}.yaml"),
         "config", "--format", "json"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    services = json.loads(result.stdout)["services"]
    assert len(services) == count
    for client_id in range(count):
        service = services[f"client-{client_id}"]
        assert service["environment"]["API_PORT"] == "9000"
        assert service["ports"][0]["target"] == 9000
        assert str(service["ports"][0]["published"]) == str(8000 + client_id)
        assert all(
            url.endswith(":9000")
            for url in service["environment"]["NEIGHBORS"].split(",")
        )
        assert "os.environ['API_PORT']" in service["healthcheck"]["test"][3]


@pytest.mark.parametrize("dataset", ["mnist", "cifar10", "ecg", "ton_iot"])
def test_image_start_command_and_healthcheck(monkeypatch, dataset) -> None:
    import urllib.request

    source = (ROOT / "experiments" / dataset / "Dockerfile").read_text()
    command = json.loads(source.split("\nCMD ")[1].strip())
    assert command == [
        "sh", "-c",
        'exec uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT"',
    ]
    assert "ENV API_PORT=9000" in source
    assert "EXPOSE 9000" in source
    health = source.split('CMD python -c "')[1].split('"')[0]
    calls = []
    monkeypatch.setenv("API_PORT", "9100")
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda url, timeout: calls.append((url, timeout)),
    )
    exec(health, {})
    assert calls == [("http://127.0.0.1:9100/health", 2)]
