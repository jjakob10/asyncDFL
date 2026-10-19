from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from experiments.paper.catalog import build_catalog, load_manifest
from scripts.experiment_compose import build_compose
from scripts.export_manifest import export_manifest, main


ROOT = Path(__file__).resolve().parents[1]


def compose_config(directory: Path) -> dict:
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose is not installed")
    environment = os.environ.copy()
    environment.pop("RUN_ID", None)
    result = subprocess.run(
        ["docker", "compose", "-f", str(directory / "compose.yaml"),
         "config", "--format", "json"],
        cwd=directory, env=environment, capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def raw_environment(path: Path) -> dict[str, str]:
    return {key: value[1:-1] for key, value in (
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
    )}


@pytest.mark.parametrize("dataset", ["mnist", "cifar10"])
def test_all_thiago_exports_match_runner_configuration(dataset, tmp_path) -> None:
    manifest = ROOT / "experiments" / "thiago" / dataset / "manifest.json"
    catalog = build_catalog(load_manifest(manifest))
    directories = export_manifest(manifest, tmp_path / "export with spaces", "test-run")
    assert len(directories) == 6
    for experiment, directory in zip(catalog, directories):
        expected_directory = ROOT / "results" / "thiago" / dataset / experiment.experiment_id / "test-run"
        expected = build_compose(experiment, "test-run", expected_directory)
        exported = json.loads((directory / "compose.yaml").read_text(encoding="utf-8"))
        compose = compose_config(directory)
        assert compose["name"] == expected["name"]
        assert set(compose["services"]) == set(expected["services"])
        common = raw_environment(directory / ".env")
        assert common["RUN_ID"] == "test-run"
        for client in experiment.clients:
            name = f"client-{client.client_id}"
            service = compose["services"][name]
            assert service["environment"] == expected["services"][name]["environment"]
            assert len(service["healthcheck"]["test"]) == 4
            assert service["ports"][0]["published"] == str(8000 + client.client_id)
            assert service["ports"][0]["target"] == 9000
            assert Path(service["volumes"][0]["source"]) == expected_directory
            assert exported["services"][name]["volumes"][0]["bind"]["create_host_path"] is True
            individual = raw_environment(directory / "env" / f"client_{client.client_id}.env")
            assert individual["CLIENT_ID"] == str(client.client_id)
            assert individual["NEIGHBORS"] == expected["services"][name]["environment"]["NEIGHBORS"]
        build = compose["services"]["client-0"]["build"]
        assert Path(build["context"]) == ROOT
        assert build["dockerfile"] == experiment.dockerfile
        assert (directory / "env" / "client_5.env").is_file()
        assert not (directory / "results").exists()


def test_changing_run_id_and_training_settings_changes_effective_compose(tmp_path) -> None:
    manifest = ROOT / "experiments/thiago/mnist/manifest.json"
    directory = export_manifest(manifest, tmp_path / "export", "first-run")[0]
    path = directory / ".env"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("RUN_ID='first-run'", "RUN_ID='second-run'")
                   .replace("ROUNDS='10'", "ROUNDS='3'"), encoding="utf-8")
    compose = compose_config(directory)
    for service in compose["services"].values():
        assert service["environment"]["RUN_ID"] == "second-run"
        assert service["environment"]["ROUNDS"] == "3"
        assert Path(service["volumes"][0]["source"]).name == "second-run"


def test_paper_manifest_exports_all_configs_and_preserves_control_clients(tmp_path) -> None:
    manifest = ROOT / "experiments/paper/manifest.json"
    csv_path = tmp_path / "external data $cache's" / "Train_Test_Network.csv"
    csv_path.parent.mkdir()
    csv_path.write_text("ts,type,label\n1,normal,0\n", encoding="utf-8")
    directories = export_manifest(manifest, tmp_path / "paper", "paper-run", ton_iot_path=csv_path)
    assert len(directories) == 158
    catalog = build_catalog(load_manifest(manifest))
    for experiment, directory in zip(catalog, directories):
        common = raw_environment(directory / ".env")
        assert common["PARTITION_MODE"] == "iid"
        expected = build_compose(experiment, "paper-run", tmp_path, ton_iot_path=csv_path)
        for client in experiment.clients:
            individual = raw_environment(directory / "env" / f"client_{client.client_id}.env")
            shared = {key: value for key, value in common.items() if not key.startswith("EXPORT_")}
            assert {**shared, **individual} == expected["services"][f"client-{client.client_id}"]["environment"]
    for experiment_id in ("mnist-8p2-top4-e32", "cifar10-8p2-top0-e2", "ton_iot-8p2-top4-e32"):
        compose = compose_config(tmp_path / "paper" / experiment_id)
        assert compose["services"]["client-9"]["environment"]["NEIGHBORS"] == ""
        assert compose["services"]["client-9"]["environment"]["PAPER_ROLE"] == "ED"
        assert compose["services"]["client-0"]["environment"]["EMPTY_CLIENT_IDS"] == "0,4,9"
        if experiment_id.startswith("ton_iot"):
            mount = compose["services"]["client-0"]["volumes"][1]
            # Canonical Compose output escapes dollars for reuse as a Compose file.
            assert Path(mount["source"].replace("$$", "$")) == csv_path
            assert mount["read_only"] is True
            assert mount["bind"]["create_host_path"] is False


def test_smoke_cli_and_default_timestamp(tmp_path) -> None:
    manifest = ROOT / "experiments/thiago/mnist/manifest.json"
    output = tmp_path / "smoke"
    assert main(["--manifest", str(manifest), "--output", str(output), "--smoke"]) == 0
    ids = set()
    for directory in output.iterdir():
        env = raw_environment(directory / ".env")
        assert env["ROUNDS"] == env["LOCAL_EPOCHS"] == "1"
        assert env["MAX_TRAIN_SAMPLES"] == "400"
        assert env["MAX_TEST_SAMPLES"] == "200"
        ids.add(env["RUN_ID"])
    assert len(ids) == 1
    assert ids.pop().startswith("export-")


@pytest.mark.parametrize("existing", ["file", "directory"])
def test_export_never_overwrites_occupied_destination(tmp_path, existing) -> None:
    output = tmp_path / "occupied"
    if existing == "file":
        original = output
    else:
        output.mkdir()
        original = output / ".env"
    original.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="new or empty"):
        export_manifest(ROOT / "experiments/thiago/mnist/manifest.json", output)
    assert original.read_text(encoding="utf-8") == "preserve"
    if output.is_dir():
        assert list(output.iterdir()) == [original]


@pytest.mark.parametrize("changes", [
    {"experiment_id": "../escape"}, {"experiment_id": "C:\\escape"},
    {"experiment_id": "CON"}, {"experiment_id": "con"},
    {"partition_mode": "unknown"}, {"client_classes": {"0": [0]}},
    {"dockerfile": "missing/Dockerfile"}, {"dataset": "missing"},
])
def test_invalid_last_experiment_does_not_write_partial_export(tmp_path, changes) -> None:
    data = load_manifest(ROOT / "experiments/thiago/mnist/manifest.json")
    data["experiments"][-1].update(changes)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    output = tmp_path / "output"
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize("run_id", ["../escape", "", "with space", "CON", "x" * 129])
def test_invalid_run_id_does_not_create_output(tmp_path, run_id) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="run ID"):
        export_manifest(ROOT / "experiments/thiago/mnist/manifest.json", output, run_id)
    assert not output.exists()


def test_missing_external_dataset_and_malformed_json_leave_output_unchanged(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(ValueError, match="ton-iot-path"):
        export_manifest(ROOT / "experiments/paper/manifest.json", output)
    assert list(output.iterdir()) == []
    manifest = tmp_path / "bad.json"
    manifest.write_text("{", encoding="utf-8")
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 2
    assert list(output.iterdir()) == []
