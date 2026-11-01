"""Frozen execution tests use tiny source bundles and a simulated Docker engine."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from experiments.johann import analysis
from experiments.johann.catalog import METHODS, compose_for
from experiments.johann.frozen import FrozenBundle
from scripts import run_johann_experiments as runner


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bundle_path(tmp_path):
    directory = tmp_path / "bundle"
    directory.mkdir()
    selected, images, files = {}, {}, {"app/runtime.py": b"preserved training source"}
    for dataset, limit, weight, digit in (("mnist", 33750, .1, "a"), ("cifar10", 56250, .5, "b")):
        methods = {"M0": {}, "M1": {}, "M2": {"merge_limit_examples": limit},
                   "M3": {"rejoin_batches": 64},
                   "M4": {"rejoin_batches": 64, "distillation_weight": weight, "distillation_temperature": 2}}
        image = {"id": "sha256:" + digit * 64, "tag": f"dfl-johann-{dataset}:frozen-test"}
        selected[dataset] = {"methods": methods, "protocol_frozen": True, "frozen_image": image}
        images[dataset] = image
        manifest = json.loads((runner.ROOT / f"experiments/johann/{dataset}/manifest.json").read_text())
        files[f"experiments/johann/{dataset}/manifest.json"] = json.dumps(manifest).encode()
    archive_path = directory / "source_snapshot.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    # Match the historical bundle: only the global manifest flag is authoritative.
    manifest = {"protocol_frozen": True, "calibration_id": "tune", "images": images,
                "datasets": {ds: {"methods": x["methods"], "protocol_frozen": False} for ds, x in selected.items()},
                "source_sha256": {name: digest(value) for name, value in files.items()},
                "source_archive_sha256": digest(archive_path.read_bytes())}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "selected_configuration.json").write_text(json.dumps({"calibration_id": "tune", "datasets": selected}))
    return directory


@pytest.mark.parametrize("dataset,limit,weight", [("mnist", 33750, .1), ("cifar10", 56250, .5)])
def test_frozen_resolution_uses_archived_manifests_and_selected_methods(bundle_path, monkeypatch, dataset, limit, weight):
    original_read = Path.read_text

    def refuse_workspace_defaults(path, *args, **kwargs):
        if path == runner.ROOT / f"experiments/johann/{dataset}/manifest.json":
            pytest.fail("frozen resolution must not read the workspace manifest")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse_workspace_defaults)
    bundle = FrozenBundle(bundle_path)
    for method in METHODS:
        spec = bundle.resolve_spec(dataset, "concentrated-r", 42, "collection", method)
        assert spec["protocol_frozen"] and spec["evaluation_split"] == "test"
        assert spec["method_parameters"] == bundle.selection["datasets"][dataset]["methods"][method]
        assert spec["methods"]["M3"] == {"rejoin_batches": 64}
        assert spec["methods"]["M2"] == {"merge_limit_examples": limit}
        assert spec["methods"]["M4"]["distillation_weight"] == weight
        assert spec["source_sha256"]["app/runtime.py"] == digest(b"preserved training source")
        compose = compose_for(spec, "frozen", Path("output"))
        assert compose["name"] == compose["name"].lower()
        assert len(compose["services"]) == 8
        for service in compose["services"].values():
            assert service["image"] == spec["expected_image_id"]
            assert service["pull_policy"] == "never" and "build" not in service
            assert service["environment"]["MAX_TRAIN_SAMPLES"] == "0"
            assert service["environment"]["METHOD"] == method
            if method in ("M3", "M4"):
                assert service["environment"]["REJOIN_BATCHES"] == "64"


@pytest.mark.parametrize("problem", ["global_flag", "selection_flag", "calibration", "parameters", "image",
                                    "archive_hash", "source_hash", "missing_manifest", "incomplete", "numeric"])
def test_inconsistent_bundle_is_rejected(bundle_path, problem):
    manifest = bundle_path / "manifest.json"
    selection = bundle_path / "selected_configuration.json"
    if problem == "global_flag":
        rewrite(manifest, lambda x: x.update(protocol_frozen=False))
    elif problem == "selection_flag":
        rewrite(selection, lambda x: x["datasets"]["mnist"].update(protocol_frozen=False))
    elif problem == "calibration":
        rewrite(selection, lambda x: x.update(calibration_id="other"))
    elif problem == "parameters":
        rewrite(selection, lambda x: x["datasets"]["mnist"]["methods"]["M3"].update(rejoin_batches=16))
    elif problem == "image":
        rewrite(selection, lambda x: x["datasets"]["mnist"]["frozen_image"].update(id="sha256:" + "c" * 64))
    elif problem == "archive_hash":
        with (bundle_path / "source_snapshot.zip").open("ab") as stream:
            stream.write(b"changed archive")
    elif problem == "source_hash":
        rewrite(manifest, lambda x: x["source_sha256"].update({"app/runtime.py": "0" * 64}))
    elif problem == "missing_manifest":
        rewrite(manifest, lambda x: x["source_sha256"].pop("experiments/johann/mnist/manifest.json"))
    else:
        for path in (manifest, selection):
            if problem == "incomplete":
                rewrite(path, lambda x: x["datasets"]["mnist"]["methods"]["M4"].pop("rejoin_batches"))
            else:
                rewrite(path, lambda x: x["datasets"]["mnist"]["methods"]["M3"].update(rejoin_batches=0))
    with pytest.raises(ValueError):
        FrozenBundle(bundle_path)


def test_requested_method_or_dataset_must_be_available(bundle_path):
    for file in ("manifest.json", "selected_configuration.json"):
        rewrite(bundle_path / file, lambda x: x["datasets"]["mnist"]["methods"].pop("M1"))
    bundle = FrozenBundle(bundle_path)
    with pytest.raises(ValueError, match="method not available"):
        bundle.resolve_spec("mnist", "concentrated-r", 8, "pilot", "M1")
    with pytest.raises(ValueError, match="dataset not available"):
        bundle.resolve_spec("unknown", "concentrated-r", 8, "pilot", "M0")


def test_run_requires_bundle_before_any_artifacts_or_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "prepare", lambda *args: pytest.fail("must not prepare"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not call subprocess"))
    assert runner.main(["run", "--results-root", str(tmp_path / "results")]) == 2
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("arguments", [
    ["pilot", "--duration", "600"], ["pilot", "--interval", "20"],
    ["run", "--duration", "600"], ["render-compose", "--interval", "20"],
    ["pilot", "--selected-config", "unused.json"], ["pilot", "--merge-limit-examples", "33750"],
    ["pilot", "--merge-limit-multiplier", "5"], ["pilot", "--rejoin-batches", "64"],
    ["pilot", "--distillation-weight", "0.1"], ["pilot", "--distillation-temperature", "2"],
    ["smoke", "--backend", "local"],
])
def test_incompatible_options_fail_before_preparing(bundle_path, monkeypatch, arguments):
    monkeypatch.setattr(runner, "prepare", lambda *args: pytest.fail("must not prepare"))
    assert runner.main([*arguments, "--frozen-bundle", str(bundle_path)]) == 2


def test_render_all_methods_is_offline_and_preserves_provenance(bundle_path, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "inspect_frozen_image", lambda *args: pytest.fail("render must not contact Docker"))
    monkeypatch.setattr(runner, "execute", lambda *args: pytest.fail("render must not execute"))
    assert runner.main(["render-compose", "--frozen-bundle", str(bundle_path), "--dataset", "all",
                        "--methods", *METHODS, "--run-id", "render", "--results-root", str(tmp_path / "results")]) == 0
    specs = list((tmp_path / "results").rglob("resolved_experiment.json"))
    assert len(specs) == 10
    for path in specs:
        spec = json.loads(path.read_text())
        assert "code_revision" not in spec and spec["orchestrator"]["code_revision"]
        assert spec["source_sha256"]["app/runtime.py"] != spec["orchestrator"]["source_sha256"]["app/runtime.py"]
        assert spec["frozen_bundle"]["manifest_sha256"] == digest((path.parent / "frozen_manifest.json").read_bytes())
        assert spec["frozen_bundle"]["selection_sha256"] == digest((path.parent / "selected_configuration.json").read_bytes())
        assert spec["evaluation_split"] == "validation" and spec["protocol_frozen"]
        compose = json.loads((path.parent / "compose.json").read_text())
        assert all(service["image"] == spec["expected_image_id"] for service in compose["services"].values())


@pytest.mark.parametrize("command", ["run", "pilot", "smoke"])
def test_cli_execution_resolves_frozen_parameters(bundle_path, tmp_path, monkeypatch, command):
    preflight, executed = [], []
    monkeypatch.setattr(runner, "inspect_frozen_image", lambda s: preflight.append(s["dataset"]))
    monkeypatch.setattr(runner, "execute", lambda d, s: executed.append(s) or True)
    extra = ["--duration", "45", "--interval", "2"] if command == "smoke" else []
    assert runner.main([command, "--dataset", "all", "--methods", *METHODS,
                        "--frozen-bundle", str(bundle_path), "--seeds", "8", "--run-id", command,
                        "--results-root", str(tmp_path / "results"), *extra]) == 0
    assert preflight == ["mnist", "cifar10"] and len(executed) == 10
    for spec in executed:
        assert spec["method_parameters"] == json.loads((bundle_path / "selected_configuration.json").read_text())["datasets"][spec["dataset"]]["methods"][spec["method"]]
        assert spec["evaluation_split"] == ("test" if command == "run" else "validation")
        assert spec["protocol_frozen"] == (command != "smoke")
        if command == "smoke":
            assert spec["duration_seconds"] == 45 and spec["evaluation_interval_seconds"] == 2
            service = compose_for(spec, "smoke", tmp_path)["services"]["client-0"]
            assert service["environment"]["MAX_TRAIN_SAMPLES"] == "800"
            assert service["environment"]["MAX_TEST_SAMPLES"] == "100"


def test_missing_second_dataset_image_blocks_entire_selection(bundle_path, tmp_path, monkeypatch, capsys):
    def image_inspect(command, **kwargs):
        assert command[:3] == ["docker", "image", "inspect"]
        missing = command[-1] == "sha256:" + "b" * 64
        return subprocess.CompletedProcess(command, 1 if missing else 0,
                                           stdout=json.dumps([{"Id": command[-1]}]), stderr="missing" if missing else "")

    monkeypatch.setattr(runner.subprocess, "run", image_inspect)
    monkeypatch.setattr(runner, "prepare", lambda *args: pytest.fail("all images must be checked before preparing"))
    assert runner.main(["run", "--dataset", "all", "--frozen-bundle", str(bundle_path),
                        "--results-root", str(tmp_path / "results")]) == 2
    assert "docker load --input" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_list_with_bundle_does_not_need_docker(bundle_path, monkeypatch, capsys):
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("list is offline"))
    assert runner.main(["list", "--dataset", "all", "--methods", *METHODS,
                        "--frozen-bundle", str(bundle_path)]) == 0
    output = capsys.readouterr().out
    assert "33750" in output and "56250" in output and "64" in output


def test_image_id_disagreement_does_not_fall_back_to_tags(bundle_path, monkeypatch):
    spec = FrozenBundle(bundle_path).resolve_spec("mnist", "concentrated-r", 8, "pilot", "M0")
    calls = []

    def wrong_image(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps([{"Id": "sha256:" + "c" * 64}]))

    monkeypatch.setattr(runner.subprocess, "run", wrong_image)
    with pytest.raises(ValueError, match="expected ID"):
        runner.inspect_frozen_image(spec)
    assert calls == [["docker", "image", "inspect", spec["expected_image_id"]]]


def test_image_removed_before_up_does_not_start_or_clean_other_containers(bundle_path, tmp_path, monkeypatch):
    bundle = FrozenBundle(bundle_path)
    spec = bundle.resolve_spec("mnist", "concentrated-r", 8, "pilot", "M0")
    directory, spec = runner.prepare(spec, "removed", tmp_path / "results", bundle)

    def missing(command, **kwargs):
        assert command == ["docker", "image", "inspect", spec["expected_image_id"]]
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="image removed")

    monkeypatch.setattr(runner.subprocess, "run", missing)
    monkeypatch.setattr(runner, "schedule_run", lambda *args: pytest.fail("must not schedule"))
    assert runner.execute(directory, spec) is False
    outcome = json.loads((directory / "outcome.json").read_text())
    assert outcome["state"] == "failed" and "docker load --input" in outcome["error"]


def setup_engine(directory, spec, monkeypatch, defect=None):
    calls, events = [], []
    project = json.loads((directory / "compose.json").read_text())["name"]
    containers = [{"Id": f"{client + 1:064x}", "Image": spec["expected_image_id"], "State": {"Running": True},
                   "Config": {"Image": "arbitrary:retagged", "Labels": {"com.docker.compose.project": project,
                              "com.docker.compose.service": f"client-{client}"}}} for client in range(8)]
    if defect == "missing":
        containers.pop()
    elif defect == "duplicate":
        containers[-1]["Config"]["Labels"]["com.docker.compose.service"] = "client-0"
    elif defect == "image":
        containers[-1]["Image"] = "sha256:" + "c" * 64
    elif defect == "stopped":
        containers[-1]["State"]["Running"] = False
    elif defect == "project":
        containers[-1]["Config"]["Labels"]["com.docker.compose.project"] = "other"
    elif defect == "duplicate_id":
        containers[-1]["Id"] = containers[0]["Id"]
    inspected_image = [{"Id": spec["expected_image_id"], "RootFS": {"Layers": ["same-layer"]}, "Config": {}}]

    def run(command, **kwargs):
        calls.append(command)
        output = ""
        if command[:3] == ["docker", "image", "inspect"]:
            assert command[-1] == spec["expected_image_id"]  # Never inspect tags.
            output = json.dumps(inspected_image)
        elif command[:2] == ["docker", "inspect"]:
            events.append("inspect-containers")
            output = json.dumps(containers)
        elif "--quiet" in command:
            output = "\n".join(c["Id"] for c in containers)
        elif "up" in command:
            assert command[4:] == ["up", "--no-build", "--pull", "never", "-d"]
            events.append("up")
        elif "exec" in command:
            output = "torch==test\n"
        elif "logs" in command:
            events.append("logs")
            output = "recorded logs"
        elif "down" in command:
            events.append("down")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", run)
    statuses = iter([[{"state": "waiting"}] * 8, [{"state": "completed"}] * 8])
    monkeypatch.setattr(runner, "api_statuses", lambda: next(statuses))

    def schedule(run_id):
        assert events[-1] == "inspect-containers"
        assert spec["observed_image_ids"] == {f"client-{c}": spec["expected_image_id"] for c in range(8)}
        events.append("start")
        return 1234

    monkeypatch.setattr(runner, "schedule_run", schedule)
    monkeypatch.setattr(analysis, "summarize_run", lambda path: events.append("summary"))
    return calls, events, inspected_image


def test_engine_image_ids_are_verified_before_start_and_analysis_remains_compatible(bundle_path, tmp_path, monkeypatch):
    bundle = FrozenBundle(bundle_path)
    spec = bundle.resolve_spec("mnist", "concentrated-r", 42, "collection", "M4")
    directory, spec = runner.prepare(spec, "verified", tmp_path / "results", bundle)
    before = {p.name: digest(p.read_bytes()) for p in bundle_path.iterdir()}
    calls, events, image = setup_engine(directory, spec, monkeypatch)
    assert runner.execute(directory, spec)
    assert events == ["up", "inspect-containers", "start", "summary", "logs", "down"]
    assert not any("--build" in c or "pull" in c or "load" in c for c in calls)
    assert json.loads((directory / "image.json").read_text()) == image
    assert len(json.loads((directory / "container_inspection.json").read_text())) == 8
    assert json.loads((directory / "outcome.json").read_text())["state"] == "completed"
    protocol = analysis.pairing_protocol(directory, spec)
    assert protocol["source_sha256"] == bundle.manifest["source_sha256"]
    assert protocol["image_filesystem_sha256"]
    assert before == {p.name: digest(p.read_bytes()) for p in bundle_path.iterdir()}


@pytest.mark.parametrize("defect", ["image", "missing", "duplicate", "stopped", "project", "duplicate_id"])
def test_container_mismatch_prevents_start_and_keeps_failure_evidence(bundle_path, tmp_path, monkeypatch, defect):
    bundle = FrozenBundle(bundle_path)
    spec = bundle.resolve_spec("cifar10", "concentrated-r", 8, "pilot", "M3")
    directory, spec = runner.prepare(spec, "mismatch", tmp_path / "results", bundle)
    _, events, _ = setup_engine(directory, spec, monkeypatch, defect)
    assert runner.execute(directory, spec) is False
    assert "start" not in events and "summary" not in events
    assert events[-2:] == ["logs", "down"]
    outcome = json.loads((directory / "outcome.json").read_text())
    assert outcome["state"] == "failed" and outcome["error"]
    assert (directory / "container_inspection.json").is_file()
    assert "recorded logs" in (directory / "container.log").read_text()
