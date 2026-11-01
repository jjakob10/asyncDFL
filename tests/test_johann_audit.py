"""Small on-disk evidence fixtures; the auditor must never call Docker/training."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from experiments.johann import analysis
from experiments.johann.audit import audit_m0
from experiments.johann.catalog import SCENARIOS
from experiments.johann.frozen import FrozenBundle
from scripts import run_johann_experiments as runner


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def file_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def audit_bundle(tmp_path):
    folder = tmp_path / "bundle"
    folder.mkdir()
    files = {"app/runtime.py": b"frozen"}
    datasets, selected, images = {}, {}, {}
    for ds, digit in (("mnist", "a"), ("cifar10", "b")):
        raw = (runner.ROOT / f"experiments/johann/{ds}/manifest.json").read_bytes()
        files[f"experiments/johann/{ds}/manifest.json"] = raw
        images[ds] = {"id": "sha256:" + digit * 64, "tag": ds + ":frozen"}
        datasets[ds] = {"methods": {"M0": {}}, "protocol_frozen": False}
        selected[ds] = {"methods": {"M0": {}}, "protocol_frozen": True, "frozen_image": images[ds]}
        (folder / f"{ds}_dependencies.lock.txt").write_text("torch==frozen\n")
    with zipfile.ZipFile(folder / "source_snapshot.zip", "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    manifest = {"protocol_frozen": True, "calibration_id": "calibration", "confirmation_id": "confirm",
                "datasets": datasets, "images": images,
                "source_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()},
                "source_archive_sha256": hashlib.sha256((folder / "source_snapshot.zip").read_bytes()).hexdigest(),
                "confirmation_runs": [f"C:\\preserved\\confirm-s{seed}" for seed in (8, 9)]}
    write_json(folder / "manifest.json", manifest)
    write_json(folder / "selected_configuration.json", {"calibration_id": "calibration", "datasets": selected})
    return FrozenBundle(folder)


def write_run(root, bundle, ds="mnist", scenario="concentrated-r", seed=8, batch="confirm",
              current=True, legacy=False, diagnostics=True, mode="pilot", suffix=""):
    spec = bundle.resolve_spec(ds, scenario, seed, mode, "M0")
    spec.update(run_id=f"{batch}-s{seed}{suffix}", batch_run_id=batch,
                image_id=bundle.manifest["images"][ds]["id"] if current else "sha256:" + "c" * 64)
    if not current:
        spec["source_sha256"] = {"app/runtime.py": "old"}
        for key in ("method_parameters", "training_age_unit", "execution_backend", "frozen_bundle", "expected_image_id"):
            spec.pop(key, None)
    directory = root / ds / ("historical" if legacy else "M0") / scenario / spec["run_id"]
    directory.mkdir(parents=True)
    write_json(directory / "resolved_experiment.json", spec)
    write_json(directory / "outcome.json", {"state": "completed", "run_id": spec["run_id"]})
    duration, interval = spec["duration_seconds"], spec["evaluation_interval_seconds"]
    for client in range(8):
        identity = {"run_id": spec["run_id"], "client_id": client}
        write_json(directory / f"client_{client}_cost.json", {**identity, "state": "completed"})
        write_json(directory / f"client_{client}_class_distribution.json", {
            **identity, "counts": [10] * 10, "training_indices_sha256": f"train-{seed}",
            "validation_indices_sha256": f"validation-{seed}",
            "local_indices_sha256": f"{scenario.split('-')[0]}-local-{client}"})
        evaluations = [{**identity, "kind": "periodic", "captured_seconds": step * interval + .01,
                        "evaluation_split": spec["evaluation_split"],
                        "test": {"recall_tracked": .8, "support_3": 10, "support_8": 10}}
                       for step in range(int(duration / interval) + 1)]
        if diagnostics and client in spec["offline_client_ids"]:
            evaluations.extend({**deepcopy(evaluations[-1]), "kind": kind}
                               for kind in ("preserved_at_return", "before_first_mix", "after_first_mix"))
        events = [{**identity, "event": "started", "elapsed_seconds": .01}]
        if client in spec["offline_client_ids"]:
            events.extend([{**identity, "event": "offline", "elapsed_seconds": duration * .3},
                           {**identity, "event": "returned", "elapsed_seconds": duration * .6}])
        for kind, rows in (("evaluations", evaluations), ("events", events),
                           ("metrics", [{**identity, "elapsed_seconds": 1}])):
            (directory / f"client_{client}_{kind}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    image = {"Id": spec["image_id"], "Architecture": "amd64", "Os": "linux",
             "RootFS": {"Layers": ["base", "current" if current else "old"]},
             "Config": {"Env": ["X=frozen" if current else "X=old"],
                        "Labels": {"com.docker.compose.project": scenario,
                                   "com.docker.compose.service": "client-0", "com.docker.compose.version": "test"}}}
    write_json(directory / "image.json", [image])
    (directory / "container_dependencies.txt").write_text("torch==frozen\n" if current else "torch==old\n")
    return directory


def references(root, bundle, ds="mnist"):
    return [write_run(root, bundle, ds, seed=seed) for seed in (8, 9)]


def test_full_matrix_inventory_and_read_only_cli(audit_bundle, tmp_path, monkeypatch):
    root, out = tmp_path / "results", tmp_path / "report"
    for ds in ("mnist", "cifar10"):
        for seed in (8, 9):
            for scenario in SCENARIOS:
                write_run(root, audit_bundle, ds, scenario, seed, legacy=seed == 8)
    original = file_hashes(root)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("audit must not call subprocess/Docker"))
    monkeypatch.setattr(analysis, "summarize_run", lambda *a: pytest.fail("must not rewrite original summaries"))
    monkeypatch.setattr(analysis, "summarize_collection", lambda *a: pytest.fail("must not rewrite collections"))
    assert runner.main(["audit-m0", "--results-root", str(root), "--frozen-bundle", str(audit_bundle.directory),
                        "--output-dir", str(out)]) == 0
    audit = json.loads((out / "auditoria.json").read_text())
    assert audit["inventory_records"] == audit["frozen_compatible_runs"] == 24
    assert audit["e2_gap_count"] == 0
    assert len(csv_rows(out / "lacunas_E2.csv")) == 0
    comparisons = csv_rows(out / "comparacoes_disponiveis.csv")
    assert len(comparisons) == 28
    assert all(r["absence_status"] == r["post_return_status"] == "available" for r in comparisons)
    pair = next(r for r in comparisons if r["comparison"] == "concentrated_c_minus_none")
    observers = json.loads(pair["observers"])
    assert observers[0]["client_ids"] == observers[1]["client_ids"] == [0, 1, 3, 4, 5, 7]
    assert file_hashes(root) == original
    assert not audit["experiments_started"] and not audit["official_evaluations_read"]
    assert all("auc" not in key for r in comparisons for key in r)


def test_old_pilots_are_usable_only_within_their_batch(audit_bundle, tmp_path):
    root = tmp_path / "results"
    references(root, audit_bundle)
    for scenario in SCENARIOS:
        write_run(root, audit_bundle, scenario=scenario, batch="old", current=False, diagnostics=False, legacy=True)
    out = tmp_path / "report"
    audit_m0(root, audit_bundle.directory, out)
    inventory = csv_rows(out / "inventario_execucoes.csv")
    old = [r for r in inventory if r["batch"] == "old"]
    assert len(old) == 6
    assert all(r["historical_use"] == "eligible_within_batch" and r["frozen_use"] == "incompatible" for r in old)
    assert all(r["status"] == "completed" for r in old)
    assert all(r["absence_status"] == r["post_return_status"] == "available"
               for r in csv_rows(out / "comparacoes_disponiveis.csv") if r["batch"] == "old")
    assert len(csv_rows(out / "lacunas_E2.csv")) == 22  # two current confirmations cover two cells


def test_numeric_json_representation_does_not_create_a_protocol_difference(audit_bundle, tmp_path):
    root, out = tmp_path / "results", tmp_path / "report"
    references(root, audit_bundle)
    directory = write_run(root, audit_bundle, scenario="concentrated-none")
    spec = json.loads((directory / "resolved_experiment.json").read_text())
    spec["duration_seconds"] = float(spec["duration_seconds"])
    spec["evaluation_interval_seconds"] = float(spec["evaluation_interval_seconds"])
    write_json(directory / "resolved_experiment.json", spec)
    audit = audit_m0(root, audit_bundle.directory, out)
    assert audit["frozen_compatible_runs"] == 3
    pair = next(r for r in csv_rows(out / "comparacoes_disponiveis.csv")
                if r["comparison"] == "concentrated_r_minus_none" and r["seed"] == "8")
    assert pair["absence_status"] == pair["post_return_status"] == "available"


@pytest.mark.parametrize("damage", ["cost", "identity", "json", "gap", "code", "environment", "data", "cadence", "events"])
def test_bad_or_incompatible_evidence_blocks_only_its_comparisons(audit_bundle, tmp_path, damage):
    root, out = tmp_path / "results", tmp_path / "report"
    references(root, audit_bundle)
    directory = write_run(root, audit_bundle, scenario="concentrated-none")
    if damage == "cost":
        (directory / "client_7_cost.json").unlink()
    elif damage == "identity":
        value = json.loads((directory / "client_7_cost.json").read_text())
        value["run_id"] = "other"
        write_json(directory / "client_7_cost.json", value)
    elif damage == "json":
        (directory / "client_7_metrics.jsonl").write_text("corrupted\n")
    elif damage == "gap":
        path = directory / "client_7_evaluations.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("\n".join(json.dumps(r) for r in rows if not 200 < r["captured_seconds"] < 330))
    elif damage in ("code", "cadence"):
        path = directory / "resolved_experiment.json"
        value = json.loads(path.read_text())
        if damage == "code":
            value["source_sha256"] = {"app/runtime.py": "changed"}
        else:
            value["evaluation_interval_seconds"] = 19
        write_json(path, value)
    elif damage == "environment":
        path = directory / "image.json"
        value = json.loads(path.read_text())
        value[0]["Config"]["Env"] = ["changed"]
        write_json(path, value)
    elif damage == "data":
        path = directory / "client_7_class_distribution.json"
        value = json.loads(path.read_text())
        value["local_indices_sha256"] = "different partition"
        write_json(path, value)
    else:
        (directory / "client_7_events.jsonl").write_text("[]\n")
    audit_m0(root, audit_bundle.directory, out)
    pair = next(r for r in csv_rows(out / "comparacoes_disponiveis.csv")
                if r["comparison"] == "concentrated_r_minus_none" and r["seed"] == "8")
    assert pair["absence_status"] == "blocked"
    if damage == "gap":
        assert pair["post_return_status"] == "available"
    else:
        assert pair["post_return_status"] == "blocked"


def test_generated_failed_smoke_and_test_are_distinguished(audit_bundle, tmp_path):
    root, out = tmp_path / "results", tmp_path / "report"
    references(root, audit_bundle)
    config = root / "static"
    config.mkdir()
    spec = audit_bundle.resolve_spec("mnist", "concentrated-r", 7, "pilot", "M0")
    spec.update(run_id="static", batch_run_id="static")
    write_json(config / "resolved_experiment.json", spec)
    failed = write_run(root, audit_bundle, batch="failed", scenario="distributed-c")
    write_json(failed / "outcome.json", {"state": "failed"})
    write_run(root, audit_bundle, mode="smoke", batch="smoke")
    official = write_run(root, audit_bundle, mode="collection", seed=42, batch="official")
    # Official payloads are deliberately unreadable: only metadata may be read.
    (official / "client_0_evaluations.jsonl").write_text("must not read")
    bad = root / "bad"
    bad.mkdir()
    (bad / "resolved_experiment.json").write_text("invalid JSON")
    audit = audit_m0(root, audit_bundle.directory, out)
    rows = csv_rows(out / "inventario_execucoes.csv")
    states = {r["status"] for r in rows}
    assert {"configuration_only", "smoke_completed", "not_completed_failed", "invalid_configuration",
            "official_test_metadata_only", "completed"} <= states
    assert str(official / "client_0_evaluations.jsonl") not in audit["input_sha256"]
    assert all(r["seed"] != "42" for r in csv_rows(out / "lacunas_E2.csv"))
    assert not any(r["batch"] == "static" for r in csv_rows(out / "comparacoes_disponiveis.csv"))


def test_unknown_reference_requires_review_instead_of_new_training(audit_bundle, tmp_path):
    root = tmp_path / "results"
    write_run(root, audit_bundle, seed=8)
    bad = write_run(root, audit_bundle, scenario="distributed-r", batch="bad")
    value = json.loads((bad / "resolved_experiment.json").read_text())
    value["batch_run_id"] = ["malformed batch"]
    write_json(bad / "resolved_experiment.json", value)
    out = tmp_path / "report"
    audit = audit_m0(root, audit_bundle.directory, out)
    assert any("mnist/seed=9" in issue for issue in audit["reference_issues"])
    assert any("cifar10" in issue for issue in audit["reference_issues"])
    assert len(audit["frozen_references"]) == 1
    missing = [r for r in csv_rows(out / "lacunas_E2.csv") if r["seed"] == "9"]
    assert len(missing) == 12
    assert all(r["status"] == "review_reference" for r in missing)
    assert any(r["status"] == "incomplete" for r in csv_rows(out / "inventario_execucoes.csv"))


@pytest.mark.parametrize("different_runtime", [False, True])
def test_reference_images_with_different_compose_ids_need_exact_frozen_content_evidence(audit_bundle, tmp_path, different_runtime):
    root, out = tmp_path / "results", tmp_path / "report"
    refs = references(root, audit_bundle)
    write_run(root, audit_bundle, scenario="concentrated-c")  # exact preserved image ID anchor
    for directory in refs:
        spec = json.loads((directory / "resolved_experiment.json").read_text())
        spec["image_id"] = "sha256:" + "d" * 64
        write_json(directory / "resolved_experiment.json", spec)
        image = json.loads((directory / "image.json").read_text())
        image[0]["Id"] = spec["image_id"]
        image[0]["Config"]["Labels"]["com.docker.compose.project"] = "different-compose-label"
        if different_runtime:
            image[0]["Config"]["Env"] = ["different actual environment"]
        write_json(directory / "image.json", image)
    audit = audit_m0(root, audit_bundle.directory, out)
    assert len(audit["frozen_references"]) == (0 if different_runtime else 2)
    assert audit["frozen_compatible_runs"] == (0 if different_runtime else 3)


def test_duplicates_are_not_selected_and_do_not_trigger_automatic_rerun(audit_bundle, tmp_path):
    root, out = tmp_path / "results", tmp_path / "report"
    references(root, audit_bundle)
    write_run(root, audit_bundle, scenario="concentrated-none")
    write_run(root, audit_bundle, scenario="concentrated-none", suffix="-duplicate")
    audit_m0(root, audit_bundle.directory, out)
    gap = next(r for r in csv_rows(out / "lacunas_E2.csv") if r["dataset"] == "mnist"
               and r["seed"] == "8" and r["scenario"] == "concentrated-none")
    assert gap["status"] == "review_duplicates"
    pair = next(r for r in csv_rows(out / "comparacoes_disponiveis.csv")
                if r["comparison"] == "concentrated_r_minus_none" and r["seed"] == "8")
    assert "duplicate" in pair["absence_reasons"]


def test_output_must_not_overwrite_originals_or_nonempty_report(audit_bundle, tmp_path):
    root = tmp_path / "results"
    directory = references(root, audit_bundle)[0]
    for output in (directory / "report", audit_bundle.directory / "report", directory):
        with pytest.raises(ValueError):
            audit_m0(root, audit_bundle.directory, output)
    out = tmp_path / "report"
    audit_m0(root, audit_bundle.directory, out)
    original = file_hashes(out)
    with pytest.raises(ValueError, match="new or empty"):
        audit_m0(root, audit_bundle.directory, out)
    assert file_hashes(out) == original
