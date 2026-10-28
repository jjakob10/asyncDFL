import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.johann.analysis import (
    integrated_outcome, observed_value, paired_bootstrap, recovery_time, summarize_collection, summarize_run,
    json_sha256, pairing_protocol, validate_paired_networks,
)
from experiments.johann.catalog import load_spec
from experiments.johann.trainer import metrics_from_counts


def row(at, value):
    return {"captured_seconds": at, "test": {"recall_tracked": value, "support_3": 5, "support_8": 5}}


def test_metrics_from_known_confusion():
    counts = np.eye(10, dtype=int) * 10
    counts[3, 3] = 5
    counts[3, 8] = 5
    metrics = metrics_from_counts(counts, .2)
    assert metrics["accuracy"] == .95
    assert metrics["recall_3"] == .5
    assert metrics["recall_8"] == 1
    assert metrics["recall_tracked"] == .75
    assert metrics["recall_other"] == 1
    assert metrics["macro_f1"] == pytest.approx((8 + 2 / 3 + .8) / 10)


def test_integration_uses_real_times_without_extrapolation_or_large_gaps():
    clients = {0: [row(0, .2), row(6, .4), row(8, .8), row(10.01, .9)]}
    result = integrated_outcome(clients, [0], 6, 10, 2)
    assert result["post_return_auc"] == pytest.approx(.6)
    assert result["complete"]
    clients[0] = [row(6.1, .4), row(8, .8), row(10, .9)]
    result = integrated_outcome(clients, [0], 6, 10, 2)
    assert result["post_return_auc"] is None
    assert result["coverage_fraction"] == pytest.approx(.975)
    clients[0] = [row(5, .2), row(10.1, .8)]
    assert integrated_outcome(clients, [0], 6, 10, 2)["coverage_fraction"] == 0
    assert observed_value(clients[0], 11, "recall_tracked", 4) is None


def test_recovery_requires_three_consecutive_points_and_preserves_failure():
    times = [6, 7, 8, 9, 10]
    assert recovery_time(times, [.8, .8, .8, .7, .8], .82, 6, 2) == 0
    assert recovery_time(times, [.8, None, .8, .7, .8], .82, 6, 2) is None
    assert recovery_time(times, [.8] * 5, None, 6, 2) is None
    assert paired_bootstrap([.2, .2, .2]) == pytest.approx((.2, .2))
    assert paired_bootstrap([.2]) == (None, None)


def write_network(root, scenario="concentrated-r", seed=42, quality=.8, mode="collection"):
    folder = root / scenario / str(seed)
    folder.mkdir(parents=True)
    spec = load_spec("mnist", scenario, seed, duration=10, interval=1, mode=mode)
    spec["run_id"] = f"{scenario}-{seed}"
    (folder / "resolved_experiment.json").write_text(json.dumps(spec))
    (folder / "outcome.json").write_text(json.dumps({"state": "completed"}))
    for client in range(8):
        identity = {"run_id": spec["run_id"], "client_id": client}
        rows = []
        for at in np.arange(.01, 10.02, 1):
            test = metrics_from_counts(np.eye(10, dtype=int))
            test["recall_tracked"] = quality
            rows.append({**identity, "captured_seconds": float(at), "model_version": int(at),
                         "kind": "periodic", "evaluation_split": spec["evaluation_split"], "test": test})
        (folder / f"client_{client}_evaluations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        distribution = {**identity, "counts": [10] * 10, "labels": [str(i) for i in range(10)],
                        "training_indices_sha256": str(seed), "validation_indices_sha256": f"v{seed}"}
        (folder / f"client_{client}_class_distribution.json").write_text(json.dumps(distribution))
        cost = {**identity, "state": "completed", "model_bytes_received": 100, "model_bytes_sent": 100,
                "optimizer_steps": 10, "examples_processed": 640, "train_seconds": 3,
                "evaluation_seconds": 1, "communication_seconds": .2, "peak_process_memory_bytes": None}
        (folder / f"client_{client}_cost.json").write_text(json.dumps(cost))
        (folder / f"client_{client}_events.jsonl").write_text("")
    return folder


def test_summaries_group_network_repetitions_and_generate_charts(tmp_path):
    first = write_network(tmp_path, seed=42, quality=.8)
    write_network(tmp_path, scenario="concentrated-c", seed=42, quality=.6)
    write_network(tmp_path, seed=43, quality=.9)
    write_network(tmp_path, scenario="concentrated-c", seed=43, quality=.6)
    summary = summarize_run(first)
    assert summary["outcomes"][0]["post_return_auc"] == pytest.approx(.8)
    assert len(summary["client_recovery"]) == 8
    comparisons = summarize_collection(tmp_path)
    contrast = next(r for r in comparisons if r["contrast"] == "concentrated_r_minus_c")
    assert contrast["paired_networks"] == 2
    assert contrast["mean_difference"] == pytest.approx(.25)
    assert list((first / "charts").glob("*.png"))
    assert (tmp_path / "paired_contrasts.png").stat().st_size > 0


def test_control_no_absence_reports_both_matched_online_groups(tmp_path):
    folder = write_network(tmp_path, scenario="distributed-none")
    summary = summarize_run(folder)
    assert {r["selection"] for r in summary["outcomes"]} == {"without_r", "without_c"}


def test_summary_rejects_mixed_run_artifacts(tmp_path):
    folder = write_network(tmp_path)
    path = folder / "client_0_cost.json"
    cost = json.loads(path.read_text())
    cost["run_id"] = "old"
    path.write_text(json.dumps(cost))
    with pytest.raises(ValueError, match="identity"):
        summarize_run(folder)


def test_collection_filters_select_one_batch_without_silent_rerun_selection(tmp_path, monkeypatch):
    from experiments.johann import analysis
    a = tmp_path / "first"
    b = tmp_path / "second"
    for directory, batch in ((a, "batch-a"), (b, "batch-b")):
        directory.mkdir()
        spec = {"run_id": f"{batch}-s42", "batch_run_id": batch, "mode": "collection"}
        (directory / "resolved_experiment.json").write_text(json.dumps(spec))
        (directory / "outcome.json").write_text(json.dumps({"state": "completed"}))
    called = []

    def summarize(directory):
        called.append(directory)
        return {"dataset": "mnist", "mode": "collection", "scenario": "concentrated-r", "seed": 42,
                "run_id": "batch-b-s42", "outcomes": []}

    monkeypatch.setattr(analysis, "summarize_run", summarize)
    monkeypatch.setattr(analysis, "_figure", lambda *a, **kw: None)
    assert summarize_collection(tmp_path, run_id="batch-b", mode="collection")
    assert called == [b]


def inspected_image(image_id, project):
    return {"Id": image_id, "Architecture": "amd64", "Os": "linux",
            "RootFS": {"Type": "layers", "Layers": ["sha256:base", "sha256:code"]},
            "Config": {"Env": ["TORCH_NUM_THREADS=1"], "Cmd": ["uvicorn", "app.main:app"],
                       "Labels": {"com.docker.compose.project": project,
                                  "com.docker.compose.service": "client-0",
                                  "com.docker.compose.version": "5.3.1", "experiment": "M0"}}}


def write_inspection(directory, image, dependencies="numpy==2.1.3\ntorch==2.6.0\n"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "image.json").write_text(json.dumps([image]))
    (directory / "container_dependencies.txt").write_text(dependencies)


def test_pairing_ignores_only_compose_image_metadata(tmp_path):
    a, b = tmp_path / "r", tmp_path / "none"
    first = inspected_image("sha256:r", "scenario-r")
    second = inspected_image("sha256:none", "scenario-none")
    second["Config"]["Labels"]["com.docker.compose.version"] = "5.4.0"
    write_inspection(a, first)
    write_inspection(b, second, "torch==2.6.0\r\nnumpy==2.1.3\r\n")
    spec = {"duration_seconds": 600, "source_sha256": {"app/runtime.py": "unchanged"}}
    assert pairing_protocol(a, {**spec, "image_id": "sha256:r"}) == pairing_protocol(
        b, {**spec, "image_id": "sha256:none"})


@pytest.mark.parametrize("difference,expected_field", [
    ("filesystem", "image_filesystem_sha256"),
    ("environment", "image_config_sha256"),
    ("command", "image_config_sha256"),
    ("platform", "image_platform"),
    ("custom_label", "image_config_sha256"),
    ("dependencies", "container_dependencies_sha256"),
    ("code", "source_sha256"),
    ("duration", "duration_seconds"),
    ("interval", "evaluation_interval_seconds"),
    ("training", "training_indices_sha256"),
    ("validation", "validation_indices_sha256"),
])
def test_pairing_still_rejects_real_differences(tmp_path, difference, expected_field):
    a, b = tmp_path / "r", tmp_path / "none"
    first = inspected_image("sha256:r", "scenario-r")
    second = inspected_image("sha256:none", "scenario-none")
    dependencies = "numpy==2.1.3\ntorch==2.6.0\n"
    if difference == "filesystem":
        second["RootFS"]["Layers"][-1] = "sha256:different-code"
    elif difference == "environment":
        second["Config"]["Env"] = ["TORCH_NUM_THREADS=2"]
    elif difference == "command":
        second["Config"]["Cmd"] = ["different-startup"]
    elif difference == "platform":
        second["Architecture"] = "arm64"
    elif difference == "custom_label":
        second["Config"]["Labels"]["experiment"] = "different-method"
    elif difference == "dependencies":
        dependencies = "numpy==2.1.4\ntorch==2.6.0\n"
    write_inspection(a, first)
    write_inspection(b, second, dependencies)
    first_spec = {"duration_seconds": 600, "evaluation_interval_seconds": 10,
                  "source_sha256": {"app/runtime.py": "unchanged"}}
    second_spec = deepcopy(first_spec)
    if difference == "code":
        second_spec["source_sha256"]["app/runtime.py"] = "modified"
    elif difference == "duration":
        second_spec["duration_seconds"] = 900
    elif difference == "interval":
        second_spec["evaluation_interval_seconds"] = 15
    networks = []
    for folder, spec in ((a, first_spec), (b, second_spec)):
        protocol = pairing_protocol(folder, spec)
        networks.append({"run_id": folder.name, "pairing_protocol": protocol,
                         "protocol_sha256": json_sha256(protocol),
                         "training_indices_sha256": "same-training",
                         "validation_indices_sha256": "same-validation"})
    if difference in {"training", "validation"}:
        networks[1][f"{difference}_indices_sha256"] = "different-data"
    with pytest.raises(ValueError, match=expected_field) as error:
        validate_paired_networks(networks, "concentrated_r_minus_none", 8)
    assert "seed=8" in str(error.value)
    assert "runs=r, none" in str(error.value)


def test_pairing_requires_exact_image_id_without_inspection(tmp_path):
    first = pairing_protocol(tmp_path, {"image_id": "sha256:r"})
    second = pairing_protocol(tmp_path, {"image_id": "sha256:none"})
    assert first != second


def test_pairing_rejects_wrong_or_incomplete_inspection(tmp_path):
    image = inspected_image("sha256:actual", "scenario-r")
    write_inspection(tmp_path, image)
    with pytest.raises(ValueError, match="image ID does not match"):
        pairing_protocol(tmp_path, {"image_id": "sha256:other"})
    del image["RootFS"]
    write_inspection(tmp_path, image)
    with pytest.raises(ValueError, match="missing image filesystem"):
        pairing_protocol(tmp_path, {})


def test_collection_pairs_different_image_ids_with_identical_runtime_contents(tmp_path):
    first = write_network(tmp_path, scenario="concentrated-r", seed=8, quality=.8)
    second = write_network(tmp_path, scenario="concentrated-none", seed=8, quality=.6)
    write_inspection(first, inspected_image("sha256:r", "scenario-r"))
    write_inspection(second, inspected_image("sha256:none", "scenario-none"))
    comparisons = summarize_collection(tmp_path)
    contrast = next(row for row in comparisons if row["contrast"] == "concentrated_r_minus_none")
    assert contrast["paired_networks"] == 1
    assert contrast["mean_difference"] == pytest.approx(.2)
