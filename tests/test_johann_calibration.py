import json

import pytest

from experiments.johann import analysis
from experiments.johann.calibration import calibration_specs, select_calibration
from scripts import run_johann_experiments as runner


def test_calibration_is_validation_only_and_deterministic():
    for dataset, n in (("mnist", 6750), ("cifar10", 5625)):
        candidates = calibration_specs(dataset)
        assert len(candidates) == 11
        assert {s["seed"] for _, s in candidates} == {7}
        assert {s["evaluation_split"] for _, s in candidates} == {"validation"}
        assert {s["scenario"] for _, s in candidates} == {"concentrated-r"}
        assert [s["method_parameters"]["merge_limit_examples"] for _, s in candidates if s["method"] == "M2"] == [n, 5*n, 10*n]


def test_selection_requires_full_grid_and_applies_predeclared_ties(tmp_path, monkeypatch):
    summaries = {}
    for trial, spec in calibration_specs("mnist"):
        folder = tmp_path / trial
        folder.mkdir()
        spec.update(calibration_id="tune", calibration_trial=trial)
        (folder / "resolved_experiment.json").write_text(json.dumps(spec))
        (folder / "outcome.json").write_text(json.dumps({"state": "completed"}))
        for client in range(8):
            (folder / f"client_{client}_class_distribution.json").write_text(json.dumps({"counts": [675] * 10}))
        summaries[folder] = {"outcomes": [{"selection": "online", "post_return_auc": .8}],
                             "pairing_protocol": {"method": spec["method"], "method_parameters": spec["method_parameters"],
                                                  "source_sha256": "same", "duration_seconds": 600},
                             "training_indices_sha256": "same", "validation_indices_sha256": "same"}
    monkeypatch.setattr(analysis, "summarize_run", lambda folder: summaries[folder])
    selected = select_calibration(tmp_path, "tune")["datasets"]["mnist"]
    assert selected["methods"]["M2"] == {"merge_limit_examples": 67500}
    assert selected["methods"]["M3"] == {"rejoin_batches": 16}
    assert selected["methods"]["M4"]["rejoin_batches"] == 16
    assert selected["methods"]["M4"]["distillation_weight"] == .1
    assert selected["protocol_frozen"] is False
    missing = tmp_path / "M3-k16" / "outcome.json"
    missing.write_text(json.dumps({"state": "failed"}))
    with pytest.raises(ValueError, match="incomplete"):
        select_calibration(tmp_path, "tune")


def test_runner_validates_all_methods_before_preparing_any_artifact(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "prepare", lambda *args: calls.append(args))
    assert runner.main(["pilot", "--methods", "M0", "M4", "--rejoin-batches", "0", "--results-root", str(tmp_path)]) == 2
    assert calls == []


def test_render_calibration_does_not_execute_or_use_official_test(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "execute", lambda *args: pytest.fail("must not launch Docker"))
    assert runner.main(["calibrate", "--dataset", "mnist", "--run-id", "tune", "--render-only",
                        "--results-root", str(tmp_path)]) == 0
    specs = [json.loads(path.read_text()) for path in tmp_path.rglob("resolved_experiment.json")]
    assert len(specs) == 11
    assert all(s["calibration_id"] == "tune" and s["evaluation_split"] == "validation" for s in specs)
