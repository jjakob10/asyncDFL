"""Predeclared validation-only search and deterministic parameter selection."""
from __future__ import annotations

import json
import math
from pathlib import Path

from experiments.johann.catalog import load_spec


def calibration_specs(dataset: str):
    trials = [(f"M2-l{factor}n", "M2", {"merge_limit_multiplier": factor}) for factor in (1, 5, 10)]
    trials += [(f"M3-k{k}", "M3", {"rejoin_batches": k}) for k in (16, 64)]
    trials += [(f"M4-k{k}-w{str(weight).replace('.', 'p')}", "M4",
                {"rejoin_batches": k, "distillation_weight": weight, "distillation_temperature": 2})
               for k in (16, 64) for weight in (.1, .5, 1.0)]
    return [(name, load_spec(dataset, "concentrated-r", seed=7, mode="pilot", method=method,
                             method_parameters=parameters)) for name, method, parameters in trials]


def select_calibration(root: Path, calibration_id: str) -> dict:
    """Require all planned candidates; never select only successful/favorable runs."""
    from experiments.johann.analysis import json_sha256, summarize_run

    datasets = {}
    for path in sorted(Path(root).rglob("resolved_experiment.json")):
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec.get("calibration_id") != calibration_id or "calibration_trial" not in spec:
            continue
        if spec["mode"] != "pilot" or spec["evaluation_split"] != "validation" or spec["seed"] != 7 or spec["scenario"] != "concentrated-r":
            raise ValueError("calibration requires validation/concentrated-r/seed 7")
        outcome = path.parent / "outcome.json"
        if not outcome.exists() or json.loads(outcome.read_text())["state"] != "completed":
            raise ValueError(f"incomplete calibration candidate: {path.parent}")
        # Recompute rather than trusting an outdated hand-edited summary.
        summary = summarize_run(path.parent)
        score = next(r["post_return_auc"] for r in summary["outcomes"] if r["selection"] == "online")
        if score is None or not math.isfinite(score):
            raise ValueError(f"calibration candidate has incomplete outcome: {path.parent}")
        protocol = dict(summary["pairing_protocol"])
        protocol.pop("method", None)
        protocol.pop("method_parameters", None)
        protocol.update(training_indices_sha256=summary["training_indices_sha256"],
                        validation_indices_sha256=summary["validation_indices_sha256"])
        distributions = [json.loads((path.parent / f"client_{client}_class_distribution.json").read_text()) for client in range(8)]
        measured_n = max(sum(d["counts"]) for d in distributions)
        if measured_n != spec["max_local_training_samples"]:
            raise ValueError("measured local sample count disagrees with calibration N")
        datasets.setdefault(spec["dataset"], []).append({
            "trial": spec["calibration_trial"], "method": spec["method"],
            "parameters": spec["method_parameters"], "score": score, "directory": str(path.parent),
            "common_protocol": json_sha256(protocol)})
    if not datasets:
        raise ValueError("no matching calibration candidates")
    selections = {}
    for dataset, candidates in datasets.items():
        expected = {name: spec for name, spec in calibration_specs(dataset)}
        if len(candidates) != len(expected) or {r["trial"] for r in candidates} != set(expected):
            raise ValueError(f"{dataset}: require exactly all eleven calibration candidates")
        for row in candidates:
            if row["parameters"] != expected[row["trial"]]["method_parameters"] or row["method"] != expected[row["trial"]]["method"]:
                raise ValueError("candidate does not match the declared search grid")
        if len({row["common_protocol"] for row in candidates}) != 1:
            raise ValueError("calibration candidates disagree about data, code or runtime protocol")
        m2 = min((r for r in candidates if r["method"] == "M2"), key=lambda r: (-r["score"], -r["parameters"]["merge_limit_examples"]))
        m3 = min((r for r in candidates if r["method"] == "M3"), key=lambda r: (-r["score"], r["parameters"]["rejoin_batches"]))
        m4 = min((r for r in candidates if r["method"] == "M4" and
                  r["parameters"]["rejoin_batches"] == m3["parameters"]["rejoin_batches"]),
                 key=lambda r: (-r["score"], r["parameters"]["distillation_weight"]))
        selections[dataset] = {"methods": {"M0": {}, "M1": {}, **{r["method"]: r["parameters"] for r in (m2, m3, m4)}},
                               "chosen_runs": [r["directory"] for r in (m2, m3, m4)], "candidates": candidates,
                               "confirmation_seeds": [8, 9], "protocol_frozen": False}
    return {"calibration_id": calibration_id, "datasets": selections}
