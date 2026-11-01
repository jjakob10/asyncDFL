from __future__ import annotations

import json
from pathlib import Path

from scripts.experiment_tools.reporting import summarize_collection, summarize_run


def test_summary_excludes_disconnected_controls_from_average(
    tmp_path: Path,
) -> None:
    records = [
        {
            "time": "2026-07-20T00:00:00+00:00",
            "run_id": "run",
            "experiment_id": "experiment",
            "topology_name": "ring",
            "client_id": 0,
            "round": 1,
            "model_version": 1,
            "paper_role": "E",
            "has_local_data": False,
            "is_connected": True,
            "sample_count": 0,
            "test": {"accuracy": 0.8, "loss": 0.4},
        },
        {
            "time": "2026-07-20T00:00:00+00:00",
            "run_id": "run",
            "experiment_id": "experiment",
            "topology_name": "ring",
            "client_id": 8,
            "round": 1,
            "model_version": 1,
            "paper_role": "D",
            "has_local_data": True,
            "is_connected": False,
            "sample_count": 100,
            "test": {"accuracy": 0.2, "loss": 1.2},
        },
    ]
    for record in records:
        path = tmp_path / f"client_{record['client_id']}_metrics.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    csv_path, markdown_path, charts = summarize_run(tmp_path)

    assert csv_path.is_file()
    assert "0.800000" in markdown_path.read_text(encoding="utf-8")
    assert len(charts) == 6
    assert (tmp_path / "combined_test_accuracy_round.csv").is_file()


def test_collection_summary_groups_experiment_runs(tmp_path: Path) -> None:
    for topology, accuracy in (
        ("ring", 0.7),
        ("star", 0.9),
    ):
        run_directory = tmp_path / topology / "shared-run"
        run_directory.mkdir(parents=True)
        record = {
            "time": "2026-07-20T00:00:00+00:00",
            "model_version": 1,
            "run_id": "shared-run",
            "experiment_id": f"mnist-{topology}",
            "topology_name": topology,
            "client_id": 0,
            "round": 1,
            "paper_role": "R",
            "has_local_data": True,
            "is_connected": True,
            "sample_count": 100,
            "test": {"accuracy": accuracy, "loss": 0.4},
        }
        (run_directory / "client_0_metrics.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        (run_directory / "resolved_experiment.json").write_text(
            json.dumps(
                {
                    "dataset": "mnist",
                    "rounds": 24,
                    "local_epochs": 32,
                }
            ),
            encoding="utf-8",
        )

    written = summarize_collection(tmp_path)

    assert len(written) == 3
    comparison = (tmp_path / "comparison.csv").read_text(encoding="utf-8")
    assert "ring" in comparison
    assert "star" in comparison
