import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.plot_metrics import (
    MetricsPlotError, build_combined_series, load_experiments, plot_final_metrics,
)


def write_client(path: Path, client: int, observations, run="run", start=True):
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for version, second, value in observations:
        row = {"client_id": client, "model_version": version,
               "time": f"2026-10-14T00:00:{second:02d}+00:00", "run_id": run,
               "test": {"accuracy": value, "loss": 1 - value}}
        if start:
            row["started_at"] = "2026-10-14T00:00:00+00:00"
        rows.append(json.dumps(row))
    (path / f"client_{client}_metrics.jsonl").write_text("\n".join(rows) + "\n")


def test_round_population_std_and_variable_participants(tmp_path):
    write_client(tmp_path, 0, [(1, 10, 0), (3, 30, 1)])
    write_client(tmp_path, 1, [(1, 20, 1), (2, 40, 0)])
    write_client(tmp_path, 2, [(1, 25, 0)])
    series = build_combined_series(load_experiments([tmp_path])[0], "accuracy")
    assert series.rounds.tolist() == [1, 2, 3]
    assert series.participants.tolist() == [3, 1, 1]
    np.testing.assert_allclose(series.mean, [1 / 3, 0, 1])
    np.testing.assert_allclose(series.std, [np.sqrt(2 / 9), 0, 0])


def test_explicit_rounds_choose_highest_version_then_last_line(tmp_path):
    write_client(tmp_path, 0, [(3, 10, 0.4), (1, 15, 0.1),
                              (3, 18, 0.8), (4, 20, 1)], start=False)
    path = tmp_path / "client_0_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row, number in zip(rows, [1, 1, 1, 2]):
        row["round"] = number
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    series = build_combined_series(load_experiments([tmp_path])[0], "accuracy")
    assert series.rounds.tolist() == [1, 2]
    assert series.mean.tolist() == [0.8, 1]


def test_round_fallback_does_not_require_started_at(tmp_path, capsys):
    write_client(tmp_path, 0, [(3, 10, 0.4), (3, 18, 0.8), (4, 20, 1)],
                 start=False)
    series = build_combined_series(load_experiments([tmp_path])[0], "accuracy")
    assert series.rounds.tolist() == [3, 4]
    assert series.mean.tolist() == [0.8, 1]
    assert "fallback" not in capsys.readouterr().err


def test_rejects_mixed_runs_and_invalid_run_id(tmp_path):
    write_client(tmp_path, 0, [(1, 10, 0.5)])
    write_client(tmp_path, 1, [(1, 10, 0.5)], run="another")
    with pytest.raises(MetricsPlotError, match="multiple run IDs"):
        load_experiments([tmp_path])
    write_client(tmp_path, 1, [(1, 10, 0.5)], run=[])
    with pytest.raises(MetricsPlotError, match="run_id"):
        load_experiments([tmp_path])


def test_combined_outputs_confusions_and_client_legend(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_client(source, 0, [(1, 10, 0.5), (2, 30, 0.8)])
    write_client(source, 1, [(1, 20, 0.4), (2, 40, 0.7)])
    (source / "client_0_confusion.json").write_text(json.dumps({
        "client_id": 0, "run_id": "run", "model_version": 2,
        "classification_type": "binary", "labels": ["0", "1"],
        "sample_count": 4, "counts": [[2, 1], [0, 1]],
    }))
    from matplotlib.axes import Axes
    from matplotlib.ticker import MaxNLocator
    original = Axes.legend
    legends = []
    def legend(axis, *args, **kwargs):
        assert axis.get_xlabel() == "Round"
        assert isinstance(axis.xaxis.get_major_locator(), MaxNLocator)
        for collection in axis.collections[1:]:
            assert collection.get_offsets()[:, 0].tolist() == [1, 2]
        legends.append(axis.get_legend_handles_labels()[1])
        return original(axis, *args, **kwargs)
    monkeypatch.setattr(Axes, "legend", legend)
    charts = tmp_path / "charts"
    charts.mkdir()
    old_output = charts / "combined_test_accuracy_time.csv"
    old_output.write_text("previous time output")
    written = plot_final_metrics(load_experiments([source]), tmp_path / "charts", True)
    assert old_output.read_text() == "previous time output"
    assert len(written) == 5
    assert all(p.stat().st_size > 0 for p in written)
    assert all("Client 0" in labels and "Client 1" in labels for labels in legends)
    with (tmp_path / "charts/combined_test_accuracy_round.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["round", "mean", "std", "participants"]
    assert [int(r["round"]) for r in rows] == [1, 2]
    assert [int(r["participants"]) for r in rows] == [2, 2]


@pytest.mark.parametrize("round_number", [None, True, 1.5, "1"])
def test_rejects_noninteger_explicit_round(tmp_path, round_number):
    write_client(tmp_path, 0, [(1, 10, 0.5)])
    path = tmp_path / "client_0_metrics.jsonl"
    record = json.loads(path.read_text())
    record["round"] = round_number
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(MetricsPlotError, match="'round' must be an integer"):
        load_experiments([tmp_path])


def test_multiple_experiments_have_separate_output_directories(tmp_path):
    sources = [tmp_path / "a/run", tmp_path / "b/run"]
    for source in sources:
        write_client(source, 0, [(1, 10, 0.5)])
    written = plot_final_metrics(load_experiments(sources), tmp_path / "plots")
    assert len({p.parent for p in written}) == 2
    assert len(written) == 8


def test_multilabel_panel_and_stale_artifact_rejection(tmp_path):
    write_client(tmp_path, 0, [(1, 10, 0.5)])
    raw = {"client_id": 0, "run_id": "run", "model_version": 1,
           "classification_type": "multilabel", "labels": [str(i) for i in range(20)],
           "sample_count": 4, "counts": [[[2, 1], [0, 1]]] * 20}
    path = tmp_path / "client_0_confusion.json"
    path.write_text(json.dumps(raw))
    experiment = load_experiments([tmp_path])[0]
    from scripts.plot_metrics import plot_confusions
    assert len(plot_confusions(experiment, tmp_path / "charts")) == 1
    raw["run_id"] = "stale"
    path.write_text(json.dumps(raw))
    with pytest.raises(MetricsPlotError, match="run ID"):
        plot_confusions(experiment, tmp_path / "charts")


