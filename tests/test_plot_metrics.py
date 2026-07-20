from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_metrics import (
    MetricsPlotError,
    available_metrics,
    build_series,
    load_experiments,
    main,
    plot_metrics,
)


def write_metrics(
    directory: Path, client_id: int, records: list[dict]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"client_{client_id}_metrics.jsonl"
    path.write_text(
        "".join(json.dumps(item) + "\n" for item in records),
        encoding="utf-8",
    )


def record(
    client_id: int,
    version: int,
    second: int,
    train: dict[str, float] | None = None,
    test: dict[str, float] | None = None,
) -> dict:
    return {
        "client_id": client_id,
        "model_version": version,
        "time": f"2026-07-18T15:39:{second:02d}+00:00",
        "train": train if train is not None else {},
        "test": test if test is not None else {},
    }


def test_plots_train_and_test_metrics_for_one_experiment(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    write_metrics(
        experiment,
        0,
        [
            record(
                0,
                2,
                20,
                {"accuracy": 0.8, "loss": 0.5},
                {"accuracy": 0.7, "loss": 0.6},
            ),
            record(
                0,
                1,
                10,
                {"accuracy": 0.6, "loss": 0.8},
                {"accuracy": 0.5, "loss": 0.9},
            ),
        ],
    )
    experiments = load_experiments([experiment])

    series, warnings = build_series(
        experiments, 0, "accuracy", "both", "model_version"
    )
    written = plot_metrics(experiments, tmp_path / "charts", "model_version", "both")

    assert warnings == []
    assert [(item.phase, item.x, item.y) for item in series] == [
        ("train", [1, 2], [0.6, 0.8]),
        ("test", [1, 2], [0.5, 0.7]),
    ]
    assert {path.name for path in written} == {
        "client_0_accuracy.png",
        "client_0_loss.png",
    }
    assert all(path.stat().st_size > 0 for path in written)


def test_matches_clients_and_metrics_across_experiments(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_metrics(
        first, 0, [record(0, 1, 10, {"accuracy": 0.5}, {"accuracy": 0.4})]
    )
    write_metrics(first, 1, [record(1, 1, 10, {"loss": 0.8}, {"loss": 0.9})])
    write_metrics(
        second, 0, [record(0, 1, 10, {"accuracy": 0.7}, {"accuracy": 0.6})]
    )

    experiments = load_experiments([first, second])
    series, warnings = build_series(
        experiments, 0, "accuracy", "both", "model_version"
    )
    client_one_series, client_one_warnings = build_series(
        experiments, 1, "loss", "both", "model_version"
    )

    assert [(item.label, item.phase) for item in series] == [
        ("first", "train"),
        ("first", "test"),
        ("second", "train"),
        ("second", "test"),
    ]
    assert warnings == []
    assert len(client_one_series) == 2
    assert any("client 1 is unavailable" in item for item in client_one_warnings)


@pytest.mark.parametrize(
    ("phase", "expected_phases"),
    [
        ("train", ["train"]),
        ("test", ["test"]),
        ("both", ["train", "test"]),
    ],
)
def test_phase_selection(
    tmp_path: Path, phase: str, expected_phases: list[str]
) -> None:
    experiment = tmp_path / "experiment"
    write_metrics(
        experiment,
        0,
        [record(0, 1, 10, {"accuracy": 0.5}, {"accuracy": 0.4})],
    )
    experiments = load_experiments([experiment])

    series, _ = build_series(
        experiments, 0, "accuracy", phase, "model_version"
    )

    assert [item.phase for item in series] == expected_phases


def test_time_axis_is_elapsed_and_sorted(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    write_metrics(
        experiment,
        0,
        [
            record(0, 2, 25, {"loss": 0.2}, {"loss": 0.3}),
            record(0, 1, 10, {"loss": 0.4}, {"loss": 0.5}),
        ],
    )
    experiments = load_experiments([experiment])

    series, _ = build_series(experiments, 0, "loss", "train", "time")

    assert series[0].x == [0.0, 15.0]
    assert series[0].y == [0.4, 0.2]


def test_missing_phase_is_warned_and_available_data_is_plotted(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_metrics(first, 0, [record(0, 1, 10, {"accuracy": 0.5}, {})])
    write_metrics(second, 0, [record(0, 1, 10, {}, {"accuracy": 0.6})])
    experiments = load_experiments([first, second])

    series, warnings = build_series(
        experiments, 0, "accuracy", "both", "model_version"
    )

    assert [(item.label, item.phase) for item in series] == [
        ("first", "train"),
        ("second", "test"),
    ]
    assert len(warnings) == 2
    assert available_metrics(experiments, 0, ("train", "test")) == {"accuracy"}


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not json\n", "malformed JSON"),
        (
            json.dumps(
                {
                    "client_id": 1,
                    "model_version": 1,
                    "time": "2026-07-18T15:39:10+00:00",
                    "train": {},
                    "test": {},
                }
            )
            + "\n",
            "does not match filename",
        ),
        (
            json.dumps(
                {
                    "client_id": 0,
                    "model_version": 1,
                    "time": "yesterday",
                    "train": {},
                    "test": {},
                }
            )
            + "\n",
            "invalid ISO-8601 timestamp",
        ),
    ],
)
def test_rejects_malformed_records(
    tmp_path: Path, contents: str, message: str
) -> None:
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "client_0_metrics.jsonl").write_text(contents, encoding="utf-8")

    with pytest.raises(MetricsPlotError, match=message):
        load_experiments([experiment])


def test_rejects_missing_and_empty_experiments(tmp_path: Path) -> None:
    with pytest.raises(MetricsPlotError, match="does not exist"):
        load_experiments([tmp_path / "missing"])

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(MetricsPlotError, match="no client_"):
        load_experiments([empty])


def test_cli_defaults_to_both_phases(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    output = tmp_path / "output"
    write_metrics(
        experiment,
        0,
        [record(0, 1, 10, {"accuracy": 0.5}, {"accuracy": 0.4})],
    )

    result = main([str(experiment), str(output)])

    assert result == 0
    assert (output / "client_0_accuracy.png").is_file()
