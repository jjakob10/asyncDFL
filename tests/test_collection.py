import json
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.collection import CollectionSource, read_json, resolve_plan, validate_schedule
from app.config import Settings
from app.federation import clone_state_dict, serialize_state_dict
from app.runtime import ClientRuntime
from app.training import DataBundle
from scripts.plot_metrics import load_experiments, plot_metrics


def settings(tmp_path, **kwargs):
    return Settings(client_id=0, client_name="client-0", client_count=2,
                    head_client_ids=(0,), rounds=5, collection_mode="sliding_window",
                    collection_window_rounds=3, collection_schedule_path=str(tmp_path / "schedule.json"),
                    results_dir=str(tmp_path), **kwargs)


def stock():
    labels = np.array([0, 1] * 6)
    dataset = TensorDataset(torch.arange(12, dtype=torch.float32).reshape(-1, 1),
                            torch.tensor(labels))
    return CollectionSource(dataset, labels, tuple(f"sample:{i}" for i in range(12)), (0, 1))


def agenda():
    return {"version": 1, "selection": "by_class", "clients": {
        "0": [{"rounds": [1, 2], "counts_by_class": {"0": 1}},
              {"rounds": [3, 3], "counts_by_class": {"1": 2}}], "1": []}}


@pytest.mark.parametrize("mode,window,expected", [
    ("accumulative", 3, [1, 2, 4, 4, 4]),
    ("sliding_window", 1, [1, 1, 2, 0, 0]),
    ("sliding_window", 3, [1, 2, 4, 3, 2]),
    ("sliding_window", 20, [1, 2, 4, 4, 4]),
])
def test_retention_exact_examples_and_no_future_data(tmp_path, mode, window, expected):
    config = replace(settings(tmp_path), collection_mode=mode, collection_window_rounds=window)
    plan = resolve_plan(agenda(), stock(), config)
    released = [set(batch.tolist()) for batch in plan.releases]
    assert sum(map(len, released)) == len(set.union(*released))
    for r, count in enumerate(expected, 1):
        loader, metrics, audit = plan.round_data(r, config)
        start = 0 if mode == "accumulative" else max(0, r - window)
        wanted = set.union(*released[start:r])
        actual = set(loader.dataset.indices) if loader else set()
        assert actual == wanted
        assert metrics["active"]["count"] == count
        assert metrics["stock_remaining"] == 12 - sum(len(part) for part in released[:r])
        assert metrics["trained"] == bool(count)
        assert set(audit["released_ids"]) == {plan.source.sample_ids[i] for i in released[r-1]}


def test_plan_is_deterministic_and_independent_of_retention(tmp_path):
    config = settings(tmp_path)
    first = resolve_plan(agenda(), stock(), config)
    second = resolve_plan(agenda(), stock(), replace(config, collection_mode="accumulative"))
    assert first.plan_hash == second.plan_hash
    assert first.resolved == second.resolved
    changed = resolve_plan(agenda(), stock(), replace(config, seed=99))
    assert changed.resolved["releases"] != first.resolved["releases"]


@pytest.mark.parametrize("rule,match", [
    ({"rounds": [0, 2], "counts_by_class": {"0": 1}}, "interval"),
    ({"rounds": [1, 6], "counts_by_class": {"0": 1}}, "interval"),
    ({"rounds": [True, 2], "counts_by_class": {"0": 1}}, "interval"),
    ({"rounds": [1, 2], "counts_by_class": {"3": 1}}, "unknown class"),
    ({"rounds": [1, 2], "counts_by_class": {"0": -1}}, "nonnegative"),
    ({"rounds": [1, 2], "counts_by_class": {"0": 1.5}}, "nonnegative"),
    ({"rounds": [1, 2], "counts_by_class": {"0": True}}, "nonnegative"),
    ({"rounds": [1, 2], "count": 1}, "counts_by_class"),
])
def test_invalid_agenda_rules(tmp_path, rule, match):
    schedule = agenda()
    schedule["clients"]["0"] = [rule]
    with pytest.raises(ValueError, match=match):
        resolve_plan(schedule, stock(), settings(tmp_path))


def test_rejects_overlap_missing_clients_and_duplicate_json(tmp_path):
    schedule = agenda()
    schedule["clients"]["0"].append({"rounds": [2, 3], "counts_by_class": {}})
    with pytest.raises(ValueError, match="overlapping"):
        resolve_plan(schedule, stock(), settings(tmp_path))
    del schedule["clients"]["1"]
    with pytest.raises(ValueError, match="every client"):
        validate_schedule(schedule, 5, 2, (0, 1))
    path = tmp_path / "duplicate.json"
    path.write_text('{"version": 1, "version": 1}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        read_json(path)


def test_insufficient_stock_rejected_before_any_round(tmp_path):
    schedule = agenda()
    schedule["clients"]["0"] = [{"rounds": [1, 3], "counts_by_class": {"1": 3}}]
    with pytest.raises(ValueError, match="client 0 round 3 class 1: requested 3, available 0"):
        resolve_plan(schedule, stock(), settings(tmp_path))


def test_chronological_selection_never_skips_or_repeats(tmp_path):
    source = stock()
    source.chronological = True
    schedule = {"version": 1, "selection": "chronological", "clients": {
        "0": [{"rounds": [1, 3], "count": 2}], "1": []}}
    plan = resolve_plan(schedule, source, settings(tmp_path))
    assert [part.tolist() for part in plan.releases] == [[0, 1], [2, 3], [4, 5], [], []]
    with pytest.raises(ValueError, match="cannot be reordered"):
        resolve_plan(agenda(), source, settings(tmp_path))
    source.chronological = False
    with pytest.raises(ValueError, match="chronologically ordered"):
        resolve_plan(schedule, source, settings(tmp_path))
    source.chronological = True
    schedule["clients"]["0"] = [{"rounds": [1, 5], "count": 3}]
    with pytest.raises(ValueError, match="client 0 round 5: requested 3, available 0"):
        resolve_plan(schedule, source, settings(tmp_path))


class RecordingTrainer:
    def __init__(self, data):
        self.data = data
        self.seen = []
        self.evaluate_loaders = []
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.block = False

    def load_data(self, settings):
        return self.data

    def create_model(self):
        return nn.Linear(1, 2)

    def train_local(self, base, loader, settings):
        self.seen.append(list(loader.dataset.indices))
        if self.block:
            self.entered.set()
            assert self.proceed.wait(5)
        return clone_state_dict(base), {"loss": 1.0, "accuracy": .5}

    def evaluate(self, state, loader):
        self.evaluate_loaders.append(loader)
        return {"loss": .8, "accuracy": .5}


def runtime_fixture(tmp_path, **kwargs):
    schedule = agenda()
    schedule["clients"]["0"] = [{"rounds": [2, 2], "counts_by_class": {"0": 2}}]
    (tmp_path / "schedule.json").write_text(json.dumps(schedule))
    source = stock()
    test = DataLoader(source.dataset, batch_size=2)
    data = DataBundle(None, test, len(source.dataset), collection_source=source)
    trainer = RecordingTrainer(data)
    config = replace(settings(tmp_path), collection_window_rounds=1, **kwargs)
    return ClientRuntime(config, trainer=trainer), trainer


def test_runtime_collection_empty_rounds_fixed_test_and_reconstructable_audit(tmp_path):
    runtime, trainer = runtime_fixture(tmp_path)
    original = clone_state_dict(runtime._model)
    assert runtime.model_snapshot()[1]["X-Sample-Count"] == "0"
    assert runtime.start("collection") == "started"
    runtime.wait(5)
    assert runtime.status()["state"] == "completed"
    records = [json.loads(line) for line in (tmp_path / "client_0_metrics.jsonl").read_text().splitlines()]
    assert [r["sample_count"] for r in records] == [0, 2, 0, 0, 0]
    assert [r["participants"] for r in records] == [0, 1, 0, 0, 0]
    assert len(trainer.seen) == 1
    assert all(loader is runtime.data.test_loader for loader in trainer.evaluate_loaders)
    assert all(torch.equal(value, original[key]) for key, value in runtime._model.items())
    audits = [json.loads(line) for line in (tmp_path / "client_0_collection.jsonl").read_text().splitlines()]
    assert len(audits) == 5
    assert audits[1]["released_ids"] == audits[2]["expired_ids"]
    plan = read_json(tmp_path / "client_0_collection_plan.json")
    assert plan["releases"] == [a["released_ids"] for a in audits]
    plots = plot_metrics(load_experiments([tmp_path]), tmp_path / "charts", "round", "both")
    assert plots and all(path.is_file() for path in plots)


def test_published_sample_count_does_not_change_while_training(tmp_path):
    runtime, trainer = runtime_fixture(tmp_path)
    trainer.block = True
    runtime.start("atomic")
    assert trainer.entered.wait(5)
    try:
        _, headers = runtime.model_snapshot()
        assert headers["X-Model-Version"] == "1"
        assert headers["X-Sample-Count"] == "0"
        assert runtime.status()["sample_count"] == 0
    finally:
        trainer.proceed.set()
        runtime.wait(5)
    assert runtime.status()["state"] == "completed"


def test_zero_weight_neighbor_is_excluded_and_failure_recorded(tmp_path, monkeypatch):
    runtime, _ = runtime_fixture(tmp_path, neighbors=("http://zero", "http://missing"), request_retries=1)
    payload = serialize_state_dict(runtime._model)
    def response(method, url, **kwargs):
        if "missing" in url:
            raise httpx.ConnectError("unavailable")
        return httpx.Response(200, content=payload, headers={
            "X-Client-ID": "1", "X-Sample-Count": "0", "X-Model-Version": "0",
            "X-Completed-Round": "0"})
    monkeypatch.setattr(runtime, "_request_with_retries", response)
    monkeypatch.setattr(runtime, "_forward_start", lambda: None)
    runtime.start("zero")
    runtime.wait(5)
    assert runtime.status()["state"] == "completed"
    assert runtime.status()["latest_metrics"]["participants"] == 0
    assert runtime.status()["recent_errors"]


def test_enabled_settings_validation_and_preflight(tmp_path, monkeypatch):
    config = settings(tmp_path)
    for invalid in (replace(config, collection_window_rounds=True),
                    replace(config, collection_window_rounds=0),
                    replace(config, collection_schedule_path=""),
                    replace(config, run_duration_seconds=2),
                    replace(config, collection_mode="unknown")):
        with pytest.raises(ValueError):
            invalid.validate()
    monkeypatch.setenv("CLIENT_COUNT", "2")
    monkeypatch.setenv("HEAD_CLIENT_IDS", "0")
    monkeypatch.setenv("COLLECTION_MODE", "sliding_window")
    monkeypatch.setenv("COLLECTION_SCHEDULE_PATH", str(tmp_path / "schedule.json"))
    monkeypatch.setenv("COLLECTION_WINDOW_ROUNDS", "4")
    assert Settings.from_env().collection_window_rounds == 4
    runtime, trainer = runtime_fixture(tmp_path)
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({"clients": {"0": "wrong"}}))
    with pytest.raises(ValueError, match="differs from validated"):
        ClientRuntime(replace(runtime.settings, collection_preflight_path=str(path)), trainer=trainer)
    path.write_text(json.dumps({"clients": {"0": runtime.collection.plan_hash}}))
    ClientRuntime(replace(runtime.settings, collection_preflight_path=str(path)), trainer=trainer)


def test_positive_neighbor_updates_client_without_active_examples(tmp_path, monkeypatch):
    runtime, _ = runtime_fixture(tmp_path, neighbors=("http://peer",))
    remote = {key: torch.full_like(value, 3) for key, value in runtime._model.items()}
    def fetched(*args):
        return clone_state_dict(remote), 7, {"client_id": 1, "sample_count": 7,
                                            "aggregation_weight": 7, "model_version": 2}
    monkeypatch.setattr(runtime, "_fetch_neighbor", fetched)
    monkeypatch.setattr(runtime, "_forward_start", lambda: None)
    runtime.start("positive-peer")
    runtime.wait(5)
    assert runtime.status()["state"] == "completed"
    assert runtime.status()["sample_count"] == 0
    assert all(torch.equal(value, remote[key]) for key, value in runtime._model.items())
    final = runtime.status()["latest_metrics"]
    assert final["effective_aggregation_weights"] == {"0": 0, "1": 1}
