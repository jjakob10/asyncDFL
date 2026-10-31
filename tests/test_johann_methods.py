import json
import threading
from dataclasses import replace

import httpx
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.aggregation import AGE_UNIT, ModelInput, aggregate_models, select_rejoin_neighbor
from app.config import Settings
from app.federation import clone_state_dict, deserialize_state_dict, serialize_state_dict
from app.run_control import RunControl
from app.timed_runtime import TimedClientRuntime
from app.training import DataBundle
from experiments.johann import trainer as shared
from experiments.johann.catalog import compose_for, load_spec
from scripts.plot_metrics import _aggregation_weights


def item(client, value, age, count=1):
    return ModelInput({"weight": torch.tensor([float(value)])}, client, age, count)


def test_age_merging_preserves_pairwise_order_and_does_not_sum_ages():
    result = aggregate_models("M1", item(0, 0, 10), [item(2, 20, 10), item(1, 10, 10)])
    assert result.state["weight"].item() == 12.5
    assert result.age == 10
    assert result.weights == {0: .25, 1: .25, 2: .5}
    assert [t["client_id"] for t in result.trace] == [1, 2]
    zero = aggregate_models("M1", item(0, 7, 0), [item(1, 9, 0)])
    assert zero.state["weight"].item() == 7
    assert zero.weights == {0: 1, 1: 0}
    result = aggregate_models("M1", item(0, 7, 0), [item(1, 9, 4)])
    assert result.state["weight"].item() == 9 and result.age == 4


@pytest.mark.parametrize("difference,action", [(9, "age_weighted"), (10, "age_weighted"), (11, "replace")])
def test_limited_merging_uses_strict_threshold(difference, action):
    result = aggregate_models("M2", item(0, 0, 10), [item(1, 20, 10 + difference)], 10)
    assert result.trace[0]["action"] == action
    assert result.age == 10 + difference
    reverse = aggregate_models("M2", item(0, 20, 10 + difference), [item(1, 0, 10)], 10)
    assert reverse.state["weight"].item() == pytest.approx(result.state["weight"].item())
    if difference > 10:
        assert reverse.weights == {0: 1, 1: 0}


def test_m0_uses_samples_and_unknown_legacy_age_is_not_invented():
    result = aggregate_models("M0", item(0, 0, 2, 3), [item(1, 20, 10, 1)])
    assert result.state["weight"].item() == 5 and result.age == 10
    assert aggregate_models("M0", item(0, 0, 2), [item(1, 20, None)]).age is None
    with pytest.raises(ValueError, match="complete"):
        aggregate_models("M1", item(0, 0, 2), [item(1, 20, None)])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_models("M2", item(0, 0, 2), [item(0, 20, 10)])


def test_neighbor_selection_uses_age_then_identifier_even_below_own_age():
    assert select_rejoin_neighbor([]) is None
    assert select_rejoin_neighbor([item(2, 1, 8), item(1, 2, 8), item(3, 3, 7)]).client_id == 1


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TinyTrainer:
    @staticmethod
    def create_model():
        return nn.Linear(1, 10)

    @staticmethod
    def train_local_controlled(base, loader, settings, control):
        state = clone_state_dict(base)
        for inputs, _ in loader:
            control.checkpoint()
            for tensor in state.values():
                tensor.add_(.01)
            control.batch_finished(len(inputs), 1)
        control.checkpoint()
        return state, {"loss": 1.0}

    @staticmethod
    def train_reintegration(base, previous, loader, settings, control):
        return shared.train_reintegration(base, previous, loader, settings, control, "mnist")

    @staticmethod
    def evaluate_detailed(state, loader):
        return shared.metrics_from_counts(np.eye(10, dtype=int))


def make_runtime(tmp_path, method="M4", offline=True, batches=2):
    settings = Settings(client_id=0, client_name="client-0", method=method,
                        run_duration_seconds=20, evaluation_interval_seconds=1,
                        offline_client_ids=(0,) if offline else (), auto_start_head=False,
                        results_dir=str(tmp_path), rejoin_batches=batches, batch_size=2,
                        neighbors=("http://peer",))
    loader = DataLoader(TensorDataset(torch.arange(5.).reshape(-1, 1), torch.arange(5)), batch_size=2)
    runtime = TimedClientRuntime(settings, DataBundle(loader, loader, 5), TinyTrainer())
    clock = Clock()
    runtime.control = RunControl(0, 20, offline, clock=clock, sleep=clock.sleep, emit=runtime._event)
    runtime.control.require_reintegration = method in {"M3", "M4"}
    original_batch = runtime.control.batch_finished

    def batch(count, seconds):
        original_batch(count, seconds)
        clock.now += 1

    runtime.control.batch_finished = batch
    runtime._run_id = "unit"
    return runtime, clock


def peers_for(runtime):
    state = clone_state_dict(runtime._model)
    metadata = {"client_id": 1, "training_age": 1, "sample_count": 5,
                "model_version": 1, "aggregation_weight": 5}
    return [ModelInput(state, 1, 1, 5, metadata)]


@pytest.mark.parametrize("method", ["M3", "M4"])
def test_rejoin_discards_suspended_work_and_publishes_once(tmp_path, monkeypatch, method):
    monkeypatch.setattr(shared, "base_trainer", lambda _: TinyTrainer)
    runtime, clock = make_runtime(tmp_path, method)
    peers = peers_for(runtime)
    runtime._neighbors = lambda _: peers
    runtime._execute()
    assert runtime.status()["state"] == "completed", runtime.status()["recent_errors"]
    events = [json.loads(line) for line in runtime.events.path.read_text().splitlines()]
    assert sum(e["event"] == "reintegration_completed" for e in events) == 1
    discarded = next(e for e in events if e["event"] == "suspended_cycle_discarded")
    assert discarded["discarded_steps"] == 3 and discarded["discarded_examples"] == 5
    reset = next(e for e in events if e["event"] == "reinitialization")
    assert reset["selected_age"] < reset["previous_age"]
    records = [json.loads(line) for line in (tmp_path / "client_0_metrics.jsonl").read_text().splitlines()]
    adapted = [r for r in records if r.get("reintegration")]
    assert len(adapted) == 1 and adapted[0]["training_age"] == 1 + 4
    assert adapted[0]["effective_aggregation_weights"] is None
    assert runtime.control.reintegration_steps == 2
    assert runtime.control.teacher_inferences == (2 if method == "M4" else 0)
    assert (tmp_path / "client_0_after_reinitialization.pt").exists()
    assert (tmp_path / "client_0_after_reintegration.pt").exists()


@pytest.mark.parametrize("no_peers", [False, True])
def test_budget_expires_without_publishing_partial_reintegration(tmp_path, monkeypatch, no_peers):
    monkeypatch.setattr(shared, "base_trainer", lambda _: TinyTrainer)
    runtime, _ = make_runtime(tmp_path, batches=100)
    peers = [] if no_peers else peers_for(runtime)
    runtime._neighbors = lambda _: peers
    runtime._execute()
    assert runtime.status()["state"] == "completed"
    assert runtime._io["reintegration_completed"] is False
    assert "reintegration_incomplete" in runtime.events.path.read_text()
    assert runtime._model_version == 1
    assert runtime._training_age == 5
    assert runtime._io["discarded_optimizer_steps"] >= 3
    if no_peers:
        assert runtime._io["reintegration_wait_seconds"] >= 8
        assert runtime.control.reintegration_steps == 0


def test_no_absence_matches_m0_without_extra_teacher_or_steps(tmp_path):
    states = []
    ages = []
    for method in ("M0", "M3", "M4"):
        runtime, _ = make_runtime(tmp_path / method, method, offline=False)
        peers = peers_for(runtime)
        runtime._neighbors = lambda _, peers=peers: peers
        runtime._execute()
        assert runtime.status()["state"] == "completed"
        assert runtime.control.reintegration_steps == runtime.control.teacher_inferences == 0
        states.append(runtime._model)
        ages.append(runtime._training_age)
    assert ages == [ages[0]] * 3
    assert all(torch.equal(states[0][k], s[k]) for s in states for k in s)


def test_zero_distillation_matches_m3_and_teacher_is_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(shared, "base_trainer", lambda _: TinyTrainer)
    loader = DataLoader(TensorDataset(torch.arange(5.).reshape(-1, 1), torch.arange(5)), batch_size=2)
    initial = TinyTrainer.create_model().state_dict()
    previous = {k: t + 1 for k, t in initial.items()}
    original = clone_state_dict(previous)
    outputs = []
    seen_teachers = []
    deepcopy = shared.deepcopy

    def capture(model):
        teacher = deepcopy(model)
        seen_teachers.append(teacher)
        return teacher

    monkeypatch.setattr(shared, "deepcopy", capture)
    for method, weight in (("M3", .5), ("M4", 0), ("M4", .5)):
        runtime, _ = make_runtime(tmp_path / f"{method}-{weight}", method, offline=False, batches=4)
        settings = replace(runtime.settings, distillation_weight=weight)
        state, _ = shared.train_reintegration(initial, previous, loader, settings, runtime.control, "mnist")
        outputs.append(state)
        assert runtime.control.reintegration_examples == 7  # 2+2+1+2, including epoch wrap
        assert runtime.control.reintegration_steps == 4
    assert all(torch.equal(outputs[0][k], outputs[1][k]) for k in initial)
    assert any(not torch.equal(outputs[0][k], outputs[2][k]) for k in initial)
    assert len(seen_teachers) == 1
    teacher = seen_teachers[0]
    assert not teacher.training
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert all(torch.equal(teacher.state_dict()[k], original[k]) for k in original)


def test_atomic_snapshot_keeps_age_and_parameters_together(tmp_path):
    runtime, clock = make_runtime(tmp_path, "M1", offline=False)
    clock.now = 1
    runtime._model = {k: torch.zeros_like(t) for k, t in runtime._model.items()}
    failures = []

    def writer():
        for i in range(1, 30):
            state = {k: torch.full_like(t, i) for k, t in runtime._model.items()}
            runtime._publish(state, i, {}, [], 0, {0: 1.0}, [])

    thread = threading.Thread(target=writer)
    thread.start()
    for _ in range(40):
        payload, headers = runtime.model_snapshot()
        age = int(headers["X-Training-Age"])
        state = deserialize_state_dict(payload)
        if int(headers["X-Model-Version"]) != age or any(not (t == age).all() for t in state.values()):
            failures.append(headers)
        assert headers["X-Training-Age-Unit"] == AGE_UNIT
    thread.join()
    assert not failures


@pytest.mark.parametrize("age,unit", [(None, None), ("-1", AGE_UNIT), ("1.2", AGE_UNIT), ("3", "steps")])
def test_invalid_neighbor_age_is_ineligible(tmp_path, monkeypatch, age, unit):
    runtime, clock = make_runtime(tmp_path, "M1", offline=False)
    clock.now = 1
    headers = {"X-Client-ID": "1", "X-Model-Version": "1", "X-Completed-Round": "1", "X-Sample-Count": "5"}
    if age is not None:
        headers.update({"X-Training-Age": age, "X-Training-Age-Unit": unit})
    monkeypatch.setattr("app.timed_runtime.httpx.get", lambda url, **kw: httpx.Response(
        200, content=serialize_state_dict(runtime._model), headers=headers, request=httpx.Request("GET", url)))
    assert runtime._fetch_neighbor("http://peer", runtime._model) is None
    assert "invalid_training_age" in runtime.events.path.read_text()


def test_effective_weights_accept_zero_and_do_not_attribute_distillation():
    record = {"effective_aggregation_weights": {"0": 0, "1": 1}}
    assert _aggregation_weights(record, "unit") == {0: 0, 1: 1}
    assert _aggregation_weights({"effective_aggregation_weights": None}, "unit") is None
    with pytest.raises(ValueError, match="sum to one"):
        _aggregation_weights({"effective_aggregation_weights": {"0": .3}}, "unit")


def test_methods_and_parameters_are_resolved_before_launch(tmp_path):
    for method in ("M0", "M1", "M2", "M3", "M4"):
        spec = load_spec("mnist", "concentrated-r", method=method)
        env = compose_for(spec, "unit", tmp_path)["services"]["client-0"]["environment"]
        assert env["METHOD"] == method and spec["evaluation_interval_seconds"] == 20
    assert load_spec("cifar10", "concentrated-r", method="M2")["method_parameters"]["merge_limit_examples"] == 5625
    assert load_spec("mnist", "concentrated-r", mode="smoke", method="M2")["method_parameters"]["merge_limit_examples"] == 90
    for method, parameters in (("M3", {"rejoin_batches": 0}), ("M2", {"merge_limit_examples": -1}),
                               ("M4", {"distillation_weight": float("nan")}), ("M1", {"rejoin_batches": 2})):
        with pytest.raises(ValueError):
            load_spec("mnist", "concentrated-r", method=method, method_parameters=parameters)
