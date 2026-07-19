from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.config import Settings
from app.federation import clone_state_dict
from app.learning import DataBundle, MnistCNN
from app.runtime import ClientRuntime


def tiny_data() -> DataBundle:
    images = torch.zeros(4, 1, 28, 28)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    return DataBundle(loader, loader, sample_count=4)


def test_runtime_completes_and_writes_results(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        client_id=0,
        client_name="client-0",
        neighbors=(),
        rounds=2,
        results_dir=str(tmp_path),
    )
    runtime = ClientRuntime(settings, tiny_data())

    def fake_train(base, loader, config):
        updated = clone_state_dict(base)
        return updated, {"loss": 1.0, "accuracy": 0.5}

    monkeypatch.setattr("app.runtime.train_local", fake_train)
    monkeypatch.setattr(
        "app.runtime.evaluate", lambda state, loader: {"loss": 0.8, "accuracy": 0.6}
    )

    assert runtime.start("run-1") == "started"
    assert runtime.start("run-1") == "already_started"
    assert runtime.start("run-2") == "conflict"
    runtime.wait(timeout=5)

    status = runtime.status()
    assert status["state"] == "completed"
    assert status["current_round"] == 2
    assert status["model_version"] == 2
    assert (tmp_path / "client_0_metrics.jsonl").read_text().count("\n") == 2
    assert (tmp_path / "client_0_final.pt").exists()


def test_unavailable_neighbor_is_recorded_and_skipped(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        client_id=0,
        client_name="client-0",
        neighbors=("http://missing:8000",),
        request_retries=1,
        results_dir=str(tmp_path),
    )
    runtime = ClientRuntime(settings, tiny_data())
    reference = clone_state_dict(MnistCNN().state_dict())

    def fail(*args, **kwargs):
        raise RuntimeError("unreachable")

    monkeypatch.setattr(runtime, "_request_with_retries", fail)

    assert runtime._fetch_neighbor("http://missing:8000", reference) is None
    assert runtime.status()["recent_errors"][-1]["operation"] == "model fetch"


def test_start_flood_visits_eight_node_ring_once(monkeypatch, tmp_path: Path) -> None:
    runtimes: dict[str, ClientRuntime] = {}
    executions = [0] * 8
    run_id = "ring-run"

    for client_id in range(8):
        left = (client_id - 1) % 8
        right = (client_id + 1) % 8
        settings = Settings(
            client_id=client_id,
            client_name=f"client-{client_id}",
            neighbors=(
                f"http://client-{left}:8000",
                f"http://client-{right}:8000",
            ),
            results_dir=str(tmp_path),
        )
        runtimes[f"http://client-{client_id}:8000"] = ClientRuntime(settings, tiny_data())

    for runtime in runtimes.values():
        client_id = runtime.settings.client_id

        def propagate(node=runtime, node_id=client_id):
            executions[node_id] += 1
            for neighbor in node.settings.neighbors:
                runtimes[neighbor].start(run_id)

        monkeypatch.setattr(runtime, "_execute", propagate)

    assert runtimes["http://client-7:8000"].start(run_id) == "started"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(
        runtime.status()["run_id"] is None for runtime in runtimes.values()
    ):
        time.sleep(0.01)
    for runtime in runtimes.values():
        runtime.wait(timeout=1)

    assert all(runtime.status()["run_id"] == run_id for runtime in runtimes.values())
    assert executions == [1] * 8
