from pathlib import Path
import time

from fastapi.testclient import TestClient

from app.config import Settings
from app.federation import serialize_state_dict
from app.main import create_app


class StubRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.run_id: str | None = None

    def close(self):
        pass

    def start(self, run_id: str) -> str:
        if self.run_id == run_id:
            return "already_started"
        if self.run_id is not None:
            return "conflict"
        self.run_id = run_id
        return "started"

    def model_snapshot(self):
        return serialize_state_dict({}), {
            "X-Client-ID": "0",
            "X-Model-Version": "0",
            "X-Completed-Round": "0",
            "X-Sample-Count": "100",
        }

    def status(self):
        return {
            "client_id": 0,
            "state": "waiting" if self.run_id is None else "running",
            "run_id": self.run_id,
        }


def make_client(tmp_path: Path) -> TestClient:
    settings = Settings(
        client_id=0,
        client_name="client-0",
        neighbors=(),
        results_dir=str(tmp_path),
    )
    runtime = StubRuntime(settings)
    return TestClient(create_app(settings, runtime, auto_start_head=False))


def test_health_status_and_model(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/health").json() == {"ready": True, "client_id": 0}
        assert client.get("/status").json()["state"] == "waiting"
        response = client.get("/model")
        assert response.status_code == 200
        assert response.headers["x-model-version"] == "0"
        assert response.headers["x-sample-count"] == "100"


def test_start_is_idempotent_and_rejects_a_different_run(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        first = client.post("/start", json={"run_id": "experiment-1"})
        duplicate = client.post("/start", json={"run_id": "experiment-1"})
        conflict = client.post("/start", json={"run_id": "experiment-2"})

        assert first.status_code == 202
        assert first.json()["result"] == "started"
        assert duplicate.status_code == 202
        assert duplicate.json()["result"] == "already_started"
        assert conflict.status_code == 409


def test_start_validates_run_id(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert client.post("/start", json={"run_id": ""}).status_code == 422


def test_head_auto_starts_with_configured_shared_run_id(tmp_path: Path) -> None:
    settings = Settings(
        client_id=8,
        client_name="client-8",
        client_count=10,
        head_client_ids=(7, 8, 9),
        run_id="shared-paper-run",
        neighbors=(),
        head_start_delay=0.01,
        results_dir=str(tmp_path),
    )
    runtime = StubRuntime(settings)

    with TestClient(create_app(settings, runtime, auto_start_head=True)) as client:
        deadline = time.monotonic() + 1
        while runtime.run_id is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.get("/status").json()["run_id"] == "shared-paper-run"
