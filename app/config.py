from __future__ import annotations

import os
import re
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _client_id(value: str) -> int:
    match = re.search(r"(\d+)$", value.strip())
    if not match:
        raise ValueError(f"CLIENT_ID/CLIENT_NAME must end in a number, got {value!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class Settings:
    client_id: int
    client_name: str
    trainer_module: str = "experiments.mnist.trainer"
    client_count: int = 8
    head_client_id: int = 7
    neighbors: tuple[str, ...] = ()
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    seed: int = 42
    partition_mode: str = "iid"
    dirichlet_alpha: float = 0.5
    min_partition_size: int = 100
    request_timeout: float = 5.0
    request_retries: int = 3
    retry_delay: float = 1.0
    head_start_delay: float = 5.0
    data_dir: str = "/data"
    results_dir: str = "/results"
    download_data: bool = False
    max_train_samples: int = 0
    max_test_samples: int = 0
    api_port: int = 8000
    torch_num_threads: int = 1

    @property
    def is_head(self) -> bool:
        return self.client_id == self.head_client_id

    @classmethod
    def from_env(cls) -> "Settings":
        raw_name = os.getenv("CLIENT_NAME", os.getenv("CLIENT_ID", "client-0"))
        client_id = _client_id(os.getenv("CLIENT_ID", raw_name))
        client_count = int(os.getenv("CLIENT_COUNT", "8"))
        api_port = int(os.getenv("API_PORT", "8000"))

        explicit_neighbors = os.getenv("NEIGHBORS", "").strip()
        if explicit_neighbors:
            neighbors = tuple(
                url.strip().rstrip("/")
                for url in explicit_neighbors.split(",")
                if url.strip()
            )
        else:
            prefix = os.getenv("GNS3_IP_PREFIX", "10.10.0.")
            offset = int(os.getenv("GNS3_IP_OFFSET", "10"))
            left = (client_id - 1) % client_count
            right = (client_id + 1) % client_count
            neighbors = (
                f"http://{prefix}{offset + left}:{api_port}",
                f"http://{prefix}{offset + right}:{api_port}",
            )

        settings = cls(
            client_id=client_id,
            client_name=raw_name,
            trainer_module=os.getenv(
                "TRAINER_MODULE", "experiments.mnist.trainer"
            ).strip(),
            client_count=client_count,
            head_client_id=int(os.getenv("HEAD_CLIENT_ID", "7")),
            neighbors=neighbors,
            rounds=int(os.getenv("ROUNDS", "10")),
            local_epochs=int(os.getenv("LOCAL_EPOCHS", "1")),
            batch_size=int(os.getenv("BATCH_SIZE", "64")),
            learning_rate=float(os.getenv("LEARNING_RATE", "0.01")),
            momentum=float(os.getenv("MOMENTUM", "0.9")),
            seed=int(os.getenv("SEED", "42")),
            partition_mode=os.getenv("PARTITION_MODE", "iid").lower(),
            dirichlet_alpha=float(os.getenv("DIRICHLET_ALPHA", "0.5")),
            min_partition_size=int(os.getenv("MIN_PARTITION_SIZE", "100")),
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "5")),
            request_retries=int(os.getenv("REQUEST_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1")),
            head_start_delay=float(os.getenv("HEAD_START_DELAY", "5")),
            data_dir=os.getenv("DATA_DIR", "/data"),
            results_dir=os.getenv("RESULTS_DIR", "/results"),
            download_data=_env_bool("DOWNLOAD_DATA", False),
            max_train_samples=int(os.getenv("MAX_TRAIN_SAMPLES", "0")),
            max_test_samples=int(os.getenv("MAX_TEST_SAMPLES", "0")),
            api_port=api_port,
            torch_num_threads=int(os.getenv("TORCH_NUM_THREADS", "1")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.trainer_module:
            raise ValueError("TRAINER_MODULE must not be empty")
        if not 0 <= self.client_id < self.client_count:
            raise ValueError("client_id must be between 0 and client_count - 1")
        if not 0 <= self.head_client_id < self.client_count:
            raise ValueError("head_client_id must be between 0 and client_count - 1")
        if self.partition_mode not in {"iid", "dirichlet"}:
            raise ValueError("PARTITION_MODE must be 'iid' or 'dirichlet'")
        if self.rounds < 1 or self.local_epochs < 1 or self.batch_size < 1:
            raise ValueError("rounds, local_epochs, and batch_size must be positive")
        if self.dirichlet_alpha <= 0:
            raise ValueError("DIRICHLET_ALPHA must be positive")
        if self.request_retries < 1:
            raise ValueError("REQUEST_RETRIES must be at least 1")
        if self.torch_num_threads < 1:
            raise ValueError("TORCH_NUM_THREADS must be at least 1")
