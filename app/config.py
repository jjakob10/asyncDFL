from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field


def parse_client_classes(value: object, client_count: int) -> dict[int, tuple[int, ...]]:
    """Validate a JSON class assignment without silently coercing labels."""
    if not isinstance(value, dict):
        raise ValueError("CLIENT_CLASSES must be a JSON object")
    result = {}
    for key, classes in value.items():
        if isinstance(key, bool) or not str(key).isdigit():
            raise ValueError("CLIENT_CLASSES keys must be client IDs")
        client_id = int(key)
        if not 0 <= client_id < client_count or client_id in result:
            raise ValueError("CLIENT_CLASSES contains an invalid or duplicate client ID")
        if not isinstance(classes, (list, tuple)) or any(
            type(label) is not int or label < 0 for label in classes
        ):
            raise ValueError("CLIENT_CLASSES values must be lists of nonnegative integers")
        if len(set(classes)) != len(classes):
            raise ValueError("CLIENT_CLASSES must not contain duplicate classes")
        result[client_id] = tuple(classes)
    return result


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

def _env_int_tuple(name: str, default: str = "") -> tuple[int, ...]:
    value = os.getenv(name, default).strip()
    if not value:
        return ()
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate client IDs")
    return result


@dataclass(frozen=True)
class Settings:
    client_id: int
    client_name: str
    trainer_module: str = "experiments.mnist.trainer"
    client_count: int = 8
    head_client_ids: tuple[int, ...] = (7,)
    run_id: str = "dfl-run"
    empty_client_ids: tuple[int, ...] = ()
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
    api_port: int = 9000
    torch_num_threads: int = 1
    experiment_id: str = "custom"
    topology_name: str = "custom"
    component_id: int = 0
    component_size: int = 1
    paper_role: str = "R"
    ton_iot_path: str = ""
    client_classes: dict[int, tuple[int, ...]] = field(default_factory=dict)
    run_duration_seconds: float = 0.0
    evaluation_interval_seconds: float = 10.0
    offline_client_ids: tuple[int, ...] = ()
    offline_start_fraction: float = 0.3
    offline_end_fraction: float = 0.6
    auto_start_head: bool = True
    johann_distribution: str = "distributed"
    evaluation_split: str = "validation"
    method: str = "M0"
    merge_limit_examples: int = 6750
    rejoin_batches: int = 16
    distillation_weight: float = 0.5
    distillation_temperature: float = 2.0

    @property
    def is_head(self) -> bool:
        return self.client_id in self.head_client_ids

    @property
    def has_local_data(self) -> bool:
        return self.client_id not in self.empty_client_ids

    @classmethod
    def from_env(cls) -> "Settings":
        raw_name = os.getenv("CLIENT_NAME", os.getenv("CLIENT_ID", "client-0"))
        client_id = _client_id(os.getenv("CLIENT_ID", raw_name))
        client_count = int(os.getenv("CLIENT_COUNT", "8"))
        api_port = int(os.getenv("API_PORT", "9000"))

        explicit_neighbors = os.getenv("NEIGHBORS")
        if explicit_neighbors is not None:
            neighbors = tuple(
                url.strip().rstrip("/")
                for url in explicit_neighbors.strip().split(",")
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
            head_client_ids=_env_int_tuple("HEAD_CLIENT_IDS", "7"),
            run_id=os.getenv("RUN_ID", "dfl-run").strip(),
            empty_client_ids=_env_int_tuple("EMPTY_CLIENT_IDS"),
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
            experiment_id=os.getenv("EXPERIMENT_ID", "custom").strip(),
            topology_name=os.getenv("TOPOLOGY_NAME", "custom").strip(),
            component_id=int(os.getenv("COMPONENT_ID", "0")),
            component_size=int(os.getenv("COMPONENT_SIZE", "1")),
            paper_role=os.getenv("PAPER_ROLE", "R").strip().upper(),
            ton_iot_path=os.getenv("TON_IOT_PATH", "").strip(),
            client_classes=parse_client_classes(
                json.loads(os.getenv("CLIENT_CLASSES", "{}")), client_count
            ),
            run_duration_seconds=float(os.getenv("RUN_DURATION_SECONDS", "0")),
            evaluation_interval_seconds=float(os.getenv("EVALUATION_INTERVAL_SECONDS", "10")),
            offline_client_ids=_env_int_tuple("OFFLINE_CLIENT_IDS"),
            offline_start_fraction=float(os.getenv("OFFLINE_START_FRACTION", "0.3")),
            offline_end_fraction=float(os.getenv("OFFLINE_END_FRACTION", "0.6")),
            auto_start_head=_env_bool("AUTO_START_HEAD", True),
            johann_distribution=os.getenv("JOHANN_DISTRIBUTION", "distributed"),
            evaluation_split=os.getenv("EVALUATION_SPLIT", "validation"),
            method=os.getenv("METHOD", "M0"),
            merge_limit_examples=int(os.getenv("MERGE_LIMIT_EXAMPLES", "6750")),
            rejoin_batches=int(os.getenv("REJOIN_BATCHES", "16")),
            distillation_weight=float(os.getenv("DISTILLATION_WEIGHT", "0.5")),
            distillation_temperature=float(os.getenv("DISTILLATION_TEMPERATURE", "2")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.method not in {"M0", "M1", "M2", "M3", "M4"}:
            raise ValueError("METHOD must be M0, M1, M2, M3 or M4")
        if self.method != "M0" and self.run_duration_seconds <= 0:
            raise ValueError("M1-M4 require the timed runtime")
        if type(self.merge_limit_examples) is not int or self.merge_limit_examples < 0:
            raise ValueError("MERGE_LIMIT_EXAMPLES must be a nonnegative integer")
        if type(self.rejoin_batches) is not int or self.rejoin_batches < 1:
            raise ValueError("REJOIN_BATCHES must be a positive integer")
        if not math.isfinite(self.distillation_weight) or self.distillation_weight < 0:
            raise ValueError("DISTILLATION_WEIGHT must be finite and nonnegative")
        if not math.isfinite(self.distillation_temperature) or self.distillation_temperature <= 0:
            raise ValueError("DISTILLATION_TEMPERATURE must be finite and positive")
        if not math.isfinite(self.run_duration_seconds) or self.run_duration_seconds < 0:
            raise ValueError("RUN_DURATION_SECONDS must be finite and nonnegative")
        if not math.isfinite(self.evaluation_interval_seconds) or self.evaluation_interval_seconds <= 0:
            raise ValueError("EVALUATION_INTERVAL_SECONDS must be finite and positive")
        if not 0 < self.offline_start_fraction < self.offline_end_fraction < 1:
            raise ValueError("offline fractions must satisfy 0 < start < end < 1")
        if len(set(self.offline_client_ids)) != len(self.offline_client_ids) or any(
            not 0 <= item < self.client_count for item in self.offline_client_ids
        ):
            raise ValueError("OFFLINE_CLIENT_IDS must contain unique valid client IDs")
        if self.offline_client_ids and not self.run_duration_seconds:
            raise ValueError("offline schedules require RUN_DURATION_SECONDS")
        if self.johann_distribution not in {"distributed", "concentrated"}:
            raise ValueError("JOHANN_DISTRIBUTION must be distributed or concentrated")
        if self.evaluation_split not in {"validation", "test"}:
            raise ValueError("EVALUATION_SPLIT must be validation or test")
        if not 1 <= self.api_port <= 65535:
            raise ValueError("API_PORT must be between 1 and 65535")
        if not self.trainer_module:
            raise ValueError("TRAINER_MODULE must not be empty")
        if not 0 <= self.client_id < self.client_count:
            raise ValueError("client_id must be between 0 and client_count - 1")
        if not self.head_client_ids:
            raise ValueError("HEAD_CLIENT_IDS must contain at least one client ID")
        if any(not 0 <= item < self.client_count for item in self.head_client_ids):
            raise ValueError(
                "HEAD_CLIENT_IDS entries must be between 0 and client_count - 1"
            )
        if any(not 0 <= item < self.client_count for item in self.empty_client_ids):
            raise ValueError(
                "EMPTY_CLIENT_IDS entries must be between 0 and client_count - 1"
            )
        if not self.run_id or len(self.run_id) > 128:
            raise ValueError("RUN_ID must contain between 1 and 128 characters")
        if self.paper_role not in {"R", "D", "E", "ED"}:
            raise ValueError("PAPER_ROLE must be one of R, D, E, or ED")
        if self.component_size < 1:
            raise ValueError("COMPONENT_SIZE must be positive")
        if self.partition_mode not in {"iid", "dirichlet", "manual"}:
            raise ValueError("PARTITION_MODE must be 'iid', 'dirichlet', or 'manual'")
        mapping = parse_client_classes(self.client_classes, self.client_count)
        if self.partition_mode == "manual":
            if set(mapping) != set(range(self.client_count)):
                raise ValueError("CLIENT_CLASSES must specify every client in manual mode")
            for client_id, classes in mapping.items():
                if bool(classes) == (client_id in self.empty_client_ids):
                    raise ValueError("CLIENT_CLASSES must be empty only for empty clients")
        if self.rounds < 1 or self.local_epochs < 1 or self.batch_size < 1:
            raise ValueError("rounds, local_epochs, and batch_size must be positive")
        if self.dirichlet_alpha <= 0:
            raise ValueError("DIRICHLET_ALPHA must be positive")
        if self.request_retries < 1:
            raise ValueError("REQUEST_RETRIES must be at least 1")
        if self.torch_num_threads < 1:
            raise ValueError("TORCH_NUM_THREADS must be at least 1")
