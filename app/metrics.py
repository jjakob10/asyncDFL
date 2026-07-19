from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class MetricsWriter:
    def __init__(self, results_dir: str, client_id: int) -> None:
        self.directory = Path(results_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"client_{client_id}_metrics.jsonl"
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())

