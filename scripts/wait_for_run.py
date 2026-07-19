"""Poll the eight Compose clients until the run completes or fails."""

from __future__ import annotations

import json
import time
import urllib.request


def status(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=3) as response:
        return json.load(response)


def main() -> None:
    while True:
        clients = [status(8000 + client_id) for client_id in range(8)]
        summary = " | ".join(
            f"{item['client_id']}:{item['state']} "
            f"{item['current_round']}/{item['total_rounds']}"
            for item in clients
        )
        print(summary, flush=True)
        states = {item["state"] for item in clients}
        if "failed" in states:
            raise SystemExit("at least one client failed")
        if states == {"completed"}:
            return
        time.sleep(5)


if __name__ == "__main__":
    main()

