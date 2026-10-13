"""Poll a configurable number of clients until the run completes or fails."""

from __future__ import annotations

import argparse
import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence


def status(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=3) as response:
        return json.load(response)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-count",
        type=int,
        default=8,
        help="number of clients to monitor (default: 8)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=8000,
        help="first client port; subsequent ports are consecutive (default: 8000)",
    )
    args = parser.parse_args(argv)
    if args.client_count < 1:
        parser.error("--client-count must be positive")
    if not 1 <= args.base_port <= 65535:
        parser.error("--base-port must be between 1 and 65535")
    if args.base_port + args.client_count - 1 > 65535:
        parser.error("the last client port must not exceed 65535")

    while True:
        try:
            clients = [
                status(args.base_port + client_id)
                for client_id in range(args.client_count)
            ]
        except (http.client.HTTPException, OSError, urllib.error.URLError) as exc:
            print(f"waiting for all client APIs: {exc}", flush=True)
            time.sleep(5)
            continue
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
