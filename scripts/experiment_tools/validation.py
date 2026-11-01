"""Validate every client's stock before starting the experiment services."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from app.collection import digest, read_json, resolve_plan
from app.config import Settings
from app.training import load_trainer


def validate_stocks(resolved: dict, directory: Path, data_dir: str | None = None) -> dict:
    schedule = read_json(directory / "schedule.json")
    seen = set()
    clients = {}
    stocks = {}
    for values in resolved["settings"]:
        settings = Settings(**values)
        if data_dir is not None:
            settings = replace(settings, data_dir=data_dir)
        settings.validate()
        data = load_trainer(settings.trainer_module).load_data(settings)
        if data.collection_source is None:
            raise ValueError("trainer did not provide collection stock")
        source = data.collection_source
        overlap = seen.intersection(source.sample_ids)
        if overlap:
            raise ValueError(f"local stocks overlap: {next(iter(overlap))}")
        seen.update(source.sample_ids)
        plan = resolve_plan(schedule, source, settings)
        key = str(settings.client_id)
        clients[key] = plan.plan_hash
        stocks[key] = {"count": len(source.dataset), "stock_hash": plan.resolved["stock_hash"],
                       "source_metadata": source.metadata}
    if set(clients) != {str(i) for i in range(len(resolved["settings"]))}:
        raise ValueError("resolved settings must declare every client exactly once")
    return {"version": 1, "schedule_hash": digest(schedule), "clients": clients, "stocks": stocks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("/results"))
    args = parser.parse_args()
    result = validate_stocks(read_json(args.directory / "resolved_experiment.json"), args.directory)
    path = args.directory / "preflight.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    print(f"Validated {len(result['clients'])} client stocks", flush=True)
