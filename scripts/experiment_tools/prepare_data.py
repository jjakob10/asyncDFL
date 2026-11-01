"""Prepare a dataset through an optional trainer capability."""
from __future__ import annotations

import argparse
from pathlib import Path

from app.training import load_trainer


def prepare_dataset(trainer_module: str, data_dir: Path) -> None:
    trainer = load_trainer(trainer_module)
    prepare = getattr(trainer, "prepare_data", None)
    if not callable(prepare):
        raise ValueError(f"trainer {trainer_module!r} must expose prepare_data(data_dir)")
    data_dir.mkdir(parents=True, exist_ok=True)
    prepare(data_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-module", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/data"))
    args = parser.parse_args(argv)
    prepare_dataset(args.trainer_module, args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
