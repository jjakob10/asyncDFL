"""Export a manifest as independently runnable Compose and environment files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.experiment_tools.catalog import build_catalog, load_manifest, reject_collection_manifest
from scripts.experiment_tools.compose import build_compose


CLIENT_ENVIRONMENT_KEYS = {
    "CLIENT_ID", "CLIENT_NAME", "NEIGHBORS", "COMPONENT_ID", "COMPONENT_SIZE",
    "PAPER_ROLE",
}


def _validate_id(value: str, name: str, pattern: str) -> None:
    if (not isinstance(value, str) or not re.fullmatch(pattern, value)
            or len(value) > 128
            or re.fullmatch(r"CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]", value, re.IGNORECASE)):
        raise ValueError(f"invalid {name}: use letters, digits, hyphens, or underscores")


def _relative_path(target: Path, directory: Path) -> str:
    """Prefer relative references, retaining absolute paths across Windows drives."""
    try:
        value = Path(os.path.relpath(target, directory)).as_posix()
    except ValueError:
        value = target.as_posix()
    return value


def _dotenv(values: dict[str, str]) -> str:
    lines = []
    for key, value in sorted(values.items()):
        if any(character in value for character in "\r\n\0"):
            raise ValueError(f"{key} must be a single-line environment value")
        escaped = value.replace("'", "\\'")
        lines.append(f"{key}='{escaped}'\n")
    return "".join(lines)


def export_manifest(
    manifest_path: Path,
    output: Path,
    run_id: str | None = None,
    smoke: bool = False,
) -> list[Path]:
    """Validate the entire export before writing to a new or empty directory."""
    data = load_manifest(manifest_path)
    if not data:
        raise ValueError("manifest must not be empty")
    reject_collection_manifest(data)
    experiments = build_catalog(data)
    if not experiments:
        raise ValueError("manifest contains no experiments")
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output must be a new or empty directory: {output}")
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("export-%Y%m%dT%H%M%S%fZ")
    _validate_id(run_id, "run ID", r"[A-Za-z0-9][A-Za-z0-9_-]*")
    results_root_value = data.get("results_root", "results/experiments")
    if not isinstance(results_root_value, str) or not results_root_value.strip():
        raise ValueError("manifest results_root must be a nonempty path string")
    results_root = (REPOSITORY_ROOT / results_root_value).resolve()
    files: dict[Path, str] = {}
    seen_ids: set[str] = set()
    for experiment in experiments:
        _validate_id(experiment.experiment_id, "experiment ID", r"[a-z0-9][a-z0-9_-]*")
        if experiment.experiment_id in seen_ids:
            raise ValueError(f"duplicate experiment ID: {experiment.experiment_id}")
        seen_ids.add(experiment.experiment_id)
        if experiment.dataset not in {"mnist", "cifar10", "harth"}:
            raise ValueError(f"unsupported export dataset: {experiment.dataset}")
        dockerfile = (REPOSITORY_ROOT / experiment.dockerfile).resolve()
        if not dockerfile.is_relative_to(REPOSITORY_ROOT) or not dockerfile.is_file():
            raise ValueError(f"Dockerfile must exist inside this checkout: {dockerfile}")
        directory = output / experiment.experiment_id
        result_directory = results_root / experiment.experiment_id
        compose = build_compose(
            experiment, run_id, result_directory / run_id, smoke,
        )
        shared: dict[str, str] | None = None
        for client in experiment.clients:
            service = compose["services"][f"client-{client.client_id}"]
            environment = service.pop("environment")
            common = {key: value for key, value in environment.items()
                      if key not in CLIENT_ENVIRONMENT_KEYS}
            if shared is None:
                shared = common
            elif shared != common:
                raise ValueError("clients disagree on shared environment values")
            client_environment = {key: environment[key] for key in CLIENT_ENVIRONMENT_KEYS}
            client_file = f"env/client_{client.client_id}.env"
            files[directory / client_file] = _dotenv(client_environment)
            service["env_file"] = [client_file]
            service["environment"] = {key: f"${{{key}}}" for key in common}
            # Use the same interpolated identity for both the model and host mount.
            service["environment"]["RUN_ID"] = "${RUN_ID:?Set RUN_ID in .env}"
            service["volumes"] = [{
                "type": "bind",
                "source": "${EXPORT_RESULTS_BASE}/${RUN_ID}",
                "target": "/results",
                "bind": {"create_host_path": True},
            }]
            if "build" in service:
                service["build"] = {
                    "context": "${EXPORT_BUILD_CONTEXT}",
                    "dockerfile": "${EXPORT_DOCKERFILE}",
                }
        host_paths = {
            "EXPORT_BUILD_CONTEXT": _relative_path(REPOSITORY_ROOT, directory),
            "EXPORT_DOCKERFILE": dockerfile.relative_to(REPOSITORY_ROOT).as_posix(),
            "EXPORT_RESULTS_BASE": _relative_path(result_directory, directory),
        }
        files[directory / ".env"] = _dotenv({**(shared or {}), **host_paths})
        # JSON is a YAML subset accepted by Docker Compose, avoiding a new dependency.
        files[directory / "compose.yaml"] = json.dumps(compose, indent=2) + "\n"

    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    return [output / experiment.experiment_id for experiment in experiments]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", help="shared execution ID (default: UTC timestamp)")
    parser.add_argument("--smoke", action="store_true", help="use the runner's capped smoke settings")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        directories = export_manifest(
            args.manifest, args.output, args.run_id, args.smoke,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Exported {len(directories)} experiments to {args.output.resolve()}")
    for directory in directories:
        print(directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
