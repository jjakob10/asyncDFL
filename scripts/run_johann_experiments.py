"""Run the time-budgeted Johann M0-M4 experiment matrix sequentially."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.johann.catalog import SCENARIOS, compose_for, load_spec
from experiments.johann.frozen import FrozenBundle
from app.aggregation import METHODS
from scripts.experiment_tools.execution import cleanup_compose


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(spec, run_id, results_root, frozen_bundle=None):
    if spec["mode"] == "collection" and not spec.get("frozen_bundle"):
        raise ValueError("collection requires --frozen-bundle")
    if spec.get("frozen_bundle") and frozen_bundle is None:
        raise ValueError("preparing a frozen spec requires its validated bundle")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", run_id):
        raise ValueError("run ID must be a safe path component of at most 91 characters")
    directory = (results_root / spec["dataset"] / spec["method"] / spec["scenario"] / f"{run_id}-s{spec['seed']}").resolve()
    directory.mkdir(parents=True, exist_ok=False)
    spec = {**spec, "run_id": directory.name, "batch_run_id": run_id}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=ROOT, capture_output=True)
    (directory / "working_tree.patch").write_bytes(diff.stdout)
    # Include hashes of untracked implementation files as well as tracked changes.
    import hashlib
    sources = {}
    for folder in ("app", "experiments", "scripts"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"}:
                sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    host = dict(code_revision=revision.stdout.strip(), working_tree_dirty=bool(dirty.stdout.strip()),
                source_sha256=sources,
                host_dependencies={name: importlib.metadata.version(name)
                                   for name in ("torch", "torchvision", "numpy", "httpx", "matplotlib")})
    if frozen_bundle is not None:
        # Training sources belong to the preserved image; host code only orchestrates it.
        spec["orchestrator"] = host
        (directory / "frozen_manifest.json").write_bytes(frozen_bundle.manifest_bytes)
        (directory / "selected_configuration.json").write_bytes(frozen_bundle.selection_bytes)
    else:
        spec.update(host)
    spec["created_at"] = datetime.now(timezone.utc).isoformat()
    json_write(directory / "resolved_experiment.json", spec)
    json_write(directory / "compose.json", compose_for(spec, spec["run_id"], directory))
    return directory, spec


def api_statuses():
    def read(client):
        response = httpx.get(f"http://127.0.0.1:{8000 + client}/status", timeout=2)
        response.raise_for_status()
        return response.json()
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(read, range(8)))


def schedule_run(run_id, lead_seconds=10):
    start_at = time.time() + lead_seconds
    def schedule(client):
        response = httpx.post(f"http://127.0.0.1:{8000 + client}/start",
                              json={"run_id": run_id, "start_at": start_at}, timeout=3)
        response.raise_for_status()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(schedule, range(8)))
    statuses = api_statuses()
    if time.time() >= start_at or any(
        s.get("run_id") != run_id or s.get("start_at") != start_at or s.get("state") == "failed"
        for s in statuses
    ):
        raise RuntimeError("not all eight clients acknowledged before the shared start")
    return start_at


def inspect_frozen_image(spec):
    expected = spec["expected_image_id"]
    result = subprocess.run(["docker", "image", "inspect", expected],
                            capture_output=True, text=True)
    restore = Path(spec["frozen_bundle"]["directory"]) / "images.tar"
    if result.returncode != 0:
        raise ValueError(f"frozen image unavailable: {expected}. Restore with docker load --input \"{restore}\"; "
                         "no build or pull will be performed")
    inspected = json.loads(result.stdout)
    if (not isinstance(inspected, list) or len(inspected) != 1
            or not isinstance(inspected[0], dict) or inspected[0].get("Id") != expected):
        raise ValueError(f"frozen image inspection disagrees with expected ID: {expected}")
    return inspected


def verify_frozen_containers(directory, spec, base):
    """Check engine-resolved image IDs, never Config.Image or mutable tags."""
    result = subprocess.run([*base, "ps", "--all", "--quiet"],
                            capture_output=True, text=True, check=True)
    ids = result.stdout.split()
    if not ids:
        json_write(directory / "container_inspection.json", [])
        raise ValueError("frozen execution requires exactly eight client containers")
    result = subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True, check=True)
    containers = json.loads(result.stdout)
    json_write(directory / "container_inspection.json", containers)
    project = json.loads((directory / "compose.json").read_text(encoding="utf-8"))["name"]
    expected_services = {f"client-{client}" for client in range(8)}
    seen = set()
    observed = {}
    if not isinstance(containers, list) or len(containers) != 8 or len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("frozen execution requires exactly eight client containers")
    if any(not isinstance(c, dict) for c in containers) or {c.get("Id") for c in containers} != set(ids):
        raise ValueError("frozen container IDs do not match the Compose project")
    for container in containers:
        labels = container.get("Config", {}).get("Labels", {}) or {}
        service = labels.get("com.docker.compose.service")
        if (labels.get("com.docker.compose.project") != project or service not in expected_services
                or service in seen or container.get("Id") not in ids):
            raise ValueError("frozen container service/project identity mismatch or duplicate")
        seen.add(service)
        if not container.get("State", {}).get("Running") or container.get("Image") != spec["expected_image_id"]:
            raise ValueError(f"frozen container image/running state mismatch: {service}; "
                             f"expected={spec['expected_image_id']}, observed={container.get('Image')}")
        observed[service] = container["Image"]
    if seen != expected_services:
        raise ValueError("frozen execution is missing client services")
    spec["observed_image_ids"] = observed
    spec["image_id"] = spec["expected_image_id"]
    json_write(directory / "resolved_experiment.json", spec)


def execute(directory, spec):
    base = ["docker", "compose", "-f", str(directory / "compose.json")]
    outcome = {"state": "failed", "run_id": spec["run_id"]}
    up_attempted = False
    try:
        if spec["mode"] == "collection" and not spec.get("frozen_bundle"):
            raise ValueError("collection requires --frozen-bundle")
        if spec.get("frozen_bundle"):
            # Also check here so an image removed after whole-selection preflight cannot be replaced.
            json_write(directory / "image.json", inspect_frozen_image(spec))
        with (directory / "docker_start.log").open("w", encoding="utf-8") as log:
            up_args = ["up", "--no-build", "--pull", "never", "-d"] if spec.get("frozen_bundle") else ["up", "--build", "-d"]
            up_attempted = True
            subprocess.run([*base, *up_args], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        deadline = time.monotonic() + 300
        while True:
            try:
                statuses = api_statuses()
                if len(statuses) == 8 and all(s["state"] == "waiting" for s in statuses):
                    break
                if any(s["state"] == "failed" for s in statuses):
                    raise RuntimeError("a client failed before scheduling")
            except (httpx.HTTPError, KeyError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("clients were not ready within five minutes")
            time.sleep(1)
        if not spec.get("frozen_bundle"):
            inspect = subprocess.run(["docker", "image", "inspect", f"dfl-johann-{spec['dataset']}:latest"],
                                     capture_output=True, text=True, check=True)
            json_write(directory / "image.json", json.loads(inspect.stdout))
            spec["image_id"] = json.loads(inspect.stdout)[0]["Id"]
        dependencies = subprocess.run([*base, "exec", "-T", "client-0", "python", "-m", "pip", "freeze"],
                                      capture_output=True, text=True, check=True)
        (directory / "container_dependencies.txt").write_text(dependencies.stdout, encoding="utf-8")
        if spec.get("frozen_bundle"):
            verify_frozen_containers(directory, spec, base)
        start_at = schedule_run(spec["run_id"])
        spec["start_at"] = start_at
        json_write(directory / "resolved_experiment.json", spec)
        json_write(directory / "schedule.json", {
            "start_at": start_at, "offline_at": start_at + spec["duration_seconds"] * spec["offline_start_fraction"],
            "return_at": start_at + spec["duration_seconds"] * spec["offline_end_fraction"],
            "end_at": start_at + spec["duration_seconds"], "offline_client_ids": spec["offline_client_ids"],
        })
        deadline = time.monotonic() + spec["duration_seconds"] + 310
        last_report = 0
        while time.monotonic() < deadline:
            statuses = api_statuses()
            json_write(directory / "last_status.json", statuses)
            if any(s["state"] == "failed" for s in statuses):
                raise RuntimeError("a client failed; inspect last_status.json and container.log")
            if all(s["state"] == "completed" for s in statuses):
                from experiments.johann.analysis import summarize_run
                summarize_run(directory)
                outcome["state"] = "completed"
                return True
            if time.monotonic() - last_report >= 15:
                print(f"{spec['dataset']} {spec['scenario']} seed={spec['seed']}: " + " | ".join(
                    f"{s['client_id']}:{s['availability']} cycles={s['current_round']}" for s in statuses), flush=True)
                last_report = time.monotonic()
            time.sleep(1)
        raise TimeoutError("run exceeded its completion grace period")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, httpx.HTTPError) as exc:
        outcome["error"] = str(exc)
        print(f"error: {exc}", file=sys.stderr)
        return False
    finally:
        outcome["finished_at"] = datetime.now(timezone.utc).isoformat()
        json_write(directory / "outcome.json", outcome)
        if up_attempted:
            try:
                result = subprocess.run([*base, "ps", "-a", "--format", "json"], capture_output=True, text=True)
                (directory / "container_status.jsonl").write_text(result.stdout, encoding="utf-8")
            finally:
                cleanup_compose(directory / "compose.json", directory)



def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("list", "render-compose", "smoke", "pilot", "run"):
        child = sub.add_parser(command)
        child.add_argument("--dataset", choices=("mnist", "cifar10", "all"), default="mnist")
        child.add_argument("--scenario", choices=SCENARIOS)
        child.add_argument("--all", action="store_true", help="all six scenarios")
        child.add_argument("--seeds", type=int, nargs="+")
        child.add_argument("--methods", choices=METHODS, nargs="+", default=["M0"])
        child.add_argument("--merge-limit-examples", type=int)
        child.add_argument("--merge-limit-multiplier", type=int)
        child.add_argument("--rejoin-batches", type=int)
        child.add_argument("--distillation-weight", type=float)
        child.add_argument("--distillation-temperature", type=float)
        child.add_argument("--selected-config", type=Path, help="validation selection JSON; does not freeze the protocol")
        child.add_argument("--frozen-bundle", type=Path,
                           help="preserved images, sources and selected parameters; required for run")
        if command == "smoke":
            child.add_argument("--backend", choices=("docker", "local"), default="docker")
        child.add_argument("--duration", type=float)
        child.add_argument("--interval", type=float)
        child.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        child.add_argument("--results-root", type=Path, default=ROOT / "results" / "johann")
    summary = sub.add_parser("summarize")
    summary.add_argument("path", type=Path)
    summary.add_argument("--run-id", help="restrict a collection to one batch ID, without -s<seed>")
    summary.add_argument("--mode", choices=("smoke", "pilot", "collection"))
    calibration = sub.add_parser("calibrate", help="eleven validation-only trials per dataset, seed 7")
    calibration.add_argument("--dataset", choices=("mnist", "cifar10", "all"), default="mnist")
    calibration.add_argument("--run-id", required=True)
    calibration.add_argument("--results-root", type=Path, default=ROOT / "results" / "johann")
    calibration.add_argument("--render-only", action="store_true")
    selection = sub.add_parser("select-calibration")
    selection.add_argument("path", type=Path)
    selection.add_argument("--run-id", required=True)
    audit = sub.add_parser("audit-m0", help="read-only inventory of existing M0 evidence and E2 gaps")
    audit.add_argument("--results-root", type=Path, default=ROOT / "results" / "johann")
    audit.add_argument("--frozen-bundle", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "audit-m0":
            from experiments.johann.audit import audit_m0
            audit_m0(args.results_root, args.frozen_bundle, args.output_dir)
            print(args.output_dir.resolve() / "relatorio.md")
            return 0
        if args.command == "select-calibration":
            from experiments.johann.calibration import select_calibration
            selection = select_calibration(args.path, args.run_id)
            path = args.path / f"{args.run_id}-selection.json"
            # The run ID is also a file component here; validate it before writing.
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,90}", args.run_id):
                raise ValueError("invalid calibration ID")
            if path.exists():
                raise ValueError("selection already exists; preserve its provenance")
            json_write(path, selection)
            print(path)
            return 0
        if args.command == "calibrate":
            from experiments.johann.calibration import calibration_specs
            datasets = ("mnist", "cifar10") if args.dataset == "all" else (args.dataset,)
            specs = [(name, {**spec, "calibration_id": args.run_id, "calibration_trial": name})
                     for dataset in datasets for name, spec in calibration_specs(dataset)]
            for name, spec in specs:
                directory, resolved = prepare(spec, f"{args.run_id}-{name}", args.results_root.resolve())
                print(directory, flush=True)
                if not args.render_only and not execute(directory, resolved):
                    return 1
            return 0
        if args.command == "summarize":
            from experiments.johann.analysis import summarize_collection, summarize_run
            if (args.path / "resolved_experiment.json").is_file():
                summarize_run(args.path)
            else:
                summarize_collection(args.path, run_id=args.run_id, mode=args.mode)
            return 0
        if args.command == "run" and args.frozen_bundle is None:
            raise ValueError("run requires --frozen-bundle; use pilot or smoke for development")
        if args.frozen_bundle is not None:
            if args.selected_config is not None:
                raise ValueError("--frozen-bundle cannot be combined with --selected-config")
            if getattr(args, "backend", "docker") == "local":
                raise ValueError("--frozen-bundle requires the Docker backend")
            if args.command != "smoke" and (args.duration is not None or args.interval is not None):
                raise ValueError("--frozen-bundle duration/interval overrides are only allowed for smoke")
        datasets = ("mnist", "cifar10") if args.dataset == "all" else (args.dataset,)
        scenarios = SCENARIOS if args.all or args.command == "list" else (args.scenario or "concentrated-r",)
        if args.all and args.scenario:
            raise ValueError("choose --all or --scenario")
        seeds = args.seeds or ([42, 43, 44, 45, 46] if args.command == "run" else [7])
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be unique")
        if len(args.methods) != len(set(args.methods)):
            raise ValueError("methods must be unique")
        if args.merge_limit_examples is not None and args.merge_limit_multiplier is not None:
            raise ValueError("choose an absolute merge limit or a multiplier")
        overrides = {key: getattr(args, key) for key in (
            "merge_limit_examples", "merge_limit_multiplier", "rejoin_batches",
            "distillation_weight", "distillation_temperature") if getattr(args, key) is not None}
        if args.frozen_bundle is not None and overrides:
            raise ValueError("--frozen-bundle cannot be combined with method parameter overrides")
        if any(key.startswith("merge_limit") for key in overrides) and "M2" not in args.methods:
            raise ValueError("merge limits require M2")
        if "rejoin_batches" in overrides and not set(args.methods) & {"M3", "M4"}:
            raise ValueError("rejoin batches require M3 or M4")
        if any(key.startswith("distillation") for key in overrides) and "M4" not in args.methods:
            raise ValueError("distillation parameters require M4")
        # Validate the entire selection before creating artifacts or launching Docker.
        specs = []
        bundle = FrozenBundle(args.frozen_bundle) if args.frozen_bundle is not None else None
        selected = json.loads(args.selected_config.read_text(encoding="utf-8")) if args.selected_config else None
        for dataset in datasets:
            for scenario in scenarios:
                for seed in seeds:
                    mode = "collection" if args.command == "run" else "pilot"
                    duration, interval = args.duration, args.interval
                    if args.command == "smoke":
                        mode = "smoke"
                        duration = duration if duration is not None else 30
                        interval = interval if interval is not None else duration / 20
                    for method in args.methods:
                        keys = ({"merge_limit_examples", "merge_limit_multiplier"} if method == "M2" else
                                {"rejoin_batches"} if method == "M3" else
                                {"rejoin_batches", "distillation_weight", "distillation_temperature"} if method == "M4" else set())
                        parameters = dict(selected["datasets"][dataset]["methods"][method]) if selected else {}
                        parameters.update({key: value for key, value in overrides.items() if key in keys})
                        specs.append(bundle.resolve_spec(dataset, scenario, seed, mode, method, duration, interval)
                                     if bundle is not None else
                                     load_spec(dataset, scenario, seed, duration, interval, mode, method, parameters))
                        specs[-1]["execution_backend"] = "local_smoke" if getattr(args, "backend", "docker") == "local" else "docker"
                        if selected:
                            specs[-1]["calibration_id"] = selected["calibration_id"]
                            specs[-1]["calibration_confirmation"] = args.command == "pilot" and seed in (8, 9)
        if args.command == "list":
            for spec in specs:
                print(f"{spec['dataset']}/{spec['method']}/{spec['scenario']} {spec['method_parameters']}")
            return 0
        if bundle is not None and args.command != "render-compose":
            # Check every selected dataset before preparing the first network.
            for dataset in datasets:
                inspect_frozen_image(next(s for s in specs if s["dataset"] == dataset))
        for spec in specs:
            directory, spec = prepare(spec, args.run_id, args.results_root.resolve(), bundle)
            print(directory, flush=True)
            if args.command != "render-compose":
                if getattr(args, "backend", "docker") == "local":
                    from scripts.johann_local_smoke import execute_local_smoke
                    success = execute_local_smoke(directory, spec, args.results_root.resolve() / "local_data" / spec["dataset"])
                else:
                    success = execute(directory, spec)
                if not success:
                    return 1
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
