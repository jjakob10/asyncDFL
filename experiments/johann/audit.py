"""Read-only inventory of M0 evidence; writes only a separate audit report."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from experiments.johann.analysis import (
    integrated_outcome, json_sha256, load_clients, pairing_protocol, validate_paired_networks,
)
from experiments.johann.catalog import GROUPS, SCENARIOS
from experiments.johann.frozen import FrozenBundle


READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError)
PROTOCOL_FIELDS = (
    "duration_seconds", "evaluation_interval_seconds", "evaluation_split", "tracked_classes",
    "group_r", "group_c", "edges", "offline_start_fraction", "offline_end_fraction",
    "validation_fraction", "concentration_fraction", "local_epochs", "batch_size", "learning_rate", "momentum",
)
INVENTORY_FIELDS = (
    "directory", "dataset", "method", "scenario", "seed", "batch", "run_id", "mode", "evaluation_split",
    "duration_seconds", "evaluation_interval_seconds", "execution_backend", "status", "completed_clients",
    "coverage", "diagnostics", "historical_use", "frozen_use", "frozen_reasons", "issues", "protocol_sha256",
)
STATUS_LABELS = {
    "completed": "Concluída e íntegra", "configuration_only": "Apenas configuração gerada",
    "smoke_completed": "Smoke concluído", "completed_partial_coverage": "Concluída com lacunas de avaliação",
    "incomplete": "Arquivos ou identificação incompletos", "official_test_metadata_only": "Teste oficial: somente metadados",
}


class Inputs:
    """Hash every input read and detect changes during the audit."""

    def __init__(self):
        self.hashes: dict[str, str] = {}
        self.missing: set[str] = set()

    def read(self, path: Path) -> bytes:
        path = path.resolve()
        try:
            value = path.read_bytes()
        except FileNotFoundError:
            self.missing.add(str(path))
            raise
        digest = hashlib.sha256(value).hexdigest()
        if str(path) in self.hashes and self.hashes[str(path)] != digest:
            raise ValueError(f"input changed during audit: {path}")
        self.hashes[str(path)] = digest
        return value

    def json(self, path: Path) -> dict:
        value = json.loads(self.read(path))
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object: {path}")
        return value

    def lines(self, path: Path) -> list[dict]:
        rows = [json.loads(line) for line in self.read(path).decode("utf-8").splitlines() if line.strip()]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"expected JSONL objects: {path}")
        return rows

    def verify_unchanged(self):
        for name, digest in list(self.hashes.items()):
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"input changed during audit: {name}")
        if any(Path(name).exists() for name in self.missing):
            raise ValueError("previously missing inputs appeared during audit; repeat with stable results")


def _identity(value: dict, spec: dict, client: int, name: str):
    if value.get("run_id") != spec["run_id"] or value.get("client_id") != client:
        raise ValueError(f"identity mismatch: {name}")


def _inspect_record(path: Path, spec: dict, inputs: Inputs) -> dict:
    directory = path.parent
    row = {key: spec.get(key, "") for key in INVENTORY_FIELDS}
    row.update(directory=str(directory.resolve()), method=spec.get("method", "unknown"),
               batch=spec.get("batch_run_id", str(spec.get("run_id", "")).rsplit("-s", 1)[0]),
               execution_backend=spec.get("execution_backend", "not_recorded"), status="incomplete",
               completed_clients=0, coverage={}, diagnostics={}, historical_use="not_eligible",
               frozen_use="not_eligible", frozen_reasons=[], issues=[], protocol_sha256="",
               _spec=spec, _base_valid=False)
    if row["method"] != "M0":
        row["issues"].append("method not recorded; M0 was not assumed from its directory")
        return row
    if spec.get("mode") == "collection" or spec.get("evaluation_split") == "test":
        row.update(status="official_test_metadata_only", historical_use="excluded_official_test")
        return row
    outcome_path = directory / "outcome.json"
    if not outcome_path.exists():
        inputs.missing.add(str(outcome_path.resolve()))
        row["status"] = ("incomplete_without_outcome" if any(directory.glob("client_*_*.json*"))
                         else "configuration_only")
        return row
    try:
        outcome = inputs.json(outcome_path)
        if outcome.get("run_id", spec.get("run_id")) != spec.get("run_id"):
            raise ValueError("outcome identity mismatch")
        if spec.get("mode") == "smoke":
            row.update(status="smoke_" + str(outcome.get("state", "unknown")), historical_use="excluded_smoke")
            return row
        if outcome.get("state") != "completed":
            row["status"] = "not_completed_" + str(outcome.get("state", "unknown"))
            return row
        if spec.get("mode") != "pilot" or spec.get("evaluation_split") != "validation":
            raise ValueError("unknown mode or evaluation split")
        if spec.get("dataset") not in ("mnist", "cifar10") or spec.get("scenario") not in SCENARIOS:
            raise ValueError("unknown dataset/scenario")
        if (type(spec.get("seed")) is not int or not isinstance(spec.get("run_id"), str)
                or not spec["run_id"] or not isinstance(row["batch"], str) or not row["batch"]):
            raise ValueError("missing seed/run identity")
        for key in PROTOCOL_FIELDS:
            if key not in spec:
                raise ValueError(f"protocol field missing: {key}")
        duration, interval = spec["duration_seconds"], spec["evaluation_interval_seconds"]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
                   for v in (duration, interval)):
            raise ValueError("invalid duration/interval")
        if spec.get("client_count") != 8 or spec["tracked_classes"] != [3, 8]:
            raise ValueError("expected eight clients and tracked classes 3/8")
        if spec["group_r"] != [0, 4] or spec["group_c"] != [2, 6]:
            raise ValueError("unexpected R/C groups")
        if not 0 < spec["offline_start_fraction"] < spec["offline_end_fraction"] < 1:
            raise ValueError("invalid absence window")
        agenda = spec["scenario"].rsplit("-", 1)[1]
        if spec.get("offline_client_ids") != list(GROUPS[agenda]):
            raise ValueError("scenario and absent clients disagree")
    except READ_ERRORS as exc:
        row["issues"].append(str(exc))
        return row

    distributions = []
    for client in range(8):
        try:
            cost = inputs.json(directory / f"client_{client}_cost.json")
            _identity(cost, spec, client, "cost")
            if cost.get("state") != "completed":
                raise ValueError(f"client {client} did not complete")
            row["completed_clients"] += 1
        except READ_ERRORS as exc:
            row["issues"].append(f"client {client}: {exc}")
        try:
            distribution = inputs.json(directory / f"client_{client}_class_distribution.json")
            _identity(distribution, spec, client, "distribution")
            counts = distribution["counts"]
            if len(counts) != 10 or any(type(v) is not int or v < 0 for v in counts):
                raise ValueError("invalid class counts")
            for key in ("training_indices_sha256", "validation_indices_sha256", "local_indices_sha256"):
                if not isinstance(distribution.get(key), str) or not distribution[key]:
                    raise ValueError(f"missing data identity: {key}")
            distributions.append(distribution)
        except READ_ERRORS as exc:
            row["issues"].append(f"client {client}: {exc}")
        for kind in ("events", "metrics", "evaluations"):
            try:
                rows = inputs.lines(directory / f"client_{client}_{kind}.jsonl")
                for value in rows:
                    _identity(value, spec, client, kind)
                if not rows:
                    raise ValueError(f"empty {kind}")
                if kind == "evaluations":
                    diagnostic = {r.get("kind") for r in rows if r.get("kind") not in ("periodic", "final")}
                    if client in spec["offline_client_ids"]:
                        row["diagnostics"][str(client)] = {
                            "kinds": sorted(str(v) for v in diagnostic),
                            "three_stages_present": {"preserved_at_return", "before_first_mix", "after_first_mix"} <= diagnostic,
                        }
                if kind == "events":
                    offline = [r for r in rows if r.get("event") == "offline"]
                    returned = [r for r in rows if r.get("event") == "returned"]
                    expected_count = int(client in spec["offline_client_ids"])
                    if len(offline) != expected_count or len(returned) != expected_count:
                        raise ValueError("absence/return events disagree with scenario")
                    if expected_count and not offline[0]["elapsed_seconds"] < returned[0]["elapsed_seconds"]:
                        raise ValueError("return precedes absence")
            except READ_ERRORS as exc:
                row["issues"].append(f"client {client} {kind}: {exc}")
    if len(distributions) == 8:
        for key in ("training_indices_sha256", "validation_indices_sha256"):
            hashes = {d[key] for d in distributions}
            if len(hashes) != 1:
                row["issues"].append(f"clients disagree about {key}")
            row[key] = distributions[0][key]
        if max(sum(d["counts"]) for d in distributions) - min(sum(d["counts"]) for d in distributions) > 1:
            row["issues"].append("unbalanced local sample counts")
        row["_local_indices"] = [d["local_indices_sha256"] for d in distributions]
    try:
        # These existing helpers only read; their inputs are hashed above/below.
        clients = load_clients(directory, spec)
        groups = ({"online": sorted(set(range(8)) - set(spec["offline_client_ids"]))}
                  if spec["offline_client_ids"] else
                  {"without_r": [1, 2, 3, 5, 6, 7], "without_c": [0, 1, 3, 4, 5, 7]})
        leave, returned = duration * spec["offline_start_fraction"], duration * spec["offline_end_fraction"]
        for group, selected in groups.items():
            row["coverage"][group] = {
                window: integrated_outcome(clients, selected, start, end, interval)["coverage_fraction"]
                for window, start, end in (("absence", leave, returned), ("post_return", returned, duration))}
    except READ_ERRORS as exc:
        row["issues"].append(f"evaluations: {exc}")
    try:
        inputs.read(directory / "image.json")
        inputs.read(directory / "container_dependencies.txt")
        if not isinstance(spec.get("source_sha256"), dict) or not spec["source_sha256"]:
            raise ValueError("missing source hashes")
        if not spec.get("image_id"):
            raise ValueError("missing recorded image ID")
        row["_protocol"] = pairing_protocol(directory, spec)
        # JSON serialization distinguishes 600.0 from 600, although their budgets
        # are equal. Canonicalize numeric protocol values before fingerprinting.
        for key in PROTOCOL_FIELDS:
            value = row["_protocol"].get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value == int(value):
                row["_protocol"][key] = int(value)
        if not row["_protocol"].get("container_dependencies_sha256"):
            raise ValueError("missing container dependencies")
        row["protocol_sha256"] = json_sha256(row["_protocol"])
    except READ_ERRORS as exc:
        row["issues"].append(f"provenance: {exc}")
    row["_base_valid"] = not row["issues"]
    if row["_base_valid"]:
        full = all(v >= 1 - 1e-8 for group in row["coverage"].values() for v in group.values())
        row.update(status="completed" if full else "completed_partial_coverage",
                   historical_use="eligible_within_batch" if full else "eligible_only_for_covered_windows")
    return row


def _network(row: dict) -> dict:
    return {**row["_spec"], "pairing_protocol": row["_protocol"], "protocol_sha256": row["protocol_sha256"],
            "training_indices_sha256": row["training_indices_sha256"],
            "validation_indices_sha256": row["validation_indices_sha256"]}


def _mismatches(rows: list[dict]) -> list[str]:
    try:
        validate_paired_networks([_network(r) for r in rows], "audit", rows[0]["seed"])
    except READ_ERRORS as exc:
        return [str(exc)]
    partitions = {}
    for row in rows:
        distribution = row["scenario"].rsplit("-", 1)[0]
        if distribution in partitions and partitions[distribution] != row["_local_indices"]:
            return ["local training partitions disagree within the same distribution"]
        partitions[distribution] = row["_local_indices"]
    return []


def _full_coverage(row: dict) -> bool:
    return bool(row["coverage"]) and all(v >= 1 - 1e-8 for g in row["coverage"].values() for v in g.values())


IMAGE_FIELDS = ("image_filesystem_sha256", "image_config_sha256", "image_platform", "container_dependencies_sha256")


def _frozen_image_evidence(paths: list[Path], bundle: FrozenBundle, inputs: Inputs):
    """Anchor image contents to the exact preserved ID, regardless of method label.

    Compose may change an image ID solely by embedding its project/service labels.
    Only metadata of the other methods is read; their model evaluations are not used.
    """
    evidence, issues = {}, []
    listed = {str(name).replace("\\", "/").rsplit("/", 1)[-1]
              for name in bundle.manifest.get("confirmation_runs", [])}
    for path in paths:
        try:
            spec = inputs.json(path)
            dataset = spec.get("dataset")
            if (dataset not in ("mnist", "cifar10") or spec.get("mode") != "pilot"
                    or spec.get("evaluation_split") != "validation" or spec.get("run_id") not in listed
                    or spec.get("batch_run_id") != bundle.manifest.get("confirmation_id")
                    or spec.get("image_id") != bundle.manifest["images"][dataset]["id"]):
                continue
            methods = bundle.selection["datasets"][dataset]["methods"]
            if (spec.get("method") not in methods or spec.get("method_parameters") != methods[spec["method"]]
                    or spec.get("source_sha256") != bundle.manifest["source_sha256"]):
                continue
            if inputs.json(path.parent / "outcome.json").get("state") != "completed":
                continue
            inputs.read(path.parent / "image.json")
            inputs.read(path.parent / "container_dependencies.txt")
            protocol = pairing_protocol(path.parent, spec)
            fingerprint = {key: protocol[key] for key in IMAGE_FIELDS}
            if dataset not in evidence:
                evidence[dataset] = {"fingerprint": fingerprint, "directories": [], "consistent": True}
            if evidence[dataset]["fingerprint"] != fingerprint:
                evidence[dataset]["consistent"] = False
            evidence[dataset]["directories"].append(str(path.parent.resolve()))
        except READ_ERRORS as exc:
            # An unrelated malformed record is already represented by the inventory.
            # Failed image evidence cannot establish equivalence with the bundle.
            issues.append(f"image evidence {path}: {exc}")
    for dataset in ("mnist", "cifar10"):
        if dataset not in evidence or not evidence[dataset]["consistent"]:
            issues.append(f"{dataset}: no consistent inspection evidence for the preserved image ID")
    return evidence, issues


def _frozen_references(records: list[dict], bundle: FrozenBundle, inputs: Inputs, image_evidence):
    listed = {str(name).replace("\\", "/").rsplit("/", 1)[-1]
              for name in bundle.manifest.get("confirmation_runs", [])}
    references, issues = {}, []
    for dataset in ("mnist", "cifar10"):
        anchor = image_evidence.get(dataset)
        if anchor is None or not anchor["consistent"]:
            issues.append(f"{dataset}: frozen image contents cannot be verified")
            continue
        try:
            lock = inputs.read(bundle.directory / f"{dataset}_dependencies.lock.txt").decode("utf-8")
            dependencies = json_sha256(sorted(line.strip() for line in lock.splitlines() if line.strip()))
        except READ_ERRORS as exc:
            issues.append(f"{dataset}: frozen dependency reference unavailable: {exc}")
            continue
        for seed in (8, 9):
            matches = []
            for row in records:
                if not (row["_base_valid"] and row["dataset"] == dataset and row["seed"] == seed
                        and row["batch"] == bundle.manifest.get("confirmation_id")
                        and row["run_id"] in listed and row["scenario"] == "concentrated-r"):
                    continue
                expected = bundle.resolve_spec(dataset, "concentrated-r", seed, "pilot", "M0")
                fields = (*PROTOCOL_FIELDS, "method", "method_parameters", "training_age_unit", "execution_backend", "source_sha256")
                different = [key for key in fields if row["_spec"].get(key) != expected.get(key)]
                for key in IMAGE_FIELDS:
                    if row["_protocol"].get(key) != anchor["fingerprint"][key]:
                        different.append(key)
                if row["_protocol"]["container_dependencies_sha256"] != dependencies:
                    different.append("dependencies")
                if different:
                    issues.append(f"reference {row['directory']}: bundle mismatch: {', '.join(different)}")
                elif _full_coverage(row):
                    matches.append(row)
            if len(matches) == 1:
                references[dataset, seed] = matches[0]
            else:
                issues.append(f"{dataset}/seed={seed}: expected one verified frozen confirmation, found {len(matches)}")
    return references, issues


def _classify_frozen(records, references):
    for row in records:
        if not row["_base_valid"]:
            continue
        reference = references.get((row["dataset"], row["seed"]))
        if reference is None:
            row.update(frozen_use="not_verifiable", frozen_reasons=["no unique frozen reference for this dataset/seed"])
            continue
        differences = _mismatches([row, reference])
        if differences:
            row.update(frozen_use="incompatible", frozen_reasons=differences)
        else:
            row.update(frozen_use="compatible" if _full_coverage(row) else "partial_coverage",
                       frozen_reasons=[] if _full_coverage(row) else ["missing temporal coverage"])


def _comparison_rows(records):
    lookup = defaultdict(list)
    groups = set()
    for row in records:
        if (row["method"] == "M0" and row["mode"] == "pilot" and row["evaluation_split"] == "validation"
                and row["dataset"] in ("mnist", "cifar10") and type(row["seed"]) is int
                and isinstance(row["batch"], str) and row["scenario"] in SCENARIOS
                and row["status"] != "configuration_only"):
            key = (row["dataset"], row["batch"], row["seed"])
            groups.add(key)
            lookup[*key, row["scenario"]].append(row)
    contrasts = {}
    for distribution in ("distributed", "concentrated"):
        for agenda in ("r", "c"):
            contrasts[f"{distribution}_{agenda}_minus_none"] = [
                (f"{distribution}-{agenda}", "online"), (f"{distribution}-none", f"without_{agenda}")]
        contrasts[f"{distribution}_r_minus_c"] = [(f"{distribution}-r", "online"), (f"{distribution}-c", "online")]
    contrasts["concentration_x_absence"] = [(f"{d}-{a}", "online") for d in ("concentrated", "distributed") for a in ("r", "c")]
    comparisons = []
    for dataset, batch, seed in sorted(groups):
        for name, terms in contrasts.items():
            candidates = [lookup[dataset, batch, seed, scenario] for scenario, _ in terms]
            reasons = []
            if any(not group for group in candidates):
                reasons.append("missing situations: " + ", ".join(s for (s, _), g in zip(terms, candidates) if not g))
            if any(len(group) > 1 for group in candidates):
                reasons.append("duplicate situations; no execution selected automatically")
            networks = [group[0] for group in candidates if len(group) == 1]
            if any(not row["_base_valid"] for row in networks):
                reasons.append("incomplete/invalid execution; see inventory")
            if not reasons:
                reasons.extend(_mismatches(networks))
            statuses = {}
            for window in ("absence", "post_return"):
                blocked = list(reasons)
                if not blocked:
                    for row, (_, selection) in zip(networks, terms):
                        if row["coverage"].get(selection, {}).get(window, 0) < 1 - 1e-8:
                            blocked.append(f"incomplete {window} coverage: {row['scenario']}/{selection}")
                statuses[window] = ("blocked" if blocked else "available", blocked)
            selectors = [{"scenario": scenario, "selection": selection,
                          "client_ids": (sorted(set(range(8)) - set(GROUPS[scenario.rsplit('-', 1)[1]]))
                                         if selection == "online" else sorted(set(range(8)) - set(GROUPS[selection[-1]])))}
                         for scenario, selection in terms]
            comparisons.append({"dataset": dataset, "batch": batch, "seed": seed, "comparison": name,
                                "absence_status": statuses["absence"][0], "absence_reasons": statuses["absence"][1],
                                "post_return_status": statuses["post_return"][0], "post_return_reasons": statuses["post_return"][1],
                                "observers": selectors, "directories": [[r["directory"] for r in g] for g in candidates],
                                "composition_note": ("R/C comparisons use different six-client groups; availability is not a causal conclusion"
                                                     if name.endswith("r_minus_c") or name == "concentration_x_absence" else "same clients in both networks")})
    return comparisons


def _gaps(records, references, bundle):
    gaps = []
    for dataset in ("mnist", "cifar10"):
        for seed in (8, 9):
            for scenario in SCENARIOS:
                candidates = [r for r in records if r["dataset"] == dataset and r["seed"] == seed and r["scenario"] == scenario
                              and r["mode"] == "pilot" and r["evaluation_split"] == "validation"]
                compatible = [r for r in candidates if r["frozen_use"] == "compatible"]
                if len(compatible) == 1:
                    continue
                if (dataset, seed) not in references:
                    status, action = "review_reference", "review reference files before deciding a new execution"
                elif len(compatible) > 1:
                    status, action = "review_duplicates", "review duplicate executions; do not start another run automatically"
                else:
                    status, action = "missing_compatible_run", "candidate for E2 after technical checks; not started"
                base = bundle.resolve_spec(dataset, scenario, seed, "pilot", "M0")
                gaps.append({"dataset": dataset, "method": "M0", "scenario": scenario, "seed": seed,
                             "mode": "pilot", "evaluation_split": "validation", "status": status, "action": action,
                             "duration_seconds": base["duration_seconds"], "evaluation_interval_seconds": base["evaluation_interval_seconds"],
                             "expected_image_id": base["expected_image_id"],
                             "existing_directories": [r["directory"] for r in candidates],
                             "reasons": [{"directory": r["directory"], "status": r["status"], "frozen_use": r["frozen_use"],
                                          "details": r["frozen_reasons"] or r["issues"]} for r in candidates]})
    return gaps


def _write_csv(path: Path, rows: list[dict], fields):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items() if k in fields})


def audit_m0(results_root: Path, bundle_directory: Path, output_directory: Path) -> dict:
    """Inventory all records; individual read failures become explicit report rows."""
    root, destination = results_root.resolve(), output_directory.resolve()
    if not root.is_dir():
        raise ValueError(f"results root does not exist: {root}")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("audit output directory must be new or empty")
    paths = sorted(root.rglob("resolved_experiment.json"))
    if (destination == root or destination.is_relative_to(bundle_directory.resolve())
            or any(destination.is_relative_to(p.parent) for p in paths)):
        raise ValueError("audit output must be outside source runs and the frozen bundle")
    inputs = Inputs()
    for name in ("manifest.json", "selected_configuration.json", "source_snapshot.zip"):
        inputs.read(bundle_directory / name)
    bundle = FrozenBundle(bundle_directory)
    records, ignored = [], 0
    for path in paths:
        try:
            spec = inputs.json(path)
            if spec.get("method") not in (None, "M0"):
                ignored += 1
                continue
            records.append(_inspect_record(path, spec, inputs))
        except READ_ERRORS as exc:
            row = {key: "" for key in INVENTORY_FIELDS}
            row.update(directory=str(path.parent.resolve()), method="unknown", status="invalid_configuration",
                       issues=[str(exc)], frozen_use="not_eligible", frozen_reasons=[], historical_use="not_eligible",
                       _base_valid=False)
            records.append(row)
    image_evidence, image_issues = _frozen_image_evidence(paths, bundle, inputs)
    references, reference_issues = _frozen_references(records, bundle, inputs, image_evidence)
    _classify_frozen(records, references)
    comparisons = _comparison_rows(records)
    gaps = _gaps(records, references, bundle)
    repository = Path(__file__).resolve().parents[2]
    tool_hashes = {name: hashlib.sha256(inputs.read(repository / name)).hexdigest() for name in (
        "experiments/johann/audit.py", "experiments/johann/analysis.py", "experiments/johann/frozen.py",
        "experiments/johann/catalog.py", "scripts/run_johann_experiments.py")}
    inputs.verify_unchanged()
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "inventario_execucoes.csv", records, INVENTORY_FIELDS)
    _write_csv(destination / "comparacoes_disponiveis.csv", comparisons, (
        "dataset", "batch", "seed", "comparison", "absence_status", "absence_reasons", "post_return_status",
        "post_return_reasons", "observers", "directories", "composition_note"))
    _write_csv(destination / "lacunas_E2.csv", gaps, (
        "dataset", "method", "scenario", "seed", "mode", "evaluation_split", "status", "action",
        "duration_seconds", "evaluation_interval_seconds", "expected_image_id", "existing_directories", "reasons"))
    states = dict(Counter(r["status"] for r in records))
    audit = {"created_at": datetime.now(timezone.utc).isoformat(), "results_root": str(root),
             "frozen_bundle": str(bundle.directory), "inventory_records": len(records), "ignored_other_methods": ignored,
             "states": states, "frozen_references": {f"{ds}/s{seed}": r["directory"] for (ds, seed), r in references.items()},
             "reference_issues": reference_issues, "frozen_image_evidence": image_evidence, "image_evidence_issues": image_issues,
             "frozen_compatible_runs": sum(r["frozen_use"] == "compatible" for r in records),
             "e2_gap_count": len(gaps), "input_sha256": inputs.hashes, "missing_inputs": sorted(inputs.missing),
             "original_inputs_unchanged": True, "experiments_started": False, "official_evaluations_read": False,
             "generator_sha256": tool_hashes["experiments/johann/audit.py"],
             "python": sys.version, "tool_source_sha256": tool_hashes}
    with (destination / "auditoria.json").open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, ensure_ascii=False, allow_nan=False)
    lines = ["# E1 — Conferência dos resultados M0", "", "## O que já existe", "",
             f"Foram inventariados **{len(records)} registros** de M0 ou configuração não identificável. "
             f"Outros métodos foram separados ({ignored} configurações). Cada linha e motivo estão em `inventario_execucoes.csv`.", "",
             "| Situação dos arquivos | Quantidade |", "|---|---:|",
             *(f"| {STATUS_LABELS.get(state, state)} | {count} |" for state, count in sorted(states.items())),
             "", "`configuration_only` significa apenas configuração gerada; não conta como treino. "
             "Smokes servem para verificação funcional. Registros de teste oficial foram apenas identificados pelos metadados.",
             "", "## O que pode ser aproveitado", "",
             "Uma execução íntegra ainda precisa de um par compatível. As comparações ficam dentro do mesmo lote e semente. "
             "Código, imagem normalizada, dependências, protocolo e dados são conferidos. Metadados desconhecidos não são inventados.",
             "", "A imagem congelada é ancorada em uma inspeção registrada com seu ID exato. As imagens de outras "
             "execuções podem ter IDs distintos por rótulos do Compose; somente os rótulos de projeto, serviço e versão "
             "do Compose são desconsiderados. Conteúdo, configuração de execução, plataforma e dependências precisam coincidir. "
             "Eventuais registros de outros métodos usados nessa conferência fornecem apenas metadados de imagem, sem usar suas avaliações.",
             "", "| Dataset | Lote | Semente | Comparações disponíveis durante / após ausência |", "|---|---|---:|---:|",
             ]
    for ds, batch, seed in sorted({(r["dataset"], r["batch"], r["seed"]) for r in comparisons}):
        group = [r for r in comparisons if (r["dataset"], r["batch"], r["seed"]) == (ds, batch, seed)]
        lines.append(f"| {ds} | {batch} | {seed} | "
                     f"{sum(r['absence_status'] == 'available' for r in group)} / "
                     f"{sum(r['post_return_status'] == 'available' for r in group)} de 7 |")
    lines += ["", "Nos pares com ausência nenhuma, R usa os mesmos seis observadores {1,2,3,5,6,7}; "
              "C usa {0,1,3,4,5,7}. Os contrastes diretos R/C e o cruzamento usam grupos diferentes: "
              "sua disponibilidade não demonstra um efeito causal isolado da concentração.",
              "", f"**{audit['frozen_compatible_runs']} execuções** são compatíveis e completas em relação às referências "
              "congeladas. Resultados antigos podem ser aproveitados para descrever seus próprios pilotos sem, por isso, "
              "poderem completar os controles da versão congelada. Motivos individuais estão nas colunas `frozen_use` e `frozen_reasons`.",
              "", "## O que falta", "",
              f"A referência de E2 contém 24 situações: dois datasets × sementes 8/9 × seis cenários. "
              f"A lista tem **{len(gaps)} pendências**, sem executar nenhuma delas.", "",
              "| Dataset | Situações a executar ou revisar | Orçamento de treino das candidatas |", "|---|---:|---:|"]
    for ds in ("mnist", "cifar10"):
        group = [r for r in gaps if r["dataset"] == ds]
        seconds = sum(r["duration_seconds"] for r in group if r["status"] == "missing_compatible_run")
        lines.append(f"| {ds} | {len(group)} | {int(seconds // 3600)} h {int(seconds % 3600 // 60)} min |")
    lines += ["", "Os tempos não incluem inicialização, avaliações finais e reserva. `review_duplicates` e "
              "`review_reference` pedem conferência documental antes de decidir um novo treino. "
              "Não se escolheu a execução mais recente ou de melhor qualidade. Semente 7, outras cadências, "
              "smokes e teste oficial não substituem a referência de E2.",
              "", "## Como ler os arquivos", "",
              "- `inventario_execucoes.csv`: pastas, configurações, conclusão, cobertura, diagnósticos e motivos de uso.",
              "- `comparacoes_disponiveis.csv`: membros de cada comparação, observadores e impedimentos por janela.",
              "- `lacunas_E2.csv`: situações sem cobertura válida congelada ou com duplicatas/referências a revisar.",
              "- `auditoria.json`: hashes dos insumos lidos, referências e arquivos ausentes.",
              "", "Cobertura é a fração observada da janela, com a regra temporal já existente. "
              "Não foram produzidas novas estimativas de efeito, intervalos, avaliações de checkpoints ou gráficos estatísticos. "
              "Nenhum arquivo de origem foi reescrito, nenhum Docker foi chamado e nenhum treino foi iniciado."]
    if reference_issues or image_issues:
        lines += ["", "## Referências que exigem revisão", "", *(f"- {issue}" for issue in [*reference_issues, *image_issues])]
    with (destination / "relatorio.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return audit
