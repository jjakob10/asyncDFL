"""Time-aligned outcomes with one independent observation per complete network."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from experiments.johann.catalog import GROUPS
from scripts.plot_metrics import load_experiments, plot_training_diagnostics


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pairing_protocol(directory: Path, spec: dict) -> dict:
    """Compare runtime contents, not Docker's scenario-specific image labels."""
    protocol = {key: spec.get(key) for key in (
        "method", "method_parameters", "training_age_unit", "execution_backend", "duration_seconds", "evaluation_interval_seconds", "evaluation_split",
        "tracked_classes", "group_r", "group_c", "edges", "offline_start_fraction", "offline_end_fraction",
        "validation_fraction", "concentration_fraction", "local_epochs", "batch_size", "learning_rate", "momentum",
        "source_sha256")}
    image_path = directory / "image.json"
    if image_path.is_file():
        images = read_json(image_path)
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
            raise ValueError(f"{image_path}: expected one inspected image")
        inspected = images[0]
        if not inspected.get("RootFS", {}).get("Layers") or not isinstance(inspected.get("Config"), dict):
            raise ValueError(f"{image_path}: missing image filesystem or configuration")
        if spec.get("image_id") and inspected.get("Id") != spec["image_id"]:
            raise ValueError(f"{image_path}: image ID does not match resolved experiment")
        config = dict(inspected["Config"])
        # Compose injects these metadata labels into otherwise identical builds.
        ignored_labels = {"com.docker.compose.project", "com.docker.compose.service", "com.docker.compose.version"}
        config["Labels"] = {key: value for key, value in (config.get("Labels") or {}).items()
                            if key not in ignored_labels}
        protocol.update(
            image_filesystem_sha256=json_sha256(inspected["RootFS"]),
            image_config_sha256=json_sha256(config),
            image_platform={key: inspected.get(key) for key in ("Architecture", "Os", "Variant")},
        )
    else:
        # Without an inspection artifact, only exact image IDs establish equality.
        protocol["image_id"] = spec.get("image_id")
    dependencies_path = directory / "container_dependencies.txt"
    protocol["container_dependencies_sha256"] = (
        json_sha256(sorted(line.strip() for line in dependencies_path.read_text(encoding="utf-8").splitlines()
                           if line.strip())) if dependencies_path.is_file() else None
    )
    return protocol


def validate_paired_networks(networks: list[dict], contrast: str, seed: int,
                            method_contrast: tuple[str, str] | None = None) -> None:
    """Name incompatible fields so a real protocol mismatch remains actionable."""
    fields = ("duration_seconds", "evaluation_interval_seconds", "evaluation_split",
              "training_indices_sha256", "validation_indices_sha256")
    mismatches = {key for key in fields if len({s.get(key) for s in networks}) > 1}
    protocols = [s.get("pairing_protocol", {}) for s in networks]
    intentional = set()
    if method_contrast is not None:
        allowed = {("M4", "M3"), *((method, "M0") for method in ("M1", "M2", "M3", "M4"))}
        if method_contrast not in allowed or tuple(s.get("method", "M0") for s in networks) != method_contrast:
            raise ValueError("unsupported or mismatched method contrast")
        if len({s.get("scenario") for s in networks}) != 1:
            mismatches.add("scenario")
        if method_contrast == ("M4", "M3"):
            k = [s.get("method_parameters", {}).get("rejoin_batches") for s in networks]
            if None in k or len(set(k)) != 1:
                mismatches.add("rejoin_batches")
        intentional = {"method", "method_parameters"}
    for key in set().union(*(p.keys() for p in protocols)):
        if key not in intentional and len({json_sha256(p.get(key)) for p in protocols}) > 1:
            mismatches.add(key)
    # Retain strict checking if a caller provides only precomputed fingerprints.
    if method_contrast is None and len({s.get("protocol_sha256") for s in networks}) > 1 and not mismatches:
        mismatches.add("protocol_sha256")
    if mismatches:
        runs = ", ".join(s["run_id"] for s in networks)
        raise ValueError(
            f"paired scenarios disagree: {', '.join(sorted(mismatches))}; "
            f"seed={seed}, contrast={contrast}, runs={runs}"
        )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_clients(directory, spec):
    clients = {}
    for client in range(8):
        path = directory / f"client_{client}_evaluations.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or any(r["client_id"] != client or r["run_id"] != spec["run_id"]
                           or r["evaluation_split"] != spec["evaluation_split"] for r in rows):
            raise ValueError(f"evaluation identity mismatch: {path}")
        usable = [r for r in rows if r["kind"] in {"periodic", "final"}]
        if not usable:
            raise ValueError(f"missing temporal evaluations for client {client}")
        for row in rows:
            if not np.isfinite(row["captured_seconds"]) or any(
                not np.isfinite(v) for v in row["test"].values()
            ):
                raise ValueError(f"non-finite evaluation in {path}")
        clients[client] = sorted(usable, key=lambda r: r["captured_seconds"])
    return clients


def observed_value(rows, at, metric, max_gap):
    """Right-continuous hold only between observations separated by <= max_gap."""
    times = [row["captured_seconds"] for row in rows]
    index = int(np.searchsorted(times, at, side="right")) - 1
    if index < 0 or index >= len(rows) - 1:
        # An exact last observation is valid as a point, never extrapolated.
        if index >= 0 and abs(at - times[index]) < 1e-9:
            return rows[index]["test"].get(metric)
        return None
    if times[index + 1] - times[index] > max_gap:
        return None
    test = rows[index]["test"]
    if metric == "recall_tracked" and (test.get("support_3", 0) == 0 or test.get("support_8", 0) == 0):
        return None
    return test.get(metric)


def integrated_outcome(clients, selected, start, end, interval, metric="recall_tracked"):
    grid = sorted({start, end, *(float(r["captured_seconds"]) for c in selected
                                 for r in clients[c] if start < r["captured_seconds"] < end)})
    area = coverage = 0.0
    for left, right in zip(grid, grid[1:]):
        values = [observed_value(clients[c], (left + right) / 2, metric, 2 * interval) for c in selected]
        if all(v is not None for v in values):
            area += float(np.mean(values)) * (right - left)
            coverage += right - left
    fraction = coverage / (end - start)
    complete = fraction >= 1 - 1e-8
    return {"post_return_auc": area / (end - start) if complete else None,
            "coverage_fraction": fraction, "complete": complete}


def recovery_time(times, values, baseline, returned, max_gap):
    if baseline is None:
        return None
    streak = []
    for at, value in zip(times, values):
        if at < returned:
            continue
        if value is None or value < baseline - .02:
            streak = []
            continue
        if streak and at - streak[-1] > max_gap:
            streak = []
        streak.append(at)
        if len(streak) >= 3:
            return streak[-3] - returned
    return None


def grouped_series(clients, selected, spec, metric):
    interval, duration = spec["evaluation_interval_seconds"], spec["duration_seconds"]
    times = np.unique(np.r_[np.arange(0, duration, interval), duration])
    values = []
    for at in times:
        samples = [observed_value(clients[c], at, metric, 2 * interval) for c in selected]
        values.append(float(np.mean(samples)) if all(v is not None for v in samples) else None)
    return times.tolist(), values


def _figure(path, draw, title, ylabel, spec=None):
    figure, axis = plt.subplots(figsize=(10, 5))
    draw(axis)
    if spec:
        duration = spec["duration_seconds"]
        axis.axvspan(duration * spec["offline_start_fraction"], duration * spec["offline_end_fraction"],
                     color="gray", alpha=.15, label="Janela de ausência (controle: todos online)")
        axis.set_xlabel("Tempo desde o início comum (s)")
        axis.set_xlim(0, duration)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=.25)
    handles, _ = axis.get_legend_handles_labels()
    if handles:
        axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def summarize_run(directory):
    directory = Path(directory)
    spec = read_json(directory / "resolved_experiment.json")
    clients = load_clients(directory, spec)
    interval, duration = spec["evaluation_interval_seconds"], spec["duration_seconds"]
    returned = duration * spec["offline_end_fraction"]
    leave = duration * spec["offline_start_fraction"]
    charts = directory / "charts"
    charts.mkdir(exist_ok=True)
    plot_training_diagnostics(load_experiments([directory])[0], charts)
    groups = {"R": list(GROUPS["r"]), "C": list(GROUPS["c"]), "demais": [1, 3, 5, 7]}
    if spec["offline_client_ids"]:
        groups["continuamente_online"] = sorted(set(range(8)) - set(spec["offline_client_ids"]))
        groups["retornantes"] = spec["offline_client_ids"]
    else:
        groups["controle_sem_R"] = sorted(set(range(8)) - set(GROUPS["r"]))
        groups["controle_sem_C"] = sorted(set(range(8)) - set(GROUPS["c"]))
    temporal = []
    for metric in ("recall_tracked", "recall_3", "recall_8", "recall_other", "macro_f1", "accuracy", "loss"):
        series = {name: grouped_series(clients, selected, spec, metric) for name, selected in groups.items()}
        for name, (times, values) in series.items():
            temporal.extend({"group": name, "metric": metric, "elapsed_seconds": at, "value": value}
                            for at, value in zip(times, values))
        def draw(axis):
            for name, (times, values) in series.items():
                axis.plot(times, [np.nan if v is None else v for v in values], label=name)
        _figure(charts / f"{metric}.png", draw, f"{spec['dataset']} — {spec['scenario']} — {metric}", metric, spec)
    def individual(axis):
        for client, rows in clients.items():
            axis.plot([r["captured_seconds"] for r in rows], [r["test"]["recall_tracked"] for r in rows],
                      label=f"Cliente {client}", alpha=.8)
    _figure(charts / "recall_clients.png", individual, "Recall das classes acompanhadas por cliente", "Recall", spec)
    counts = []
    distributions = []
    costs = []
    diagnostics = []
    for client in range(8):
        distribution = read_json(directory / f"client_{client}_class_distribution.json")
        cost = read_json(directory / f"client_{client}_cost.json")
        if any(value.get("run_id") != spec["run_id"] or value.get("client_id") != client
               for value in (distribution, cost)):
            raise ValueError("artifact identity mismatch")
        if cost["state"] != "completed":
            raise ValueError("cannot summarize a failed client as a complete network")
        distributions.append(distribution)
        costs.append(cost)
        counts.append(distribution["counts"])
        all_rows = [json.loads(line) for line in (directory / f"client_{client}_evaluations.jsonl").read_text().splitlines() if line]
        for row in all_rows:
            if row["kind"] not in {"periodic", "final"}:
                diagnostics.append({"client_id": client, "kind": row["kind"],
                                    "captured_seconds": row["captured_seconds"],
                                    "model_version": row["model_version"], **row["test"]})
    if len({d["training_indices_sha256"] for d in distributions}) != 1 or len({d["validation_indices_sha256"] for d in distributions}) != 1:
        raise ValueError("clients disagree about global train/validation selection")
    counts = np.array(counts)
    if counts.sum(1).max() - counts.sum(1).min() > 1:
        raise ValueError("client training sizes are not balanced")
    figure, axis = plt.subplots(figsize=(10, 5))
    heat = axis.imshow(counts, aspect="auto")
    axis.set_xticks(range(10), distributions[0]["labels"], rotation=40)
    axis.set_yticks(range(8), [f"Cliente {i}" for i in range(8)])
    axis.set_title("Exemplos de treinamento por classe e cliente")
    figure.colorbar(heat, ax=axis, label="Exemplos")
    figure.tight_layout()
    figure.savefig(charts / "class_distribution.png", dpi=150)
    plt.close(figure)
    def timeline(axis):
        for client in range(8):
            axis.broken_barh([(0, duration)], (client - .3, .6), facecolors="tab:green")
            if client in spec["offline_client_ids"]:
                axis.broken_barh([(leave, returned - leave)], (client - .3, .6), facecolors="tab:red")
            path = directory / f"client_{client}_events.jsonl"
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event["event"] in {"offline", "returned", "training_paused", "training_resumed"}:
                    axis.scatter(event["elapsed_seconds"], client, color="black", s=15, marker="|")
        axis.set_yticks(range(8))
    _figure(charts / "availability.png", timeline, "Disponibilidade e transições observadas", "Cliente", spec)
    outcomes = []
    selections = ({"online": groups["continuamente_online"]} if spec["offline_client_ids"] else
                  {"without_r": groups["controle_sem_R"], "without_c": groups["controle_sem_C"]})
    for name, selected in selections.items():
        outcome = integrated_outcome(clients, selected, returned, duration, interval)
        times, values = grouped_series(clients, selected, spec, "recall_tracked")
        baseline_samples = [observed_value(clients[c], leave - 1e-6, "recall_tracked", 2 * interval) for c in selected]
        baseline = float(np.mean(baseline_samples)) if all(v is not None for v in baseline_samples) else None
        recovery = recovery_time(times, values, baseline, returned, 2 * interval)
        outcomes.append({"selection": name, **outcome, "pre_departure_recall": baseline,
                         "recovery_seconds": recovery, "recovered": recovery is not None})
    # Include individual recovery, especially returnees; never drop non-recovering clients.
    individual_recovery = []
    for client in range(8):
        prior = [r for r in clients[client] if r["captured_seconds"] < leave]
        baseline = prior[-1]["test"]["recall_tracked"] if prior else None
        times = [r["captured_seconds"] for r in clients[client]]
        values = [r["test"]["recall_tracked"] for r in clients[client]]
        recovery = recovery_time(times, values, baseline, returned, 2 * interval)
        individual_recovery.append({"client_id": client, "returned": client in spec["offline_client_ids"],
                                    "recovery_seconds": recovery, "recovered": recovery is not None})
    summary = {"run_id": spec["run_id"], "dataset": spec["dataset"], "scenario": spec["scenario"],
               "method": spec.get("method", "M0"), "method_parameters": spec.get("method_parameters", {}),
               "seed": spec["seed"], "mode": spec["mode"], "duration_seconds": duration,
               "evaluation_interval_seconds": interval, "evaluation_split": spec["evaluation_split"],
               "training_indices_sha256": distributions[0]["training_indices_sha256"],
               "validation_indices_sha256": distributions[0]["validation_indices_sha256"],
               "outcomes": outcomes, "client_recovery": individual_recovery,
               "final_accuracy": float(np.mean([r[-1]["test"]["accuracy"] for r in clients.values()])),
               "model_bytes_received": sum(c["model_bytes_received"] for c in costs),
               "model_bytes_sent": sum(c["model_bytes_sent"] for c in costs),
               "optimizer_steps": sum(c["optimizer_steps"] for c in costs),
               "examples_processed": sum(c["examples_processed"] for c in costs),
               "train_seconds": sum(c["train_seconds"] for c in costs),
               "evaluation_seconds": sum(c["evaluation_seconds"] for c in costs),
               "communication_seconds": sum(c["communication_seconds"] for c in costs),
               "peak_process_memory_bytes": [c["peak_process_memory_bytes"] for c in costs],
               "actual_tracked_fraction_in_r": [float(counts[[0, 4], c].sum() / counts[:, c].sum()) for c in (3, 8)]}
    for key in ("reintegration_optimizer_steps", "reintegration_examples", "reintegration_train_seconds",
                "teacher_inferences", "teacher_examples", "teacher_seconds", "reintegration_wait_seconds",
                "discarded_optimizer_steps", "discarded_examples", "diagnostic_snapshot_bytes_peak", "teacher_model_bytes"):
        summary[key] = sum(c.get(key, 0) for c in costs)
    summary["reintegration_completed_clients"] = sum(bool(c.get("reintegration_completed")) for c in costs)
    protocol = pairing_protocol(directory, spec)
    summary["pairing_protocol"] = protocol
    summary["protocol_sha256"] = json_sha256(protocol)
    summary["image_id"] = spec.get("image_id")
    def outcomes_plot(axis):
        for i, item in enumerate(outcomes):
            if item["post_return_auc"] is not None:
                axis.bar(i, item["post_return_auc"])
            else:
                axis.text(i, .02, f"Incompleto\n{item['coverage_fraction']:.1%}", ha="center")
        axis.set_xticks(range(len(outcomes)), [r["selection"] for r in outcomes])
        axis.set_ylim(0, 1)
    _figure(charts / "post_return_outcome.png", outcomes_plot, "Desfecho da rede completa", "AUC normalizada")
    def recovery_plot(axis):
        for item in individual_recovery:
            recovered = item["recovered"]
            axis.scatter(item["client_id"], item["recovery_seconds"] if recovered else duration - returned,
                         marker="o" if recovered else "x", color="tab:blue" if recovered else "tab:red")
        axis.set_xlabel("Cliente; X indica recuperação não observada")
        axis.set_xticks(range(8))
    _figure(charts / "recovery.png", recovery_plot, "Recuperação por cliente (observações dependentes)", "Segundos após retorno")
    for cost_key in ("model_bytes_received", "optimizer_steps"):
        def cost_plot(axis):
            axis.scatter([c[cost_key] for c in costs], [clients[c][-1]["test"]["recall_tracked"] for c in range(8)])
            axis.set_xlabel(cost_key)
        _figure(charts / f"quality_vs_{cost_key}.png", cost_plot,
                "Clientes da mesma rede: comparação descritiva de custo", "Recall final acompanhado")
    write_json(directory / "summary.json", summary)
    write_csv(directory / "summary.csv", [{**{k: summary[k] for k in ("run_id", "dataset", "scenario", "seed", "mode")}, **r}
                                           for r in outcomes])
    write_csv(directory / "temporal_groups.csv", temporal)
    write_csv(directory / "diagnostics.csv", diagnostics)
    write_csv(directory / "client_recovery.csv", individual_recovery)
    lines = ["# Baseline M0", "", f"Dataset: {spec['dataset']}; cenário: {spec['scenario']}; semente: {spec['seed']}.",
             f"Avaliação: {spec['evaluation_split']}. Acurácia final média: {summary['final_accuracy']:.4f}.", "",
             "| Grupo | AUC pós-retorno | Cobertura | Recuperação (s) |", "|---|---:|---:|---:|"]
    for row in outcomes:
        lines.append(f"| {row['selection']} | {row['post_return_auc']} | {row['coverage_fraction']:.2%} | {row['recovery_seconds']} |")
    lines += ["", "Valores ausentes indicam observação incompleta ou recuperação não observada.",
              "Bytes representam somente payloads; tempos somados dos processos não são tempo de parede da rede.",
              "Oito clientes são observações dependentes de uma única execução."]
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def paired_bootstrap(differences, seed=12345, repetitions=10000):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(1)
    return tuple(float(x) for x in np.quantile(means, [.025, .975]))


def summarize_collection(root: Path, run_id: str | None = None, mode: str | None = None):
    root = Path(root)
    summaries = []
    failed = []
    for path in sorted(root.rglob("resolved_experiment.json")):
        spec = read_json(path)
        batch_id = spec.get("batch_run_id", spec["run_id"].rsplit("-s", 1)[0])
        if (run_id is not None and batch_id != run_id) or (mode is not None and spec["mode"] != mode):
            continue
        outcome_path = path.parent / "outcome.json"
        if not outcome_path.exists() or read_json(outcome_path)["state"] != "completed":
            failed.append({"directory": str(path.parent), "state": read_json(outcome_path)["state"] if outcome_path.exists() else "not_executed"})
            continue
        summaries.append(summarize_run(path.parent))
    write_json(root / "excluded_runs.json", failed)
    if not summaries:
        raise ValueError("no completed networks matched the collection filters")
    rows = []
    lookup = {}
    for summary in summaries:
        summary.setdefault("method", "M0")
        summary.setdefault("method_parameters", {})
        summary["method_configuration"] = json_sha256(summary["method_parameters"])
        key = (summary["dataset"], summary["mode"], summary["method"], summary["method_configuration"], summary["scenario"], summary["seed"])
        if key in lookup:
            raise ValueError(f"duplicate network repetition: {key}; choose a narrower collection root")
        lookup[key] = summary
        for outcome in summary["outcomes"]:
            rows.append({**{k: summary[k] for k in ("dataset", "mode", "method", "method_configuration", "scenario", "seed", "run_id")},
                         **outcome, "model_bytes_received": summary["model_bytes_received"],
                         "optimizer_steps": summary["optimizer_steps"], "train_seconds": summary["train_seconds"],
                         "evaluation_seconds": summary["evaluation_seconds"],
                         **{key: summary.get(key, 0) for key in ("reintegration_optimizer_steps", "reintegration_examples",
                            "teacher_inferences", "teacher_seconds", "reintegration_wait_seconds",
                            "discarded_optimizer_steps", "discarded_examples")}})
    write_csv(root / "collection.csv", rows)
    comparisons = []
    incompatible_method_pairs = []
    for dataset, mode, method, configuration in sorted({(s["dataset"], s["mode"], s["method"], s["method_configuration"]) for s in summaries}):
        contrasts = {"concentrated_r_minus_c": [("concentrated-r", "online", 1), ("concentrated-c", "online", -1)],
                     "distributed_r_minus_c": [("distributed-r", "online", 1), ("distributed-c", "online", -1)],
                     "concentration_x_absence": [("concentrated-r", "online", 1), ("concentrated-c", "online", -1),
                                                 ("distributed-r", "online", -1), ("distributed-c", "online", 1)]}
        for distribution in ("distributed", "concentrated"):
            for agenda in ("r", "c"):
                contrasts[f"{distribution}_{agenda}_minus_none"] = [
                    (f"{distribution}-{agenda}", "online", 1), (f"{distribution}-none", f"without_{agenda}", -1)]
        for name, terms in contrasts.items():
            differences, used_seeds = [], []
            candidates = sorted({s["seed"] for s in summaries if s["dataset"] == dataset and s["mode"] == mode
                                 and s["method"] == method and s["method_configuration"] == configuration})
            for seed in candidates:
                networks = [lookup.get((dataset, mode, method, configuration, scenario, seed)) for scenario, _, _ in terms]
                if any(s is None for s in networks):
                    continue
                validate_paired_networks(networks, name, seed)
                values = [next(r["post_return_auc"] for r in network["outcomes"] if r["selection"] == selection)
                          for network, (_, selection, _) in zip(networks, terms)]
                if any(v is None for v in values):
                    continue
                differences.append(sum(v * term[2] for v, term in zip(values, terms)))
                used_seeds.append(seed)
            low, high = paired_bootstrap(differences)
            comparisons.append({"dataset": dataset, "mode": mode, "method": method, "method_configuration": configuration,
                                "contrast": name, "paired_networks": len(differences),
                                "seeds": ",".join(map(str, used_seeds)),
                                "mean_difference": float(np.mean(differences)) if differences else None,
                                "ci95_low": low, "ci95_high": high})
    # Distinct configurations are never pooled. Each configured method is paired
    # against the baseline, and M4 against M3 with an identical student budget.
    for left_method, right_method in [(m, "M0") for m in ("M1", "M2", "M3", "M4")] + [("M4", "M3")]:
        groups = sorted({(s["dataset"], s["mode"], s["scenario"]) for s in summaries
                         if s["method"] == left_method})
        for dataset, mode, scenario in groups:
            if (left_method, right_method) == ("M4", "M3") and scenario != "concentrated-r":
                continue
            configs = {m: sorted({s["method_configuration"] for s in summaries if s["dataset"] == dataset
                                  and s["mode"] == mode and s["scenario"] == scenario and s["method"] == m})
                       for m in (left_method, right_method)}
            for left_config in configs[left_method]:
                for right_config in configs[right_method]:
                    differences, used_seeds = [], []
                    candidates = sorted({s["seed"] for s in summaries if s["dataset"] == dataset and s["mode"] == mode})
                    for seed in candidates:
                        networks = [lookup.get((dataset, mode, m, c, scenario, seed)) for m, c in (
                            (left_method, left_config), (right_method, right_config))]
                        if any(s is None for s in networks):
                            continue
                        name = f"{left_method}_minus_{right_method}/{scenario}"
                        if (left_method, right_method) == ("M4", "M3") and (
                            networks[0]["method_parameters"].get("rejoin_batches") !=
                            networks[1]["method_parameters"].get("rejoin_batches")
                        ):
                            incompatible_method_pairs.append({"dataset": dataset, "mode": mode,
                                "scenario": scenario, "seed": seed, "left_configuration": left_config,
                                "right_configuration": right_config, "reason": "rejoin_batches_mismatch"})
                            continue
                        validate_paired_networks(networks, name, seed, (left_method, right_method))
                        selection = "without_r" if scenario.endswith("none") else "online"
                        values = [next(r["post_return_auc"] for r in s["outcomes"] if r["selection"] == selection) for s in networks]
                        if any(v is None for v in values):
                            continue
                        differences.append(values[0] - values[1])
                        used_seeds.append(seed)
                    low, high = paired_bootstrap(differences)
                    comparisons.append({"dataset": dataset, "mode": mode, "method": f"{left_method}-{right_method}",
                                        "method_configuration": f"{left_config}/{right_config}",
                                        "contrast": f"{left_method}_minus_{right_method}/{scenario}",
                                        "paired_networks": len(differences), "seeds": ",".join(map(str, used_seeds)),
                                        "mean_difference": float(np.mean(differences)) if differences else None,
                                        "ci95_low": low, "ci95_high": high})
    write_csv(root / "paired_comparisons.csv", comparisons)
    write_json(root / "incompatible_method_pairs.json", incompatible_method_pairs)
    def outcome_plot(axis):
        for dataset, mode in sorted({(r["dataset"], r["mode"]) for r in rows}):
            eligible = [r for r in rows if r["dataset"] == dataset and r["mode"] == mode and r["post_return_auc"] is not None]
            axis.scatter([f"{dataset}\n{mode}/{r['method']}/{r['method_configuration'][:6]}\n{r['scenario']}\n{r['selection']}" for r in eligible],
                         [r["post_return_auc"] for r in eligible], label=f"{dataset}/{mode}")
        axis.tick_params(axis="x", labelsize=6)
    _figure(root / "post_return_outcomes.png", outcome_plot, "Desfecho por execução completa", "AUC normalizada")
    for cost in ("model_bytes_received", "optimizer_steps", "train_seconds", "teacher_seconds"):
        def draw(axis):
            for method in sorted({row["method"] for row in rows}):
                selected = [row for row in rows if row["method"] == method and row["post_return_auc"] is not None]
                axis.scatter([row[cost] for row in selected], [row["post_return_auc"] for row in selected], label=method)
            axis.set_xlabel(cost)
        _figure(root / f"quality_vs_{cost}.png", draw, "Qualidade versus custo", "AUC normalizada")
    valid = [r for r in comparisons if r["mean_difference"] is not None]
    if valid:
        figure, axis = plt.subplots(figsize=(11, 6))
        for i, row in enumerate(valid):
            mean = row["mean_difference"]
            if row["ci95_low"] is not None:
                axis.plot([row["ci95_low"], row["ci95_high"]], [i, i], color="tab:blue")
            axis.scatter(mean, i, color="tab:blue")
        axis.set_yticks(range(len(valid)), [f"{r['dataset']}/{r['mode']}/{r['method']}/{r['method_configuration'][:6]}/{r['contrast']} (n={r['paired_networks']})" for r in valid], fontsize=7)
        axis.axvline(0, color="gray", linestyle="--")
        axis.set_xlabel("Diferença pareada de AUC; intervalo bootstrap de 95%")
        figure.tight_layout()
        figure.savefig(root / "paired_contrasts.png", dpi=150)
        plt.close(figure)
    (root / "collection.md").write_text(
        f"# Coleção Johann M0–M4\n\n{len(summaries)} redes completas; {len(failed)} execuções não concluídas ou não executadas.\n\n"
        "Contrastes em paired_comparisons.csv usam diferenças pareadas por semente. "
        "IC ausente com menos de duas redes; pilotos e coleta são analisados separadamente. "
        "Desfechos incompletos permanecem na tabela, mas não alimentam contrastes.\n", encoding="utf-8")
    return comparisons
