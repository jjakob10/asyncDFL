"""Read preserved Johann inputs without importing or extracting archived code."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from experiments.johann.catalog import load_spec


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _object(value, description: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"frozen bundle: {description} must be an object")
    return value


class FrozenBundle:
    """Validated, in-memory copy of the bundle used throughout one invocation."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.manifest_bytes = (self.directory / "manifest.json").read_bytes()
        self.selection_bytes = (self.directory / "selected_configuration.json").read_bytes()
        self.manifest = _object(json.loads(self.manifest_bytes), "manifest")
        self.selection = _object(json.loads(self.selection_bytes), "selection")
        if self.manifest.get("protocol_frozen") is not True:
            raise ValueError("frozen bundle: manifest is not frozen")
        if not self.manifest.get("calibration_id") or self.manifest["calibration_id"] != self.selection.get("calibration_id"):
            raise ValueError("frozen bundle: calibration IDs disagree")
        datasets = _object(self.manifest.get("datasets"), "manifest datasets")
        selected = _object(self.selection.get("datasets"), "selected datasets")
        images = _object(self.manifest.get("images"), "images")
        sources = _object(self.manifest.get("source_sha256"), "source hashes")
        if not datasets or set(datasets) != set(selected) or set(datasets) != set(images):
            raise ValueError("frozen bundle: dataset selections disagree")
        archive_bytes = (self.directory / "source_snapshot.zip").read_bytes()
        if sha256(archive_bytes) != self.manifest.get("source_archive_sha256"):
            raise ValueError("frozen bundle: source archive SHA-256 mismatch")
        self.dataset_manifests = {}
        try:
            with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ValueError("frozen bundle: duplicate archive entries")
                for name, digest in sources.items():
                    if sha256(archive.read(name)) != digest:
                        raise ValueError(f"frozen bundle: source hash mismatch: {name}")
                for dataset in datasets:
                    if dataset not in {"mnist", "cifar10"}:
                        raise ValueError(f"frozen bundle: unsupported dataset: {dataset}")
                    name = f"experiments/johann/{dataset}/manifest.json"
                    if name not in sources:
                        raise ValueError(f"frozen bundle: missing source hash: {name}")
                    self.dataset_manifests[dataset] = _object(json.loads(archive.read(name)), name)
        except (zipfile.BadZipFile, KeyError) as exc:
            raise ValueError(f"frozen bundle: invalid source archive: {exc}") from exc
        required = {"M0": set(), "M1": set(), "M2": {"merge_limit_examples"},
                    "M3": {"rejoin_batches"},
                    "M4": {"rejoin_batches", "distillation_weight", "distillation_temperature"}}
        for dataset in datasets:
            choice = _object(selected[dataset], f"selection for {dataset}")
            preserved = _object(datasets[dataset], f"manifest for {dataset}")
            image = _object(images[dataset], f"image for {dataset}")
            if choice.get("protocol_frozen") is not True:
                raise ValueError(f"frozen bundle: selection for {dataset} is not frozen")
            # Older bundles retain protocol_frozen=false in manifest.datasets.
            # The global manifest and final selection are the authoritative flags.
            if (image != choice.get("frozen_image") or not isinstance(image.get("id"), str)
                    or not re.fullmatch(r"sha256:[0-9a-f]{64}", image["id"])):
                raise ValueError(f"frozen bundle: image IDs disagree or are incomplete: {dataset}")
            methods = _object(choice.get("methods"), f"methods for {dataset}")
            if not methods or methods != preserved.get("methods"):
                raise ValueError(f"frozen bundle: method parameters disagree: {dataset}")
            for method, parameters in methods.items():
                if method not in required or not isinstance(parameters, dict) or set(parameters) != required[method]:
                    raise ValueError(f"frozen bundle: incomplete parameters: {dataset}/{method}")
                # Reuse the catalog's numeric validation without workspace defaults.
                self.resolve_spec(dataset, "concentrated-r", 7, "pilot", method)

    def resolve_spec(self, dataset: str, scenario: str, seed: int, mode: str,
                     method: str, duration=None, interval=None) -> dict:
        if dataset not in self.dataset_manifests:
            raise ValueError(f"frozen bundle: dataset not available: {dataset}")
        methods = self.selection["datasets"][dataset]["methods"]
        if method not in methods:
            raise ValueError(f"frozen bundle: method not available: {dataset}/{method}")
        if mode != "smoke" and (duration is not None or interval is not None):
            raise ValueError("frozen bundle: duration/interval overrides are only allowed for smoke")
        base = deepcopy(self.dataset_manifests[dataset])
        base["methods"] = deepcopy(methods)
        try:
            spec = load_spec(dataset, scenario, seed, duration, interval, mode, method,
                             methods[method], base_manifest=base)
        except TypeError as exc:
            raise ValueError(f"frozen bundle: invalid manifest values for {dataset}: {exc}") from exc
        image = self.manifest["images"][dataset]
        spec.update(protocol_frozen=mode != "smoke", calibration_id=self.selection["calibration_id"],
                    calibration_confirmation=mode == "pilot" and seed in (8, 9),
                    source_sha256=deepcopy(self.manifest["source_sha256"]),
                    expected_image_id=image["id"],
                    frozen_bundle={"directory": str(self.directory),
                                   "manifest_sha256": sha256(self.manifest_bytes),
                                   "selection_sha256": sha256(self.selection_bytes),
                                   "source_archive_sha256": self.manifest["source_archive_sha256"],
                                   "image_tag": image.get("tag"),
                                   "scientific_protocol_complete": mode != "smoke"})
        return spec
