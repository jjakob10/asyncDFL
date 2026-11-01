"""Prepare nonoverlapping HARTH windows once; clients read memory-mapped arrays."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.collection import read_json

CLASS_CODES = (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140)
CLASS_NAMES = ("walking", "running", "shuffling", "stairs_ascending", "stairs_descending",
               "standing", "sitting", "lying", "cycling_sit", "cycling_stand",
               "cycling_sit_inactive", "cycling_stand_inactive")
CHANNELS = ("back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z")
WINDOW_LENGTH = 250
SAMPLE_PERIOD_NS = 20_000_000
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/779/harth.zip"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download(root: Path) -> Path:
    """Download the official archive; copy only participant CSV basenames."""
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = root / "harth.zip"
    if not archive.is_file():
        temporary = archive.with_suffix(".part")
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        names = [name for name in zipped.namelist()
                 if re.fullmatch(r"S\d+\.csv", Path(name).name)]
        if len(names) != 22 or len({Path(name).name for name in names}) != 22:
            raise ValueError("HARTH archive must contain 22 distinct participant CSVs")
        for name in names:
            destination = raw / Path(name).name
            temporary = destination.with_suffix(".part")
            with zipped.open(name) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
    return raw


def participant_windows(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"timestamp", "label", *CHANNELS}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: missing columns {sorted(required - set(frame.columns))}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise").astype("int64").to_numpy()
    values = frame[list(CHANNELS)].to_numpy(dtype=np.float32)
    raw_labels = pd.to_numeric(frame["label"], errors="raise").to_numpy()
    if (not len(frame) or not np.isfinite(values).all() or pd.isna(frame["timestamp"]).any()
            or not np.isin(raw_labels, CLASS_CODES).all()):
        raise ValueError(f"{path}: empty data, nonfinite inputs, invalid timestamp or class")
    differences = np.diff(timestamps)
    if np.any(differences <= 0):
        raise ValueError(f"{path}: timestamps must increase strictly")
    # A cadence deviation larger than 50% splits the recording; windows never bridge it.
    boundaries = np.r_[0, np.flatnonzero(np.abs(differences - SAMPLE_PERIOD_NS)
                                        > SAMPLE_PERIOD_NS // 2) + 1, len(frame)]
    windows, labels, starts, rows = [], [], [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        for offset in range(int(left), int(right) - WINDOW_LENGTH + 1, WINDOW_LENGTH):
            segment = raw_labels[offset:offset + WINDOW_LENGTH]
            counts = np.array([np.count_nonzero(segment == code) for code in CLASS_CODES])
            labels.append(CLASS_CODES[int(counts.argmax())])
            windows.append(values[offset:offset + WINDOW_LENGTH].T)
            starts.append(timestamps[offset])
            rows.append(offset)
    return (np.asarray(windows, dtype=np.float32).reshape(-1, 6, WINDOW_LENGTH),
            np.asarray(labels, dtype=np.int64), np.asarray(starts, dtype=np.int64),
            np.asarray(rows, dtype=np.int64))


def prepare(raw: Path, output: Path) -> dict:
    paths = sorted(path for path in raw.glob("S*.csv") if re.fullmatch(r"S\d+", path.stem))
    if not paths:
        raise ValueError(f"no participant CSVs in {raw}")
    output.mkdir(parents=True, exist_ok=True)
    participants = {}
    for path in paths:
        arrays = participant_windows(path)
        if not len(arrays[0]):
            raise ValueError(f"{path}: no complete windows")
        files = {}
        for kind, array in zip(("x", "labels", "timestamps", "rows"), arrays):
            destination = output / f"{path.stem}_{kind}.npy"
            temporary = destination.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(destination)
            files[kind] = {"name": destination.name, "sha256": sha256_file(destination)}
        participants[path.stem] = {
            "window_count": len(arrays[0]), "raw_sha256": sha256_file(path), "files": files}
    metadata = {
        "version": 1, "source": "https://archive.ics.uci.edu/dataset/779/harth",
        "download_url": DOWNLOAD_URL, "license": "CC BY 4.0",
        "window_length": WINDOW_LENGTH, "stride": WINDOW_LENGTH,
        "sample_period_ns": SAMPLE_PERIOD_NS, "gap_tolerance_ns": SAMPLE_PERIOD_NS // 2,
        "majority_tie": "smallest_original_code", "channels": list(CHANNELS),
        "class_codes": list(CLASS_CODES), "class_names": list(CLASS_NAMES),
        "participants": participants,
    }
    temporary = output / "metadata.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(output / "metadata.json")
    return metadata


def read_metadata(root: Path, verify: bool = False) -> dict:
    metadata = read_json(root / "metadata.json")
    if (metadata.get("version") != 1 or metadata.get("window_length") != WINDOW_LENGTH
            or metadata.get("class_codes") != list(CLASS_CODES)):
        raise ValueError("unsupported HARTH preparation metadata")
    if verify:
        for participant in metadata["participants"].values():
            for file in participant["files"].values():
                if sha256_file(root / file["name"]) != file["sha256"]:
                    raise ValueError(f"HARTH prepared file hash mismatch: {file['name']}")
    return metadata


def split_participants(metadata: dict) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    subjects = sorted(metadata["participants"])
    if len(subjects) != 22:
        raise ValueError("HARTH examples require exactly 22 participants")
    ordered = np.random.default_rng(42).permutation(subjects).tolist()
    return tuple(ordered[:4]), tuple(tuple(ordered[4 + 3*i:7 + 3*i]) for i in range(6))


class HarthWindows(Dataset):
    def __init__(self, root: Path, subjects: tuple[str, ...], metadata: dict,
                 limit: int = 0) -> None:
        self.arrays = []
        entries = []
        for subject in subjects:
            files = metadata["participants"][subject]["files"]
            arrays = {key: np.load(root / value["name"], mmap_mode="r", allow_pickle=False)
                      for key, value in files.items()}
            count = metadata["participants"][subject]["window_count"]
            if (arrays["x"].shape != (count, 6, WINDOW_LENGTH)
                    or any(arrays[key].shape != (count,) for key in ("labels", "timestamps", "rows"))):
                raise ValueError(f"HARTH prepared array shape mismatch: {subject}")
            slot = len(self.arrays)
            self.arrays.append(arrays)
            entries.extend((int(t), subject, slot, i) for i, t in enumerate(arrays["timestamps"]))
        entries.sort()
        self.entries = entries[:limit] if limit else entries
        self.labels = np.asarray([self.arrays[slot]["labels"][i] for _, _, slot, i in self.entries],
                                 dtype=np.int64)
        self.sample_ids = tuple(f"harth:{subject}:row:{int(self.arrays[slot]['rows'][i])}"
                                for _, subject, slot, i in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        _, _, slot, offset = self.entries[index]
        inputs = torch.from_numpy(self.arrays[slot]["x"][offset].copy())
        return inputs, CLASS_CODES.index(int(self.labels[index]))
