from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from app.config import Settings
from experiments.harth.data import (
    CHANNELS, CLASS_CODES, HarthWindows, participant_windows, prepare,
    read_metadata, split_participants,
)
from experiments.harth import trainer


def write_subject(path, labels, timestamps=None):
    count = len(labels)
    if timestamps is None:
        timestamps = pd.date_range("2026-01-01", periods=count, freq="20ms")
    frame = pd.DataFrame({"timestamp": timestamps, "label": labels,
                          **{name: np.arange(count, dtype=np.float32) / 1000 for name in CHANNELS}})
    frame.to_csv(path, index=False)


@pytest.fixture
def prepared_harth(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for index in range(22):
        write_subject(raw / f"S{index+1:03}.csv",
                      np.repeat([CLASS_CODES[index % 12], CLASS_CODES[(index+1) % 12], 1], 250))
    root = tmp_path / "harth" / "prepared"
    metadata = prepare(raw, root)
    return tmp_path, root, metadata


def test_windows_use_six_channels_majority_and_smallest_code_tie(tmp_path):
    path = tmp_path / "S001.csv"
    write_subject(path, np.r_[[14] * 125, [1] * 125, [130] * 250, [7] * 17])
    x, labels, timestamps, rows = participant_windows(path)
    assert x.shape == (2, 6, 250)
    assert x.dtype == np.float32
    assert labels.tolist() == [1, 130]
    assert rows.tolist() == [0, 250]
    assert timestamps[1] - timestamps[0] == 5_000_000_000
    assert x[0, 0, -1] == pytest.approx(.249)


def test_windows_do_not_bridge_gaps_or_incomplete_segments(tmp_path):
    path = tmp_path / "S001.csv"
    times = pd.date_range("2026-01-01", periods=600, freq="20ms").to_numpy()
    times[200:] += np.timedelta64(5, "s")
    write_subject(path, np.ones(600), times)
    x, _, _, rows = participant_windows(path)
    assert len(x) == 1
    assert rows.tolist() == [200]


@pytest.mark.parametrize("error", ["nonfinite", "label", "time", "missing"])
def test_rejects_invalid_raw_data(tmp_path, error):
    path = tmp_path / "S001.csv"
    write_subject(path, np.ones(250))
    frame = pd.read_csv(path)
    if error == "nonfinite":
        frame.loc[0, "back_x"] = np.inf
    elif error == "label":
        frame.loc[0, "label"] = 99
    elif error == "time":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    else:
        frame.drop(columns=["thigh_x"], inplace=True)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        participant_windows(path)


def test_preparation_hashes_mmap_order_and_disjoint_subjects(prepared_harth):
    _, root, metadata = prepared_harth
    assert read_metadata(root, verify=True) == metadata
    test, groups = split_participants(metadata)
    assert len(test) == 4 and all(len(group) == 3 for group in groups)
    assert len(set(test).union(*map(set, groups))) == 22
    assert not set(test).intersection(set().union(*map(set, groups)))
    assert (test, groups) == split_participants(metadata)
    seen = set()
    for group in groups:
        dataset = HarthWindows(root, group, metadata)
        assert isinstance(dataset.arrays[0]["x"], np.memmap)
        assert dataset.entries == sorted(dataset.entries)
        assert not seen.intersection(dataset.sample_ids)
        seen.update(dataset.sample_ids)
        inputs, target = dataset[0]
        assert inputs.shape == (6, 250) and 0 <= target < 12
    file = root / metadata["participants"][test[0]]["files"]["x"]["name"]
    with file.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_metadata(root, verify=True)


def test_harth_trainer_learning_evaluation_and_original_codes(prepared_harth):
    data_root, _, _ = prepared_harth
    settings = Settings(client_id=0, client_name="client-0", client_count=6,
                        head_client_ids=(5,), data_dir=str(data_root), batch_size=4)
    torch.set_num_threads(1)
    data = trainer.load_data(settings)
    assert data.collection_source.chronological
    assert data.collection_source.class_codes == CLASS_CODES
    assert len(data.train_loader.dataset) == 9
    assert len(data.test_loader.dataset) == 12
    initial = trainer.create_model().state_dict()
    state, metrics = trainer.train_local(initial, data.train_loader, settings)
    assert np.isfinite(metrics["loss"])
    assert any(not torch.equal(state[name], initial[name]) for name in state)
    evaluated = trainer.evaluate(state, data.test_loader)
    assert 0 <= evaluated["accuracy"] <= 1
    confusion = trainer.evaluate_confusion(state, data.test_loader)
    assert np.asarray(confusion["counts"]).shape == (12, 12)
    assert np.asarray(confusion["counts"]).sum() == 12
    assert not set(data.collection_source.sample_ids).intersection(data.test_loader.dataset.sample_ids)
    limited = trainer.load_data(replace(settings, max_train_samples=2, max_test_samples=3))
    assert len(limited.train_loader.dataset) == 2
    assert len(limited.test_loader.dataset) == 3
