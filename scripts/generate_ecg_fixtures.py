"""Generate synthetic ECG fixtures in an isolated Linux environment.

Requires h5py==3.16.0 only for regeneration, never for running the test suite.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def generate(output: Path) -> None:
    import h5py
    import numpy as np

    cases = ("contiguous", "gzip", "wrong_shape", "nonfinite", "missing_record")
    for case in cases:
        for location_index in range(2):
            location = f"client{location_index + 1}"
            directory = output / case / "ECG" / "preprocessed" / location
            directory.mkdir(parents=True, exist_ok=True)
            with h5py.File(
                directory / "records.h5", "w", libver="earliest"
            ) as records:
                for split, count in (("train", 12), ("test", 2)):
                    rows = []
                    for row_index in range(count):
                        ecg_id = (
                            f"{location_index + 1}{0 if split == 'train' else 1}"
                            f"{row_index:04d}"
                        )
                        rows.append({
                            "ECG_ID": ecg_id,
                            "Code_Label": "0;1" if row_index % 2 == 0 else "2",
                            "Location": location,
                        })
                        special = (
                            location_index == 0
                            and split == "test"
                            and row_index == 0
                        )
                        if special and case == "missing_record":
                            continue
                        shape = (
                            (12, 4_999)
                            if special and case == "wrong_shape"
                            else (12, 5_000)
                        )
                        value = (
                            location_index * 100
                            + (20 if split == "test" else 0)
                            + row_index + 1
                        )
                        sample = np.full(shape, value, dtype=np.float16)
                        sample[0, 0] = -value
                        sample[1, 1] = value + 0.5
                        if special and case == "nonfinite":
                            sample[0, 0] = np.nan
                        options = (
                            {} if case == "contiguous" else {"compression": "gzip"}
                        )
                        records.create_dataset(ecg_id, data=sample, **options)
                    with (directory / f"{split}.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as stream:
                        writer = csv.DictWriter(
                            stream, fieldnames=("ECG_ID", "Code_Label", "Location")
                        )
                        writer.writeheader()
                        writer.writerows(rows)


def verify(output: Path) -> None:
    """Compare both readers in the isolated fixture maintenance environment."""
    import h5py
    import numpy as np
    import pyfive

    count = 0
    for path in sorted(output.rglob("records.h5")):
        with (
            h5py.File(path, "r") as expected,
            pyfive.File(str(path), "r") as actual,
        ):
            assert set(expected.keys()) == set(actual.keys()), path
            for key in expected.keys():
                reference = expected[key][...]
                result = actual[key][...]
                assert reference.dtype == result.dtype, (path, key)
                np.testing.assert_array_equal(reference, result)
                count += 1
    print(f"Both readers agree on {count} synthetic records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/ecg"))
    parser.add_argument(
        "--verify", action="store_true", help="compare generated data with pyfive"
    )
    args = parser.parse_args()
    generate(args.output)
    if args.verify:
        verify(args.output)
