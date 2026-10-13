# Synthetic ECG fixtures

These files contain generated numeric signals, not patient data. Each case has
two locations, twelve training records and two test records per location, with
CSV diagnosis labels. The generator uses float16 arrays of shape `(12, 5000)`.
Values identify the location, split and row, with two distinctive entries to
test numeric fidelity and record association.

Cases: `contiguous`, `gzip`, `wrong_shape`, `nonfinite`, and `missing_record`.
Invalid cases change only the first test record of client1 and otherwise use
gzip. HDF5 is generated with `libver="earliest"`. Tests copy these files into
temporary directories; no HDF5 writer is needed to execute pytest.

To regenerate from the repository root in PowerShell, start Docker Desktop:

```powershell
docker run --rm -v "${PWD}:/workspace" -w /workspace python:3.11-slim-bookworm sh -c 'pip install numpy==2.1.3 h5py==3.16.0 && python scripts/generate_ecg_fixtures.py'
```

To also compare decoded shapes, dtypes and values with both readers, add
`pyfive==1.2.1` to that isolated install command and pass `--verify` to the
generator. This comparison includes NaN values in the invalid fixture.

The generator overwrites these synthetic fixtures. Do not run it against a
real dataset. h5py is solely a dependency of this isolated generation step;
it is absent from runtime and test requirements. Binary files need not be
byte-identical on regeneration; their decoded values and metadata must match.

These cases validate the reader and regression behavior, not compatibility or
performance for every real FedCVD HDF5 file.
