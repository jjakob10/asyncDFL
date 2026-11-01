# Decentralized Federated Learning Experiments

FastAPI/PyTorch clients train and exchange CPU model snapshots directly over
HTTP. There is no central aggregation server and no global round barrier.
Each client trains, reads available neighbors, aggregates compatible states,
evaluates and publishes a new model version.

The repository contains these experiment families:

| Family | Dataset / purpose |
|---|---|
| `mnist` | Image CNN; standalone rings and stars with 4 or 8 clients. |
| `cifar10` | Image CNN; standalone rings and stars with 4 or 8 clients. |
| `harth` | Temporal CNN for activity windows, with a subject-held-out test set. |
| `johann` | Timed MNIST/CIFAR-10 experiments, availability schedules and methods M0–M4. |
| `thiago` | Twelve fixed-data MNIST/CIFAR-10 experiments with 6 clients. |
| `thiago-coleta-simulada` | Six accumulation/sliding-window examples, 6-client rings. |

## Repository layout

```text
app/                         API, settings, runtime, federation, collection and metrics
experiments/
  mnist/, cifar10/, harth/    Dataset/model trainers and optional preparation
  johann/                    Timed protocol, calibration, analysis and manifests
  thiago/                    Fixed-data manifests and PowerShell entry point
  thiago-coleta-simulada/     Collection manifests, agendas, Dockerfile and documentation
scripts/
  experiment_tools/          Shared catalog, topology, Compose, execution and reports
  run_experiments.py         Fixed-data command; explicit --manifest required
  run_collection_experiments.py  Separate collection command
  run_johann_experiments.py  Timed Johann command
  export_manifest.py         Standalone fixed-data Compose/environment export
  plot_metrics.py            Curves, comparisons, distributions and confusion matrices
  wait_for_run.py            Client status monitoring
results/                     Ignored datasets, execution artifacts and checkpoints
outputs/                     Presentation/research deliverables
tests/                      Unit and regression tests
```

Experiment command modules are entry points. Reusable functions belong in
`scripts.experiment_tools`; importing another command to reuse a helper is
unnecessary. The application contains no catalog or experiment-family defaults.

## Setup

Use Python 3.11-compatible code. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For the commands below, activate the environment or use its Python executable.
Docker Desktop must provide a running Linux engine for container runs.
Requirements use CPU-only PyTorch. No dataset or model checkpoint is versioned.

## Standalone MNIST and CIFAR-10

The default Compose variants use Dirichlet partitions. Files ending in
`.iid.yaml` use IID partitions. Both datasets provide:

| Topology | Clients | Compose filename |
|---|---:|---|
| Ring | 4 | `compose.ring.4.yaml` / `compose.ring.4.iid.yaml` |
| Ring | 8 | `compose.ring.8.yaml` / `compose.ring.8.iid.yaml` |
| Star | 4 | `compose.star.4.yaml` / `compose.star.4.iid.yaml` |
| Star | 8 | `compose.star.8.yaml` / `compose.star.8.iid.yaml` |

Build, start and remove a network, substituting `cifar10` when needed:

```powershell
docker compose -f experiments/mnist/compose.ring.8.yaml build
docker compose -f experiments/mnist/compose.ring.8.yaml up -d
python scripts/wait_for_run.py --client-count 8
docker compose -f experiments/mnist/compose.ring.8.yaml down
```

The images prepare train and test data during construction. The runtime reads
those prepared datasets without downloading. Copy the dataset's `.env.example`
to `.env` to override rounds, epochs, optimizer, sample caps, seed, request
behavior and partition settings. An explicit `--env-file` is also supported.
Use one round and small caps for smoke tests; zero caps mean the full dataset.

The internal API port is 9000. Host ports are 8000 plus client ID. Ring clients
consult their predecessor and successor, including the wraparound connection.
In a star, client 0 consults the leaves and every leaf consults client 0.
The designated head propagates the initial start; it does not act as a server
for aggregation. Model snapshots are published atomically under a lock.

## Thiago fixed-data experiments

The two manifests each define six experiments: ring/star crossed with
IID, Dirichlet and manual class partitions. Every experiment has six clients,
ten rounds and one local epoch. Defaults and IDs remain in the manifests.

```powershell
python scripts/run_experiments.py list --manifest experiments/thiago/mnist/manifest.json
python scripts/run_experiments.py render-compose thiago-mnist-ring-6-manual --manifest experiments/thiago/mnist/manifest.json --run-id inspection
python scripts/run_experiments.py smoke --all --manifest experiments/thiago/mnist/manifest.json --run-id smoke-mnist
python scripts/run_experiments.py run --all --manifest experiments/thiago/cifar10/manifest.json --run-id full-cifar10
```

A smoke uses one round, one epoch, up to 400 total candidate training examples
and up to 200 test examples. The runner builds each image, monitors completion,
saves logs and removes the network before proceeding. Results follow each
manifest's `results_root`, under `<experiment-id>/<run-id>/`. Host ports are
shared, so these networks run sequentially.

`experiments/thiago/run_experiments.ps1 -RunId full-01` runs all twelve cases
and creates charts. Fixed-data commands require an explicit manifest; collection
manifests must use the collection command rather than silently ignoring arrivals.
The generic Compose builder retains its established `dfl-paper-<dataset>` image
tags for compatibility; these tags do not refer to an experiment catalog.

### Export standalone Compose configurations

```powershell
python scripts/export_manifest.py --manifest experiments/thiago/mnist/manifest.json --output results/exports/thiago-mnist --run-id exported-01
```

Each exported experiment contains `compose.yaml`, a shared `.env` and one
`env/client_<id>.env` per client. Run Docker Compose from that directory.
The run ID controls both runtime identity and the result mount; changing it in
`.env` selects a new result directory. Host references use relative paths when
possible. Build context and Dockerfile must point into a checkout containing
the code. The export validates all entries before writing and refuses occupied
destinations. Export currently supports fixed-data manifests, not collection.

### Manual partitions

`PARTITION_MODE=manual` requires `CLIENT_CLASSES`, a JSON object mapping every
client ID to a list of allowed integer classes. Cover all available classes.
Classes assigned to multiple clients are divided without duplicating samples;
empty lists are allowed only for explicitly empty clients. Partitioning and
sample capping are deterministic. The existing Thiago manifests demonstrate
concentration on selected clients.

## Simulated collection and HARTH

See [the collection guide](experiments/thiago-coleta-simulada/README.md) for
agenda syntax, defaults, HARTH preparation and outputs.

```powershell
python scripts/run_collection_experiments.py list
python scripts/run_collection_experiments.py smoke --all --run-id collection-smoke-01
python scripts/run_collection_experiments.py run --all --run-id collection-full-01
```

The six cases pair MNIST, CIFAR-10 and HARTH with accumulation and a three-round
sliding window. A schedule releases previously unseen local examples at the
start of each local round. Accumulation retains every arrival; a window retains
arrivals from the current round and the previous N−1 rounds. Empty rounds still
advance the window. Expiration removes training examples, not learned weights.
The test set stays fixed. Clients with no active examples skip training and
have zero aggregation weight; if all weights are zero, the model is retained.

The executor validates every selected client's complete schedule against actual
stocks before starting any client services. Quotas cannot exceed stock;
examples cannot be shared across clients or collected twice. Preflight hashes
are checked again by each service. MNIST/CIFAR-10 schedules use class quotas;
HARTH consumes chronological windows without choosing classes.

HARTH uses six sensor channels, five-second nonoverlapping windows and twelve
activity classes. Four participants form the held-out test set; the remaining
18 form six groups of three. Preparation and subject/window semantics are
implemented by its trainer and documented in the collection guide. Prepared
arrays use memory mapping and SHA-256 integrity checks.

All three trainers expose optional preparation through the same command:

```powershell
python -m scripts.experiment_tools.prepare_data --trainer-module experiments.harth.trainer --data-dir results/coleta-data
```

Replace the module with `experiments.mnist.trainer` or
`experiments.cifar10.trainer` for the image datasets. No preparation or class
rules for an individual dataset live in the shared catalog or validator.

## Johann timed experiments

Johann retains its own temporal executor, eight-client topology, availability
schedules, M0–M4 methods, calibration and analysis. It does not use collection
rounds. Read [its protocol](experiments/johann/README.md) before pilot or final
runs; development defaults are not a frozen scientific configuration.

```powershell
python scripts/run_johann_experiments.py list --dataset all
python scripts/run_johann_experiments.py smoke --dataset all --methods M0 --scenario concentrated-r --run-id johann-smoke-01
```

Calibration, selected configurations, seeds, execution duration and observation
cadence remain controlled by the existing Johann manifests and command options.
Official Johann `run` now requires `--frozen-bundle <directory>`: preserved
parameters and image IDs are used without rebuilding or pulling, and all eight
container image IDs are verified before training starts. `pilot` and `smoke`
remain available for development. See the Johann protocol for frozen execution
and offline `render-compose` examples.

## Runtime API and settings

| Endpoint | Behavior |
|---|---|
| `GET /health` | Readiness and client identity. |
| `POST /start` | Idempotent start with a shared run ID; supports a scheduled start. |
| `GET /model` | Published CPU state and matching version, round and sample count. |
| `GET /status` | Progress, state, latest metrics and recorded errors. |

Typical direct configuration includes `TRAINER_MODULE`, `CLIENT_ID`,
`CLIENT_COUNT`, `HEAD_CLIENT_IDS`, `NEIGHBORS`, `RUN_ID`, `ROUNDS`,
`LOCAL_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `MOMENTUM`, `SEED`,
`DATA_DIR`, `RESULTS_DIR` and request retry settings. Use explicit neighbor
URLs outside Compose. Start propagation retries survive local completion;
unavailable model neighbors are recorded and skipped.

Collection is disabled by default. Its settings are `COLLECTION_MODE`,
`COLLECTION_WINDOW_ROUNDS`, `COLLECTION_SCHEDULE_PATH` and optionally
`COLLECTION_PREFLIGHT_PATH`. Collection requires the round-based runtime.
Timed Johann settings remain documented with that experiment.

## Results and charts

Each run saves per-client JSONL metrics, final CPU checkpoints, topology and
class-distribution artifacts, plus an optional final confusion matrix. Round
metrics contain published versions, local weight, observed neighbors, errors,
train/test metrics and per-class recall/support. Collection additionally saves
resolved arrival plans, release/expiration IDs and per-round stock/retention
counts. Fitting and evaluation use distinct fixed datasets.

```powershell
python scripts/plot_metrics.py --show-neighbor-count --show-client-points results/thiago/mnist/<experiment-id>/<run-id> outputs/thiago-example
python scripts/run_experiments.py summarize results/thiago/mnist
```

Individual plots use model version by default; `--x-axis time` uses elapsed
wall time. Combined round plots align client observations, show population
mean and standard deviation and can overlay individual points. Missing rounds
and zero-support recalls remain gaps. Confusion counts, class distributions
and direct parameter mixing contributions have separate charts/CSVs; mixing
weights are not a historical attribution of model knowledge.

The plotting helpers retain generic multiclass, binary and multilabel support
and can read historical metric files independently of which trainers remain
in the checkout. Legacy metadata fields such as `paper_role` are preserved.
Johann timed plots continue to distinguish periodic/final observations from
intervention diagnostics.

## Extending and reusing the code

Follow [the trainer contract](trainer_instruction.md). Required functions remain
`create_model`, `load_data`, `train_local` and `evaluate`; final confusion,
preparation and collection metadata are optional capabilities.

Within `scripts.experiment_tools`:

- `catalog` / `topology`: explicit manifests, `ExperimentSpec`, graph components and heads.
- `compose`: client environments, Docker services, mounts and healthchecks.
- `collection`: collection settings and Compose augmentation; no default manifest.
- `validation`: stock disjointness, schedule feasibility and plan hashes.
- `prepare_data`: dispatch through a trainer's optional preparation function.
- `execution`: rendering, round monitoring, logs, outcomes and project cleanup.
- `reporting`: per-run and cross-run summaries for current/historical metrics.

Dataset-specific code belongs with its trainer; retention and arrivals belong
in `app.collection`. Adding a collection example requires a trainer with a
source/class-code capability, an explicit manifest and an agenda. Command
entry points choose defaults, not the shared library.

## GNS3

Build images from this checkout and transfer them to the target Docker engine.
Set explicit reachable IP-based `NEIGHBORS` URLs using internal port 9000;
Compose service names and host-published ports are not node IPs. Keep trainer,
client count, topology, head IDs and run ID consistent on every node. Persist
results separately for each experiment. In stars, client 0 is the head; in
the supplied standalone rings, the last client is the head. Validate IP
reachability, health, start propagation and one capped round before a full run.
Rebuild/recreate nodes after code or image changes.

## Development and validation

```powershell
python -m pytest
python -m compileall -q app experiments tests scripts
docker compose -f experiments/mnist/compose.ring.8.yaml config --quiet
docker compose -f experiments/cifar10/compose.ring.8.yaml config --quiet
```

Tests cover API behavior, deterministic partitions, compatibility, atomic
publication, failures, collection retention, HARTH segmentation and subject
splits, catalogs, Compose generation, exports and generic metrics. Unit tests
use synthetic data without network downloads. Container smokes are separate
from scientific collection; see the dated validation records for evidence.

The current reorganization preserved the pre-refactor implementation in a
published commit before removing retired experiment families. Results, datasets,
images and research deliverables were retained. Regenerate old Compose/export
files that reference moved trainer/tool modules. Existing results remain
readable; rebuilding images is required to run the new code.

## Security boundary

Model exchange is trusted plaintext. Only deserialize snapshots within an
isolated, trusted experiment network. Do not commit secrets, `.env`, datasets,
checkpoints or `results/`. Research and presentation deliverables belong in
`outputs/`; generated execution artifacts belong in ignored `results/`.
