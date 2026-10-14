# Decentralized Federated Learning Experiments

This repository provides a generic decentralized federated learning client and
pluggable experiment trainers. Independent FastAPI clients train on disjoint
dataset partitions and exchange model snapshots with their configured neighbors.
The standalone MNIST and CIFAR-10 Compose stacks support rings and stars of
four or eight clients, with non-IID Dirichlet training partitions by default.
ECG supports rings of four or eight clients.
The paper experiment runner also supports regular and irregular graphs,
isolated clients, and clients without local training data.

The built-in experiments are:

- `mnist`: the original grayscale CNN baseline.
- `cifar10`: a compact three-block RGB CNN baseline.
- `ecg`: FedCVD's 12-lead ResNet-34 for 20-label ECG diagnosis.
- `ton_iot`: a tabular binary network-intrusion classifier, run through the
  paper experiment runner.

All experiments use the same REST API, asynchronous FedAvg
runtime, metrics format, and configuration surface.

## Repository layout

```text
app/                       Generic client, runtime, federation, and trainer API
experiments/
  mnist/
    trainer.py             MNIST model, data, training, and evaluation
    Dockerfile
    compose.ring.8.yaml    Eight-client ring
    compose.ring.4.yaml    Four-client ring
    compose.star.4.yaml    Four-client star
    compose.star.8.yaml    Eight-client star
    compose.*.iid.yaml    IID variants of each ring/star size
    .env.example
  cifar10/
    trainer.py             CIFAR-10 model, data, training, and evaluation
    Dockerfile
    compose.ring.8.yaml    Eight-client ring
    compose.ring.4.yaml    Four-client ring
    compose.star.4.yaml    Four-client star
    compose.star.8.yaml    Eight-client star
    compose.*.iid.yaml    IID variants of each ring/star size
    .env.example
  ecg/
    trainer.py             ECG model, lazy HDF5 data, training, and evaluation
    Dockerfile
    compose.ring.8.yaml    Eight-client ring
    compose.ring.4.yaml    Four-client ring
    .env.example
  ton_iot/
    trainer.py             Tabular preprocessing, binary model, and evaluation
    Dockerfile
  paper/
    manifest.json          Dataset, topology, and epoch matrix
    catalog.py             Resolved experiment configurations
scripts/                   Monitoring and metric plotting
  run_paper_experiments.py Compose generation, execution, and summaries
tests/                     Generic runtime and experiment regression tests
results/
  paper/                   Runner results grouped by experiment and run ID
  mnist-{ring,star}-{4,8}/   Generated MNIST results by topology and size
  cifar10-{ring,star}-{4,8}/ Generated CIFAR-10 results by topology and size
  ecg-4/                   Generated four-client ECG results
  ecg-8/                   Generated eight-client ECG results
```

## Trainer interface

The `TRAINER_MODULE` environment variable selects an importable trainer module:

```dotenv
TRAINER_MODULE=experiments.mnist.trainer
```

Each trainer module must provide these functions:

```python
def create_model() -> torch.nn.Module: ...
def load_data(settings: Settings) -> DataBundle: ...
def train_local(base_state, train_loader, settings) -> tuple[StateDict, dict]: ...
def evaluate(state, test_loader) -> dict: ...
```

The client validates the module at startup. The runtime has no dataset or model
imports and delegates model creation, data loading, local training, and
evaluation to the selected trainer. Shared deterministic seeding and IID or
Dirichlet partition generation are available from `app.training`.

To add an experiment, create a new importable directory under `experiments/`,
implement the trainer interface, and copy one of the existing Dockerfile and
Compose configurations. Set `TRAINER_MODULE` to the new module and change the
Dockerfile's dataset pre-download step.

## Run an experiment

Docker Engine or Docker Desktop with Compose v2 is required. Each image uses
CPU-only PyTorch. MNIST and CIFAR-10 download their datasets during the build.
ECG and TON_IOT require external datasets mounted read-only at runtime. With
these inputs available, containers can run without internet access.

Build and start MNIST:

```bash
docker compose -f experiments/mnist/compose.ring.8.yaml build
docker compose -f experiments/mnist/compose.ring.8.yaml up
```

Build and start CIFAR-10:

```bash
docker compose -f experiments/cifar10/compose.ring.8.yaml build
docker compose -f experiments/cifar10/compose.ring.8.yaml up
```

Choose one standalone configuration from this matrix. Each file defines the
entire experiment; use it directly rather than combining it with another file.

| Dataset | Topology | Nodes | Compose file | Results directory |
| ------- | -------- | ----: | ------------ | ----------------- |
| MNIST | Ring | 4 | `experiments/mnist/compose.ring.4.yaml` | `results/mnist-ring-4/` |
| MNIST | Ring | 8 | `experiments/mnist/compose.ring.8.yaml` | `results/mnist-ring-8/` |
| MNIST | Star | 4 | `experiments/mnist/compose.star.4.yaml` | `results/mnist-star-4/` |
| MNIST | Star | 8 | `experiments/mnist/compose.star.8.yaml` | `results/mnist-star-8/` |
| CIFAR-10 | Ring | 4 | `experiments/cifar10/compose.ring.4.yaml` | `results/cifar10-ring-4/` |
| CIFAR-10 | Ring | 8 | `experiments/cifar10/compose.ring.8.yaml` | `results/cifar10-ring-8/` |
| CIFAR-10 | Star | 4 | `experiments/cifar10/compose.star.4.yaml` | `results/cifar10-star-4/` |
| CIFAR-10 | Star | 8 | `experiments/cifar10/compose.star.8.yaml` | `results/cifar10-star-8/` |

### IID variants

The existing image experiments default to Dirichlet partitioning. These eight
standalone variants fix `PARTITION_MODE=iid`, even if an environment file sets
`PARTITION_MODE=dirichlet`. Each reuses its dataset image and adds `-iid` to
the project name, experiment ID, default run ID, and results directory.
Training hyperparameters and topologies match their Dirichlet counterparts.

| Dataset | Topology | Nodes | Compose file | Results directory |
| ------- | -------- | ----: | ------------ | ----------------- |
| MNIST | Ring | 4 | `experiments/mnist/compose.ring.4.iid.yaml` | `results/mnist-ring-4-iid/` |
| MNIST | Ring | 8 | `experiments/mnist/compose.ring.8.iid.yaml` | `results/mnist-ring-8-iid/` |
| MNIST | Star | 4 | `experiments/mnist/compose.star.4.iid.yaml` | `results/mnist-star-4-iid/` |
| MNIST | Star | 8 | `experiments/mnist/compose.star.8.iid.yaml` | `results/mnist-star-8-iid/` |
| CIFAR-10 | Ring | 4 | `experiments/cifar10/compose.ring.4.iid.yaml` | `results/cifar10-ring-4-iid/` |
| CIFAR-10 | Ring | 8 | `experiments/cifar10/compose.ring.8.iid.yaml` | `results/cifar10-ring-8-iid/` |
| CIFAR-10 | Star | 4 | `experiments/cifar10/compose.star.4.iid.yaml` | `results/cifar10-star-4-iid/` |
| CIFAR-10 | Star | 8 | `experiments/cifar10/compose.star.8.iid.yaml` | `results/cifar10-star-8-iid/` |

For example:

```bash
docker compose -f experiments/mnist/compose.ring.4.iid.yaml up --build -d
python scripts/wait_for_run.py --client-count 4
docker compose -f experiments/mnist/compose.ring.4.iid.yaml down
```

All Compose stacks keep published ports `8000`, `8001`, and so on, mapping
to internal port `9000`. Local monitoring therefore still starts at 8000.
Run stacks sequentially when they use the same published host ports.

Every node receives a disjoint training partition and trains, including the
star center (`client-0`). Ring nodes exchange models with their two adjacent
neighbors. Star leaves exchange only with the center, which exchanges with
all leaves. Both topologies use the existing asynchronous decentralized
rounds. All nodes evaluate on the same test set (or the same capped subset).
The run ID defaults to the dataset, topology, and node count, such as
`mnist-star-4`; each stack also has a distinct Compose project name.

For example, build, start, monitor, and stop the four-node MNIST star:

```bash
docker compose -f experiments/mnist/compose.star.4.yaml build
docker compose -f experiments/mnist/compose.star.4.yaml up -d
python scripts/wait_for_run.py --client-count 4
docker compose -f experiments/mnist/compose.star.4.yaml down
```

Substitute any file from the matrix and use `--client-count 8` for eight nodes.

ECG uses an external preprocessed FedCVD data tree. Build and start the
eight-client ring:

```bash
docker compose -f experiments/ecg/compose.ring.8.yaml build
docker compose -f experiments/ecg/compose.ring.8.yaml up
```

Or run the four-client ring with the same `dfl-ecg:latest` image:

```bash
docker compose -f experiments/ecg/compose.ring.4.yaml build
docker compose -f experiments/ecg/compose.ring.4.yaml up
```

Only run one stack at a time because their published host-port ranges overlap.
In rings, the last client (3 or 7) waits for the other clients to become
healthy and then starts the experiment. Stars use client 0 as the initiator.
The node count and initiator are fixed by the selected Compose file.
For eight-client stacks, monitor completion in another terminal:

```bash
python scripts/wait_for_run.py
```

For four-client stacks, monitor ports 8000 through 8003 with:

```bash
python scripts/wait_for_run.py --client-count 4
```

Use `--client-count N` for any positive number of clients. The script polls
consecutive ports on `127.0.0.1`, starting at 8000 by default. For example, to
monitor 12 clients on ports 9000 through 9011:

```bash
python scripts/wait_for_run.py --client-count 12 --base-port 9000
```

The script prints each client's state and round every five seconds, retries
while APIs are unavailable, and exits successfully when every client is
`completed`. If a client reports `failed`, it exits with an error; inspect that
client's `/status` endpoint and container logs. The paper runner monitors its
configured client count automatically.

Stop the selected stack with:

```bash
docker compose -f experiments/mnist/compose.ring.8.yaml down
docker compose -f experiments/cifar10/compose.ring.8.yaml down
docker compose -f experiments/ecg/compose.ring.8.yaml down
docker compose -f experiments/ecg/compose.ring.4.yaml down
```

Per-round JSONL files and final PyTorch state dictionaries are written to the
experiment-specific directory under `results/`.

### Override defaults

Copy the appropriate example and edit it:

```bash
cp experiments/mnist/.env.example experiments/mnist/.env
docker compose --env-file experiments/mnist/.env \
  -f experiments/mnist/compose.ring.8.yaml up
```

The same dataset `.env` works with all eight of its topology/partition files.
The MNIST and CIFAR-10 variants without `.iid` default to `PARTITION_MODE=dirichlet`,
`DIRICHLET_ALPHA=0.5`, `MIN_PARTITION_SIZE=100`, and `SEED=42`.
Set these in `.env` or change their defaults in the selected Compose file.
Use `PARTITION_MODE=iid` to select uniform random splitting instead, or select
a `.iid.yaml` variant to keep IID fixed regardless of the environment file.

Dirichlet draws a distribution across clients independently for each class.
Smaller positive alpha values generally produce more concentrated class
distributions; larger values tend toward more even distributions. Partition
sizes can differ. The seed makes the partition reproducible for the same
dataset, sample cap, number of clients, and partition parameters.

For a quick smoke test, copy the dataset's `.env.example` to `.env` and edit
these values (the training cap applies globally before partitioning):

```dotenv
ROUNDS=1
PARTITION_MODE=dirichlet
DIRICHLET_ALPHA=0.5
MIN_PARTITION_SIZE=100
MAX_TRAIN_SAMPLES=800
MAX_TEST_SAMPLES=200
```

Then run, for example, the eight-node CIFAR-10 star:

```bash
docker compose --env-file experiments/cifar10/.env \
  -f experiments/cifar10/compose.star.8.yaml up --build -d
python scripts/wait_for_run.py --client-count 8
docker compose --env-file experiments/cifar10/.env \
  -f experiments/cifar10/compose.star.8.yaml down
```

The image trainers automatically lower the requested minimum partition size
to `min(MIN_PARTITION_SIZE, max(1, training_samples // CLIENT_COUNT // 2))`.
For 800 training samples and eight clients, the effective minimum is 50.
Dirichlet sampling retries until every partition meets that minimum; an
impossible dataset size or exhaustion of the 1,000 attempts produces an error.
Use at least one training sample per client. Sample caps of zero use the full
datasets. Recreate containers for a new execution; choose a custom `RUN_ID`
and archive existing results if you want to preserve multiple runs of the
same configuration.

For ECG, copy the shared example and set `ECG_DATA_DIR` to the host directory
containing `ECG/preprocessed`:

```bash
cp experiments/ecg/.env.example experiments/ecg/.env
```

The checked-in default
`../../../FedCVD/data` matches a FedCVD checkout adjacent to this repository.
An absolute path is more portable:

```dotenv
# Windows with Docker Desktop
ECG_DATA_DIR=C:/Users/you/Documents/FedCVD/data

# Linux
ECG_DATA_DIR=/home/you/FedCVD/data
```

The expected layout is:

```text
ECG_DATA_DIR/
  ECG/
    preprocessed/
      <location>/
        train.csv
        test.csv
        records.h5
```

The four reference locations occupy approximately 9.35 GB. Compose mounts this
directory at `/data` read-only; it is never copied into the image or written by
the clients. The trainer merges all source locations and then deterministically
repartitions the combined training rows. The four-client topology therefore
matches FedCVD's client count but does not map one runtime client to one source
location.

ECG reads HDF5 with `pyfive==1.2.1`, a Python reader that does not load the
native `h5py` extensions blocked by Smart App Control on some Windows systems.
The input layout and files stay the same. Reinstall dependencies with
`python -m pip install -r requirements-dev.txt` and rebuild the ECG image after
updating this checkout. No change to Windows security settings is required.

The test suite validates synthetic HDF5 files with contiguous and gzip storage,
including float16 signals and invalid records. Some HDF5 features or compression
filters may not be supported by pyfive. Compatibility and performance with your
real FedCVD files must be checked separately; read errors include the file and
record ID, and the trainer does not fall back to h5py.

For a quick smoke test, use:

```dotenv
ROUNDS=1
MAX_TRAIN_SAMPLES=1600
MAX_TEST_SAMPLES=1000
```

For example:

```bash
docker compose --env-file experiments/ecg/.env \
  -f experiments/ecg/compose.ring.4.yaml up
```

The most useful settings are:

| Variable             |       Default | Meaning                                |
| -------------------- | ------------: | -------------------------------------- |
| `TRAINER_MODULE`     | MNIST trainer | Import path for the experiment trainer |
| `CLIENT_COUNT`       |           `8` | Number of clients                      |
| `HEAD_CLIENT_IDS`    |           `7` | Comma-separated clients that auto-start |
| `RUN_ID`             |     `dfl-run` | Shared identity for one execution       |
| `EMPTY_CLIENT_IDS`   |       *(none)* | Clients with no local training data     |
| `ROUNDS`             |          `10` | Decentralized training rounds          |
| `LOCAL_EPOCHS`       |           `1` | Local epochs per round                 |
| `BATCH_SIZE`         |          `64` | Training batch size                    |
| `LEARNING_RATE`      |        `0.01` | SGD learning rate                      |
| `MOMENTUM`           |         `0.9` | SGD momentum                           |
| `PARTITION_MODE`     |         `iid` | `iid` or `dirichlet`                   |
| `DIRICHLET_ALPHA`    |         `0.5` | Dirichlet label-skew strength          |
| `MIN_PARTITION_SIZE` |         `100` | Minimum examples in a Dirichlet shard  |
| `SEED`               |          `42` | Model, partition, and loader seed      |
| `MAX_TRAIN_SAMPLES`  |           `0` | Optional global training cap           |
| `MAX_TEST_SAMPLES`   |           `0` | Optional test cap                      |
| `TORCH_NUM_THREADS`  |           `1` | CPU threads per client                 |
| `API_PORT`          |        `9000` | Internal API and derived neighbor port |

These are application defaults. The standalone MNIST/CIFAR-10 Compose files
default to `dirichlet`, except the `.iid.yaml` variants, which fix `iid`.
They also set identity and topology metadata for the selected experiment.
The paper catalog keeps its existing defaults.

`NEIGHBORS` may explicitly list comma-separated URLs. If omitted, clients
derive ring neighbors from `CLIENT_ID`, `CLIENT_COUNT`, `GNS3_IP_PREFIX`, and
`GNS3_IP_OFFSET`. An explicitly empty value (`NEIGHBORS=""`) gives the client
no neighbors, for isolated local training or evaluation.

### Migrate configuration from the main branch

Replace `HEAD_CLIENT_ID=7` with `HEAD_CLIENT_IDS=7`. The old variable is no
longer read; a custom head configured only through it will fall back to the
default head list. Multiple heads use comma-separated IDs, such as
`HEAD_CLIENT_IDS=3,7`, with one head per connected component.

Automatic starts now use the configured `RUN_ID` instead of generating a UUID.
Give every client in an execution the same value. Each process accepts one
execution: repeating its run ID is idempotent, while a different ID conflicts
after it has started. Recreate the containers to start another execution.

The standalone Compose stacks fix their head IDs and neighbor URLs. Changing
those requires editing their Compose configuration; adding environment values
to an `.env` file only overrides variables referenced by that Compose file.

## Round behavior and API

For every round, each client:

1. trains a copy of its latest model if it has a nonempty local partition;
2. fetches the latest snapshots from its configured neighbors;
3. performs FedAvg over the available models with weight `max(sample_count, 1)`;
4. atomically publishes the aggregate; and
5. evaluates and records the new model.

Clients listed in `EMPTY_CLIENT_IDS` receive no training samples. They skip
local training, record an empty `train` metrics object, and still exchange,
aggregate, and evaluate models. Their models retain aggregation weight 1;
`sample_count` remains 0 in model headers and metrics. Nonempty clients use
their number of unique local training examples as the aggregation weight.

Clients are asynchronous. An unavailable neighbor is recorded and skipped after
the configured retries. A start run ID is forwarded through configured neighbors and accepted
exactly once by every process.

The HTTP interface is unchanged:

- `GET /health`: readiness and client ID.
- `POST /start`: accepts `{"run_id": "experiment-name"}`.
- `GET /model`: current trusted PyTorch state dictionary and version headers.
- `GET /status`: progress, latest metrics, neighbor versions, and errors.

Status and round metrics also identify the experiment, topology, connected
component, head status, presence of local data, and paper role. The paper runner
sets `EXPERIMENT_ID`, `TOPOLOGY_NAME`, `COMPONENT_ID`, `COMPONENT_SIZE`, and
`PAPER_ROLE` from the resolved graph. For manual custom deployments, configure
these metadata fields to match the graph; the runtime does not infer them.

### Interpret metrics

MNIST, CIFAR-10, and TON_IOT report the fraction of examples classified
correctly. ECG is multilabel: predictions use `sigmoid(logits) >= 0.5`, and
`accuracy` counts a record as correct only when all 20 label decisions match
its target. ECG loss is binary cross-entropy with logits averaged over all
label decisions. These accuracy definitions should be considered when comparing
datasets.

## Plot metrics

Plot one or more result directories:

```bash
python scripts/plot_metrics.py results/mnist outputs/mnist
python scripts/plot_metrics.py results/mnist results/cifar10 outputs/comparison
python scripts/plot_metrics.py results/ecg-4 outputs/ecg-4
python scripts/plot_metrics.py results/ecg-8 outputs/ecg-8
```

Use `--phase train`, `--phase test`, or `--x-axis time` to change the plots.

The plot command also writes two combined test charts for each experiment:
`combined_test_accuracy_round.png` and `combined_test_loss_round.png`. These
always use integer rounds, independently of the individual chart options.

Test observations are grouped by their recorded `round` (or `model_version`
for older results without a round field). The mean gives equal weight to each
client with a recorded value at that round, including clients without training
data and disconnected clients. Missing client values are not interpolated or
carried forward. Shading represents one population standard deviation above
and below the mean, rather than a confidence interval. Participation can change
between rounds: the matching CSV files report `round`, `mean`, `std`, and
`participants`. A single participant has zero standard deviation; a client
with one observation contributes only at that round. Duplicate client/round
records use the highest model version, then the last line. Results from
different run IDs must not be mixed in the same directory.

Previously generated `_time` charts and CSVs are preserved. New combined
outputs use the `_round` suffix; individual charts still support `--x-axis time`.

Add `--show-client-points` to overlay every original test observation, with a
distinct color and legend entry per client:

```bash
python scripts/plot_metrics.py --show-client-points results/mnist outputs/mnist
```

On successful completion, each built-in trainer writes
`client_<id>_confusion.json` alongside the final checkpoint. It contains final
test-set counts, client/run identifiers, model version, timestamp, labels, and
the number of evaluated examples. The plot command renders it as
`client_<id>_confusion.png`: actual classes are rows, predicted classes are
columns, and cells contain absolute counts. MNIST/CIFAR-10 use 10×10 matrices;
TON_IoT uses a binary 2×2 matrix. ECG uses a panel of 20 binary matrices, one
per diagnosis code, with 0 meaning absent and 1 meaning present. Existing
prediction rules and accuracy definitions are preserved.

When several experiment directories are supplied, the new charts and CSVs are
placed in separate `experiment_<index>_<label>` subdirectories. Existing
per-client comparison charts remain in the output root. Older results without
confusion artifacts still produce metric charts, with a warning.

The paper runner generates these additional artifacts during `run`, `smoke`,
and `summarize`; each command accepts `--show-client-points`. For example:

```bash
python scripts/run_paper_experiments.py summarize results/paper --show-client-points
```

## Reproduce the GLow paper experiments

The checked-in paper manifest expands to 158 decentralized configurations:
MNIST and CIFAR-10 regular 8+2 and 16+4 topology/epoch sweeps, their sparse
and dense irregular graphs, and the TON_IOT regular topology sweeps. These
experiments intentionally use this framework's asynchronous runtime rather
than GLow's sequential head selection.

List or inspect configurations without starting containers:

```bash
python scripts/run_paper_experiments.py list --dataset mnist
python scripts/run_paper_experiments.py render-compose \
  mnist-8p2-top4-e32 --run-id trial-1
```

Run one capped smoke experiment, one full experiment, or a filtered matrix:

```bash
python scripts/run_paper_experiments.py smoke mnist-8p2-top4-e32
python scripts/run_paper_experiments.py run mnist-8p2-top4-e32
python scripts/run_paper_experiments.py run --all \
  --dataset mnist --topology topology-4 --local-epochs 32
```

Each generated run uses one shared `RUN_ID` across every connected component.
It is an execution identity and idempotency key, not a client or model
version. The highest client ID in each connected component is included in
`HEAD_CLIENT_IDS`; topology 0 therefore makes every isolated client a head.
Results and resolved configuration are stored under
`results/paper/<experiment-id>/<run-id>/`. The runner waits for all clients,
writes CSV/Markdown summaries and accuracy/loss charts, then removes the
containers.

The paper roles describe each client's data and designated control status:

| Role | Meaning |
| ---- | ------- |
| `R` | Regular client with local training data |
| `E` | Empty client without local training data |
| `D` | Disconnected control with local training data |
| `ED` | Disconnected control without local training data |

`is_connected` is true when the client's component has more than one member.
The reported "Connected E/R average accuracy" is the unweighted arithmetic
mean of final test accuracies for connected clients with roles `E` or `R`.
Roles `D` and `ED` and all singleton components are excluded, but still appear
in per-client summaries. When no client is eligible, the aggregate is `n/a`.

Regenerate summaries for an existing run or a collection of runs with:

```bash
python scripts/run_paper_experiments.py summarize results/paper/<experiment-id>/<run-id>
python scripts/run_paper_experiments.py summarize results/paper
```

TON_IOT is not redistributed. Download `Train_Test_Network.csv` from the
official dataset and provide it as a read-only input:

```bash
python scripts/run_paper_experiments.py smoke ton_iot-8p2-top4-e32 \
  --ton-iot-path C:/datasets/Train_Test_Network.csv
```

For direct runtime configuration, set `TRAINER_MODULE=experiments.ton_iot.trainer`
and `TON_IOT_PATH` to the CSV path visible inside the client container. The
runner mounts the file read-only and sets this path automatically. The CSV must
contain `ts`, `type`, and a binary `label` (0 or 1), alongside numeric and
categorical feature columns. The trainer removes `ts` and `type`, creates a
seeded stratified 80/20 train/test split, and fits numeric scaling and one-hot
encoding on the training split before assigning local partitions. It records
preprocessing metadata under the run's results directory.

## Build for GNS3

Build the desired experiment image:

```bash
docker build -f experiments/mnist/Dockerfile -t dfl-mnist:latest .
docker build -f experiments/cifar10/Dockerfile -t dfl-cifar10:latest .
docker build -f experiments/ecg/Dockerfile -t dfl-ecg:latest .
```

If GNS3 uses another Docker engine, transfer the rebuilt image with
`docker save` and `docker load`. Create a Docker template with one adapter
and `/results` as a persistent volume. The API listens on **9000** inside
every experiment container, avoiding the existing GNS3 service on port 8000.
Keep the image startup command; remove old template overrides that launch an
interactive shell or start Uvicorn on port 8000. The image uses:

```dockerfile
ENV API_PORT=9000
EXPOSE 9000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"$API_PORT\""]
```

The shell expands `API_PORT` and `exec` makes Uvicorn the main process so it
receives shutdown signals. Image and Compose healthchecks read the same
variable. If starting Uvicorn manually, pass `--host 0.0.0.0 --port 9000`;
`API_PORT` alone does not change a manually supplied Uvicorn command.

For the existing four-client GNS3 ring, configure:

```dotenv
CLIENT_NAME=%vm-name%
TRAINER_MODULE=experiments.mnist.trainer
CLIENT_COUNT=4
HEAD_CLIENT_IDS=3
API_PORT=9000
RUN_ID=gns3-ring-4-iid
ROUNDS=10
LOCAL_EPOCHS=1
BATCH_SIZE=64
PARTITION_MODE=iid
SEED=42
GNS3_IP_PREFIX=10.10.0.
GNS3_IP_OFFSET=10
HEAD_START_DELAY=5
TORCH_NUM_THREADS=1
```

Verify that each node receives its actual name as `CLIENT_NAME` (`client-0`
through `client-3`); set it individually if the template does not expand
`%vm-name%`. Alternatively, set `CLIENT_ID` explicitly per node. Remove any
stale `CLIENT_ID` that overrides the node name. Use `HEAD_CLIENT_IDS` (plural);
the old `HEAD_CLIENT_ID` is no longer read.

Assign `10.10.0.10/24` through `10.10.0.13/24` to the corresponding node
interfaces and attach them to the same Ethernet segment. The prefix/offset
variables only construct neighbor URLs; they do not configure interfaces.
Leave `NEIGHBORS` **absent**, not empty: `NEIGHBORS=""` means an isolated
client. Client 0 then contacts `http://10.10.0.13:9000` and
`http://10.10.0.11:9000`. Start clients 0 through 2, verify their `/health`
endpoints, and then start client 3, which initiates the shared run.

For CIFAR-10 use the CIFAR-10 image and
`TRAINER_MODULE=experiments.cifar10.trainer`. For eight-client rings use
`CLIENT_COUNT=8`, `HEAD_CLIENT_IDS=7`, names `client-0` through `client-7`,
and IPs `10.10.0.10/24` through `10.10.0.17/24`; start client 7 last.

For stars use `HEAD_CLIENT_IDS=0` and explicit per-node `NEIGHBORS` URLs:
the center (`client-0`, IP `10.10.0.10`) lists all leaves, and every leaf
lists only `http://10.10.0.10:9000`. For four nodes the center uses
`http://10.10.0.11:9000,http://10.10.0.12:9000,http://10.10.0.13:9000`;
for eight, extend that list through `10.10.0.17:9000`. Start the leaves
before the center. Set `CLIENT_COUNT` and a shared `RUN_ID` for the selected
scenario. Persist `/results` separately for each experiment.

After migrating, rebuild the images, transfer them to the GNS3 Docker engine,
and recreate nodes using the updated image/template. Verify IP reachability,
`http://<node-ip>:9000/health`, start propagation, and completion on `/status`
with one round and capped samples before a full run. Compose service names
and published host ports are not substitutes for these GNS3 node IPs.

For ECG, use `dfl-ecg:latest`,
`TRAINER_MODULE=experiments.ecg.trainer`, and mount the host FedCVD data
directory read-only at `/data` on every node. Use client IDs 0 through 7 with
head 7 for the eight-client topology, or IDs 0 through 3 with head 3 for the
four-client topology. GNS3 must provide the data volume because the ECG image
does not embed the approximately 9.35 GB dataset.

## Development and tests

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app experiments tests scripts
python -m pytest
docker compose -f experiments/mnist/compose.ring.8.yaml config --quiet
docker compose -f experiments/cifar10/compose.ring.8.yaml config --quiet
docker compose -f experiments/ecg/compose.ring.8.yaml config --quiet
docker compose -f experiments/ecg/compose.ring.4.yaml config --quiet
```

The suite covers trainer discovery and validation, model smoke training,
partitions, federation, API behavior, ring propagation, unavailable neighbors,
atomic version updates, and result files.

ECG tests read versioned synthetic fixtures and do not require h5py. Their
provenance and isolated Linux/Docker regeneration procedure are documented in
`tests/fixtures/ecg/README.md`. The fixture generator requires h5py only in that
separate maintenance environment.

## Security boundary

This baseline assumes a trusted isolated experiment network. Model files and
start messages use plaintext HTTP without authentication. Do not deserialize
models from untrusted nodes.

## Migration of Compose filenames and API port

For MNIST, CIFAR-10, and ECG, `compose.yaml` is now
`compose.ring.8.yaml`, and `compose.4.yaml` is now `compose.ring.4.yaml`.
The old filenames are removed. Star filenames are unchanged. Update local
commands and automation references accordingly. Stop existing stacks using
their previous configuration before rebuilding/recreating them; retain the
existing results directories. Existing project identities remain unchanged.

The internal API port is now 9000 for MNIST, CIFAR-10, ECG, TON-IoT, and
generated paper experiments. Published host ports remain unchanged. Rebuild
images and recreate containers using the new Compose files, for example:

```bash
docker compose -f experiments/mnist/compose.ring.8.yaml up --build -d
```

For capped IID smoke tests, use a separate environment file with `ROUNDS=1`,
`MAX_TRAIN_SAMPLES=400`, `MAX_TEST_SAMPLES=40` and invoke Compose with
`--env-file <smoke.env>` for each dataset's four-client IID ring. Monitor with
`python scripts/wait_for_run.py --client-count 4`, inspect logs and health,
and stop each stack before starting the next dataset.
