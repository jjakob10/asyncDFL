# Decentralized Federated Learning Experiments

This repository provides a generic decentralized federated learning client and
pluggable experiment trainers. Eight independent FastAPI clients form a logical
ring, train on disjoint dataset partitions, and exchange model snapshots only
with their two neighbors.

The built-in experiments are:

- `mnist`: the original grayscale CNN baseline.
- `cifar10`: a compact three-block RGB CNN baseline.

Both experiments use the same ring topology, REST API, asynchronous FedAvg
runtime, metrics format, and configuration surface.

## Repository layout

```text
app/                       Generic client, runtime, federation, and trainer API
experiments/
  mnist/
    trainer.py             MNIST model, data, training, and evaluation
    Dockerfile
    compose.yaml
    .env.example
  cifar10/
    trainer.py             CIFAR-10 model, data, training, and evaluation
    Dockerfile
    compose.yaml
    .env.example
scripts/                   Monitoring and metric plotting
tests/                     Generic runtime and experiment regression tests
results/
  mnist/                   Generated MNIST results
  cifar10/                 Generated CIFAR-10 results
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
CPU-only PyTorch and downloads its dataset during the build, allowing the
containers to run without internet access.

Build and start MNIST:

```bash
docker compose -f experiments/mnist/compose.yaml build
docker compose -f experiments/mnist/compose.yaml up
```

Build and start CIFAR-10:

```bash
docker compose -f experiments/cifar10/compose.yaml build
docker compose -f experiments/cifar10/compose.yaml up
```

Only run one stack at a time because both publish host ports 8000 through 8007.
Client 7 waits for the other clients to become healthy and then starts the
experiment. Monitor completion in another terminal:

```bash
python scripts/wait_for_run.py
```

Stop the selected stack with:

```bash
docker compose -f experiments/mnist/compose.yaml down
docker compose -f experiments/cifar10/compose.yaml down
```

Per-round JSONL files and final PyTorch state dictionaries are written to the
experiment-specific directory under `results/`.

### Override defaults

Copy the appropriate example and edit it:

```bash
cp experiments/mnist/.env.example experiments/mnist/.env
docker compose --env-file experiments/mnist/.env \
  -f experiments/mnist/compose.yaml up
```

For a quick smoke test, use:

```dotenv
ROUNDS=1
MAX_TRAIN_SAMPLES=1600
MAX_TEST_SAMPLES=1000
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

`NEIGHBORS` may explicitly list comma-separated URLs. If omitted, clients
derive ring neighbors from `CLIENT_ID`, `CLIENT_COUNT`, `GNS3_IP_PREFIX`, and
`GNS3_IP_OFFSET`.

## Round behavior and API

For every round, each client:

1. trains a copy of its latest model on its local partition;
2. fetches the latest snapshots from both ring neighbors;
3. performs sample-count-weighted FedAvg over the available models;
4. atomically publishes the aggregate; and
5. evaluates and records the new model.

Clients are asynchronous. An unavailable neighbor is recorded and skipped after
the configured retries. A start run ID is flooded around the ring and accepted
exactly once by every process.

The HTTP interface is unchanged:

- `GET /health`: readiness and client ID.
- `POST /start`: accepts `{"run_id": "experiment-name"}`.
- `GET /model`: current trusted PyTorch state dictionary and version headers.
- `GET /status`: progress, latest metrics, neighbor versions, and errors.

## Plot metrics

Plot one or more result directories:

```bash
python scripts/plot_metrics.py results/mnist outputs/mnist
python scripts/plot_metrics.py results/mnist results/cifar10 outputs/comparison
```

Use `--phase train`, `--phase test`, or `--x-axis time` to change the plots.

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

TON_IOT is not redistributed. Download `Train_Test_Network.csv` from the
official dataset and provide it as a read-only input:

```bash
python scripts/run_paper_experiments.py smoke ton_iot-8p2-top4-e32 \
  --ton-iot-path C:/datasets/Train_Test_Network.csv
```

## Build for GNS3

Build the desired experiment image:

```bash
docker build -f experiments/mnist/Dockerfile -t dfl-mnist:latest .
docker build -f experiments/cifar10/Dockerfile -t dfl-cifar10:latest .
```

If GNS3 uses another Docker engine, transfer the selected image with
`docker save` and `docker load`. Create one reusable Docker template using the
image, one adapter, port 8000, `/results` as a persistent volume, and:

```dotenv
CLIENT_NAME=%vm-name%
TRAINER_MODULE=experiments.mnist.trainer
CLIENT_COUNT=8
HEAD_CLIENT_IDS=7
RUN_ID=gns3-ring
ROUNDS=10
LOCAL_EPOCHS=1
BATCH_SIZE=64
PARTITION_MODE=iid
DIRICHLET_ALPHA=0.5
SEED=42
GNS3_IP_PREFIX=10.10.0.
GNS3_IP_OFFSET=10
HEAD_START_DELAY=5
TORCH_NUM_THREADS=1
```

Use `experiments.cifar10.trainer` with the CIFAR-10 image. Name the nodes
`client-0` through `client-7`, attach them to the same Ethernet segment, and
assign `10.10.0.10/24` through `10.10.0.17/24`. The derived neighbor URLs
enforce the logical ring. Start clients 0 through 6 before client 7.

## Development and tests

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q app experiments tests scripts
python -m pytest
docker compose -f experiments/mnist/compose.yaml config --quiet
docker compose -f experiments/cifar10/compose.yaml config --quiet
```

The suite covers trainer discovery and validation, model smoke training,
partitions, federation, API behavior, ring propagation, unavailable neighbors,
atomic version updates, and result files.

## Security boundary

This baseline assumes a trusted isolated experiment network. Model files and
start messages use plaintext HTTP without authentication. Do not deserialize
models from untrusted nodes.
