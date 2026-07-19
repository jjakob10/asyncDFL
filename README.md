# Decentralized Federated Learning on GNS3

This repository is a deliberately small decentralized federated learning (DFL)
baseline. Eight independent FastAPI containers train a PyTorch CNN on disjoint
MNIST partitions. Each client exchanges models only with its two logical ring
neighbors:

```text
client-0 -- client-1 -- client-2 -- client-3
   |                                  |
client-7 -- client-6 -- client-5 -- client-4
```

Client 7 is the head only for starting the experiment. It is not an aggregation
server. Once started, all eight clients train and aggregate independently.

## How a round works

For every round, a client:

1. copies its latest published model and trains the copy on its local data;
2. requests the latest completed model snapshot from both neighbors;
3. performs sample-count-weighted FedAvg over its trained copy and all
   successfully retrieved neighbor snapshots;
4. atomically publishes the aggregate as its next model version; and
5. evaluates the aggregate on the MNIST test set and records the result.

Clients are asynchronous. A fast client may retrieve version 3 from one neighbor
and version 2 from the other. There is no global round barrier. If a neighbor
cannot be reached after the configured retries, the client records the error and
aggregates the models that are available.

The start message contains a run ID and is flooded around the ring. A client
accepts its first run ID exactly once; duplicate deliveries are harmless. A
different run ID is rejected until the container is restarted.

## Run with Docker Compose

Requirements:

- Docker Engine or Docker Desktop with Compose v2
- approximately 3 GB of free disk space for the CPU PyTorch image
- enough CPU/RAM to run eight training containers concurrently

Copy `.env.example` to `.env` if you want to change the defaults. Then run:

```bash
docker compose build
docker compose up
```

Compose waits for clients 0 through 6 to become healthy before it starts client
7. Five seconds after client 7 becomes ready, it creates a run ID and begins the
one-time start flood.

In another terminal, monitor all clients:

```bash
python scripts/wait_for_run.py
```

Useful individual requests:

```bash
curl http://localhost:8007/status
curl http://localhost:8000/health
curl -o client-3.pt http://localhost:8003/model
```

Host ports 8000 through 8007 correspond to clients 0 through 7. Per-round JSONL
files and final state dictionaries are written to `results/`.

For a quick smoke test, put the following in `.env`:

```dotenv
ROUNDS=1
MAX_TRAIN_SAMPLES=1600
MAX_TEST_SAMPLES=1000
```

Stop and remove the containers with:

```bash
docker compose down
```

Restart the containers before beginning another experiment because one process
accepts only one run ID.

## Configuration

The most useful environment variables are:

| Variable | Default | Meaning |
|---|---:|---|
| `CLIENT_ID` | derived from name | Numeric ID from 0 to 7 |
| `CLIENT_NAME` | `client-0` | Node name; a trailing integer is parsed as the ID |
| `CLIENT_COUNT` | `8` | Number of clients |
| `HEAD_CLIENT_ID` | `7` | Client that creates the initial run ID |
| `NEIGHBORS` | derived | Comma-separated neighbor base URLs |
| `ROUNDS` | `10` | Decentralized training rounds |
| `LOCAL_EPOCHS` | `1` | Local epochs per round |
| `BATCH_SIZE` | `64` | Training batch size |
| `LEARNING_RATE` | `0.01` | SGD learning rate |
| `MOMENTUM` | `0.9` | SGD momentum |
| `PARTITION_MODE` | `iid` | `iid` or `dirichlet` |
| `DIRICHLET_ALPHA` | `0.5` | Label-skew strength; smaller is more heterogeneous |
| `MIN_PARTITION_SIZE` | `100` | Minimum examples in a Dirichlet shard |
| `SEED` | `42` | Model, partition, and loader seed |
| `REQUEST_TIMEOUT` | `5` | Per-request timeout in seconds |
| `REQUEST_RETRIES` | `3` | Attempts for start/model requests |
| `RETRY_DELAY` | `1` | Seconds between attempts |
| `HEAD_START_DELAY` | `5` | Head delay before creating the run |
| `RESULTS_DIR` | `/results` | Metrics and final-model directory |
| `MAX_TRAIN_SAMPLES` | `0` | Optional global training cap; 0 means full MNIST |
| `MAX_TEST_SAMPLES` | `0` | Optional test cap; 0 means full MNIST |
| `TORCH_NUM_THREADS` | `1` | CPU threads per client; prevents eight-way oversubscription |

If `NEIGHBORS` is omitted, the service derives its ring neighbors from
`CLIENT_ID`, `CLIENT_COUNT`, `GNS3_IP_PREFIX` (default `10.10.0.`), and
`GNS3_IP_OFFSET` (default `10`).

## REST API

### `GET /health`

Returns readiness and client ID.

### `POST /start`

```json
{"run_id": "experiment-2026-07-17"}
```

Returns HTTP 202 for the first or a duplicate delivery. Returns HTTP 409 if the
container has already accepted another run.

### `GET /model`

Returns a trusted PyTorch state dictionary as `application/octet-stream`.
Metadata is carried in these response headers:

- `X-Client-ID`
- `X-Model-Version`
- `X-Completed-Round`
- `X-Sample-Count`

### `GET /status`

Returns current state, completed round, model version, sample count, latest
metrics, neighbor versions used in the last round, and recent communication
errors.

## Build and load the image for GNS3

GNS3 must have access to the built image on the server that executes Docker
nodes. If Docker and GNS3 use the same engine, build it directly:

```bash
docker build -t dfl-mnist:latest .
```

If GNS3 runs in the GNS3 VM or on a remote server, transfer the image:

```bash
docker save dfl-mnist:latest -o dfl-mnist.tar
# Copy dfl-mnist.tar to the GNS3 VM/server, then:
docker load -i dfl-mnist.tar
```

MNIST is downloaded during the image build and stored under `/data`, so the
topology does not need internet access at runtime.

## Create the GNS3 topology

### 1. Create one reusable Docker template

In **Preferences → Docker containers**, create a template with:

- image: `dfl-mnist:latest`
- adapters: `1`
- start command: leave empty so the image `CMD` runs
- console: `none` (or an auxiliary shell if desired)
- HTTP port: `8000` if your GNS3 version offers the field
- persistent volume: `/results`
- environment:

```dotenv
CLIENT_NAME=%vm-name%
CLIENT_COUNT=8
HEAD_CLIENT_ID=7
ROUNDS=10
LOCAL_EPOCHS=1
BATCH_SIZE=64
PARTITION_MODE=iid
DIRICHLET_ALPHA=0.5
SEED=42
GNS3_IP_PREFIX=10.10.0.
GNS3_IP_OFFSET=10
REQUEST_TIMEOUT=5
REQUEST_RETRIES=3
RETRY_DELAY=1
HEAD_START_DELAY=5
TORCH_NUM_THREADS=1
```

The `%vm-name%` template variable lets every node use the same image and
template. GNS3 documents Docker template environment variables and persistent
volumes in its
[Docker template guide](https://docs.gns3.com/docs-3.1-en/web-ui/template-preferences-docker).

### 2. Add nodes and links

1. Add one built-in Ethernet switch.
2. Add eight instances of the Docker template.
3. Name them exactly `client-0` through `client-7`.
4. Connect each container's `eth0` to the switch.

The physical topology is a shared Ethernet segment, while the calculated
neighbor URLs enforce the application-level ring.

### 3. Assign static addresses

With each container stopped, right-click it and choose **Edit config**. Configure
`eth0` using this table:

| Node | Address | Ring neighbors |
|---|---|---|
| client-0 | `10.10.0.10/24` | `.17`, `.11` |
| client-1 | `10.10.0.11/24` | `.10`, `.12` |
| client-2 | `10.10.0.12/24` | `.11`, `.13` |
| client-3 | `10.10.0.13/24` | `.12`, `.14` |
| client-4 | `10.10.0.14/24` | `.13`, `.15` |
| client-5 | `10.10.0.15/24` | `.14`, `.16` |
| client-6 | `10.10.0.16/24` | `.15`, `.17` |
| client-7 | `10.10.0.17/24` | `.16`, `.10` |

For example, client 0's `/etc/network/interfaces` entry is:

```text
auto eth0
iface eth0 inet static
    address 10.10.0.10
    netmask 255.255.255.0
```

Change only the address for the other clients. A gateway is unnecessary for
this isolated topology. GNS3 explains why Docker-node IP configuration is
manual in its [Docker networking documentation](https://docs.gns3.com/docs/emulators/docker-support-in-gns3).

### 4. Start the experiment

1. Start clients 0 through 6.
2. Wait until their APIs are ready. From a client console, for example:

   ```bash
   python -c "import urllib.request; print(urllib.request.urlopen('http://10.10.0.10:8000/health').read())"
   ```

3. Start client 7 last.
4. After five seconds, client 7 starts itself and floods the start message.
5. Inspect each auxiliary console, `/status`, or the persistent `/results`
   volume to verify completion.

## Tests

Install the development dependencies and run:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

The suite covers deterministic IID/Dirichlet partitions, partition integrity,
weighted FedAvg, model compatibility and serialization, API behavior,
one-run-only semantics, atomic version progress, result files, and unavailable
neighbors.

## Security boundary

This baseline assumes a trusted, isolated experiment network. Model files and
start messages use plaintext HTTP without authentication. Do not deserialize
models from untrusted nodes. QKD-derived key establishment, encryption,
authentication, TLS, resumable checkpoints, and multiple runs per process are
intentionally outside the baseline and can be added around the REST model
transfer later.
