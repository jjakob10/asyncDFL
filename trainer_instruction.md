# Instructions for Implementing an Experiment Trainer

Use this guide to add a dataset/model trainer that can be loaded by the
decentralized federated-learning runtime. The canonical examples are
`experiments/mnist/trainer.py` and `experiments/cifar10/trainer.py`.

## Goal and file location

Create an importable Python package and trainer module:

```text
experiments/
  <experiment_name>/
    __init__.py
    trainer.py
```

Use a lowercase, underscore-separated experiment name. Select the trainer at
runtime with:

```dotenv
TRAINER_MODULE=experiments.<experiment_name>.trainer
```

The runtime imports this module dynamically. Do not add dataset- or
model-specific logic to `app/runtime.py`.

## Required imports and interface

The trainer module must expose all four of these module-level callables with
compatible signatures:

```python
from torch import nn
from torch.utils.data import DataLoader

from app.config import Settings
from app.federation import StateDict
from app.training import DataBundle


def create_model() -> nn.Module: ...


def load_data(settings: Settings) -> DataBundle: ...


def train_local(
    base_state: StateDict,
    train_loader: DataLoader,
    settings: Settings,
) -> tuple[StateDict, dict[str, float]]: ...


def evaluate(
    state: StateDict,
    test_loader: DataLoader,
) -> dict[str, float]: ...
```

`app.training.load_trainer` verifies that the names exist and are callable, but
it does not validate their return types. Follow the contract exactly.

## Runtime lifecycle and invariants

For each client, the runtime:

1. Seeds Python, NumPy, and PyTorch with `settings.seed`.
2. Calls `load_data(settings)`.
3. Calls `create_model()` and publishes its initial `state_dict`.
4. For every round, calls `train_local` unless the client has no local data.
5. Averages compatible local and neighbor states with FedAvg.
6. Calls `evaluate` on the averaged state and the client's test loader.

Consequently:

- `load_data` may establish a dataset-derived model dimension before the first
  `create_model` call, as the TON_IoT trainer does. Prefer fixed dimensions when
  possible.
- Every client and every call to `create_model` must produce the same parameter
  names, shapes, and dtypes. FedAvg rejects incompatible state dictionaries.
- Model construction must be deterministic and must not depend on a client's
  local partition size or label distribution.
- Keep all model snapshots returned by `train_local` on the CPU.
- Do not mutate `base_state` in place. Load it into a newly created model.
- The project currently trains on CPU. Do not unconditionally move models or
  batches to CUDA.

## 1. Define the model

Create an `nn.Module` whose input and output shapes match the processed data and
learning objective. Then make `create_model` return a fresh instance:

```python
class ExampleNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, CLASS_COUNT),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def create_model() -> nn.Module:
    return ExampleNetwork()
```

Choose the final layer, target dtype, loss, and prediction rule as one coherent
set:

- Multiclass classification: output shape `(batch, classes)`, integer
  `torch.long` targets, `CrossEntropyLoss`, and `argmax`.
- Binary classification: raw logits of shape `(batch,)` or `(batch, 1)`,
  floating-point targets with the same effective shape,
  `BCEWithLogitsLoss`, and `sigmoid(logits) >= 0.5`.
- Regression: output and target shapes must agree; use an appropriate
  regression loss and metrics.

Do not put a softmax before `CrossEntropyLoss` or a sigmoid before
`BCEWithLogitsLoss`; both losses expect raw logits.

## 2. Load, preprocess, cap, and partition data

`load_data` must return:

```python
DataBundle(
    train_loader=<DataLoader or None>,
    test_loader=<DataLoader>,
    sample_count=<number of examples assigned to this client>,
)
```

Follow this sequence:

1. Load or read the complete training and test data from `settings.data_dir` or
   another explicitly configured path.
2. Apply deterministic preprocessing that produces the tensor shape and dtype
   expected by the model.
3. Obtain a one-dimensional NumPy array of integer class labels for the
   training examples. `make_partitions` uses these labels for both IID and
   class-aware Dirichlet partitioning.
4. Apply `settings.max_train_samples` deterministically before partitioning.
   Preserve a mapping from the capped label positions back to original dataset
   indices when using `Subset`.
5. Call the shared `make_partitions` helper.
6. Select `partitions[settings.client_id]`.
7. Create a shuffled local training loader only when the local partition is
   nonempty.
8. Apply `settings.max_test_samples` deterministically if requested.
9. Create a non-shuffled test loader over a common evaluation set.

Use the shared partitioner as follows:

```python
partitions = make_partitions(
    labels,
    settings.client_count,
    settings.partition_mode,
    settings.seed,
    settings.dirichlet_alpha,
    min(
        settings.min_partition_size,
        max(1, len(labels) // settings.client_count // 2),
    ),
    empty_client_ids=settings.empty_client_ids,
)
```

The adaptive minimum follows the existing image trainers and keeps capped smoke
tests practical. The dataset must still contain enough samples for all
nonempty clients. `make_partitions` raises an error if it cannot satisfy the
requested partition.

For a torchvision-style dataset, preserve original indices after capping:

```python
labels = np.asarray(train_dataset.targets, dtype=np.int64)
candidate_indices = np.arange(len(labels))

if settings.max_train_samples:
    rng = np.random.default_rng(settings.seed)
    candidate_indices = rng.permutation(candidate_indices)[
        : settings.max_train_samples
    ]
    labels = labels[candidate_indices]

partitions = make_partitions(...)  # Use the full call shown above.
local_indices = candidate_indices[partitions[settings.client_id]]
train_subset = Subset(train_dataset, local_indices.tolist())
```

Create the loaders deterministically:

```python
generator = torch.Generator().manual_seed(
    settings.seed + settings.client_id
)
train_loader = None
if len(train_subset):
    train_loader = DataLoader(
        train_subset,
        batch_size=settings.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=max(settings.batch_size, 256),
    shuffle=False,
    num_workers=0,
)

return DataBundle(train_loader, test_loader, len(train_subset))
```

Important data rules:

- Pass `settings.empty_client_ids` to `make_partitions`. An intentionally empty
  client must return `train_loader=None` and `sample_count=0`; the runtime then
  skips local training safely.
- `sample_count` is the number of unique local examples, not the number of
  batches or the number of example visits across local epochs. It becomes the
  client's FedAvg weight.
- Use `settings.download_data` for dataset download behavior. Containers are
  normally expected to use data downloaded during image construction and run
  without internet access.
- Use `num_workers=0` unless the experiment has a tested reason to change it;
  this is reliable inside the existing containers.
- Fit data-dependent preprocessing without test leakage. All clients must
  nevertheless derive compatible input dimensions and semantics. For tabular
  data, fit a shared/global preprocessing schema before selecting the local
  partition, handle unseen categories, convert tensors to suitable dtypes, and
  reject missing or non-finite results when appropriate.
- The same seeded inputs must produce the same cap, split, preprocessing, and
  partition.

## 3. Implement local training

Always start from the supplied federated state:

```python
def train_local(
    base_state: StateDict,
    train_loader: DataLoader,
    settings: Settings,
) -> tuple[StateDict, dict[str, float]]:
    model = create_model()
    model.load_state_dict(base_state)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=settings.learning_rate,
        momentum=settings.momentum,
    )
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for _ in range(settings.local_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (
                outputs.argmax(dim=1) == targets
            ).sum().item()
            total_seen += batch_size

    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    metrics = {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }
    return state, metrics
```

Adapt the optimizer, criterion, and prediction rule to the task. Use
`settings.local_epochs`, `settings.learning_rate`, and, when applicable,
`settings.momentum`; do not silently hard-code configuration values already
provided by `Settings`.

Accumulate loss weighted by batch size so the final mean is per example.
Metrics must be plain finite Python `float` values suitable for JSON output,
not tensors or NumPy scalar objects. `"loss"` and `"accuracy"` are the
established keys for classification trainers. Additional scalar metrics are
allowed if needed, but keep their names stable across rounds.

The runtime never calls `train_local` for an empty client, so this function may
assume it receives a real loader.

## 4. Implement evaluation

Evaluation must load the supplied aggregated state, disable gradient tracking,
switch the model to evaluation mode, and return dataset-level means:

```python
@torch.inference_mode()
def evaluate(
    state: StateDict,
    test_loader: DataLoader,
) -> dict[str, float]:
    model = create_model()
    model.load_state_dict(state)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for inputs, targets in test_loader:
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        total_correct += (
            outputs.argmax(dim=1) == targets
        ).sum().item()
        total_seen += targets.size(0)

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }
```

Use a summed loss or explicitly multiply a mean batch loss by batch size.
Never average batch means directly when the last batch may be smaller. Do not
shuffle or train on the test loader.

## Settings available to a trainer

The commonly relevant `Settings` fields are:

| Field | Purpose |
| --- | --- |
| `client_id`, `client_count` | Select this client's partition |
| `empty_client_ids` | Clients intentionally assigned no local data |
| `seed` | Deterministic capping, partitioning, splitting, and loader order |
| `partition_mode` | `"iid"` or `"dirichlet"` |
| `dirichlet_alpha` | Dirichlet label-skew concentration |
| `min_partition_size` | Requested minimum nonempty partition size |
| `data_dir` | Dataset storage location |
| `download_data` | Whether the dataset loader may download |
| `max_train_samples`, `max_test_samples` | Optional deterministic sample caps; `0` means unlimited |
| `batch_size` | Local training batch size |
| `local_epochs` | Local passes per decentralized round |
| `learning_rate`, `momentum` | Optimizer configuration |
| `results_dir` | Optional experiment metadata/artifact output location |

If a new dataset needs a new environment setting, add a typed field, environment
mapping, and validation to `app/config.py`; document it and update relevant
tests and example environment files. Do this only when the existing settings
cannot express the requirement.

## Experiment packaging

After `trainer.py` works:

1. Add `experiments/<experiment_name>/__init__.py`.
2. Copy an existing experiment Dockerfile and update:
   - dependency installation,
   - dataset pre-download/preparation,
   - `TRAINER_MODULE`,
   - any dataset-specific environment variables or copied files.
3. Copy and update an existing `compose.yaml` when a Compose-based experiment
   is needed.
4. Keep generated datasets, checkpoints, `.env` files, and results out of Git.
5. Ensure every client image contains the same trainer code and dataset schema.

## Required tests and validation

Add focused tests under `tests/`, normally in `tests/test_trainers.py` or a new
`tests/test_<experiment_name>.py`.

At minimum, verify:

- `load_trainer("experiments.<experiment_name>.trainer")` succeeds.
- A synthetic batch has the expected input/output shape.
- `train_local` accepts `create_model().state_dict()` and returns a compatible
  state dictionary.
- Every returned state tensor is on the CPU.
- Training and evaluation return the expected finite metric keys and values.
- Data capping is deterministic.
- IID and Dirichlet partitions select valid examples.
- An ID in `empty_client_ids` yields `train_loader=None` and `sample_count=0`.
- `sample_count` equals the local dataset size.
- Dataset-specific missing files, invalid columns, labels, shapes, or
  non-finite values fail with clear errors.

Use a tiny synthetic dataset or capped samples for tests; do not require a
network download.

Run:

```powershell
python -m pytest
python -m compileall -q app tests scripts experiments
```

For a quick end-to-end container smoke test, use one round, one local epoch,
small train/test caps, and then monitor completion with
`python scripts/wait_for_run.py`.

## Final implementation checklist

- [ ] The experiment directory is an importable package.
- [ ] All four required callables exist with compatible signatures.
- [ ] `create_model` always produces an identical federatable state schema.
- [ ] Preprocessing matches model input shape, target dtype, loss, and metrics.
- [ ] Capping, splitting, partitioning, and loader shuffling are deterministic.
- [ ] `make_partitions` receives all relevant settings, including
      `empty_client_ids`.
- [ ] Empty clients return no training loader and a zero sample count.
- [ ] `sample_count` is the number of unique local examples.
- [ ] Training begins from `base_state` and returns detached CPU tensors.
- [ ] Evaluation uses the supplied aggregated state and inference mode.
- [ ] Losses and metrics are aggregated per example and returned as Python
      floats.
- [ ] Unit tests, compile checks, and a small smoke test pass.
