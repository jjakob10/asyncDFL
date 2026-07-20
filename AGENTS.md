# Repository Guidelines

## Project Structure & Module Organization

Application code lives in `app/`. `main.py` defines the FastAPI surface, while
`runtime.py` coordinates start propagation and decentralized rounds.
`training.py` defines the pluggable trainer contract and shared partitioning;
`federation.py` handles model serialization and FedAvg; `config.py` maps
environment variables into validated settings. Dataset/model implementations
and their container definitions live under `experiments/<name>/`.

Tests are under `tests/` and mirror these responsibilities. Operational helpers
live in `scripts/`, container definitions are in `Dockerfile` and
`compose.yaml`, and generated experiment artifacts belong in the ignored
`results/` directory. Presentation and research deliverables are stored in
`outputs/`.

## Build, Test, and Development Commands

- `python -m venv .venv` creates an isolated Python environment.
- `python -m pip install -r requirements-dev.txt` installs runtime and test
  dependencies.
- `python -m pytest` runs the complete quiet pytest suite configured by
  `pytest.ini`.
- `docker compose -f experiments/mnist/compose.yaml build` builds the CPU-only
  MNIST image; substitute `cifar10` to build that experiment.
- `docker compose -f experiments/mnist/compose.yaml up -d` starts the
  eight-client ring; use
  `python scripts/wait_for_run.py` to monitor completion.
- `docker compose -f experiments/mnist/compose.yaml down` removes the experiment
  containers and network.

Copy an experiment's `.env.example` to `.env` to override rounds, partition
mode, sample caps, or request behavior. Use capped samples and one round for
quick smoke tests.

## Coding Style & Naming Conventions

Use Python 3.11-compatible code, four-space indentation, type hints, and concise
docstrings where behavior is not obvious. Follow PEP 8 naming: `snake_case` for
functions and variables, `PascalCase` for classes, and uppercase environment
variables. Keep model snapshots on CPU and preserve locking around published
runtime state. No formatter or linter is currently mandated; keep imports
ordered and run `python -m compileall -q app tests scripts` before submission.

## Testing Guidelines

Use pytest and name files `test_<area>.py` and functions `test_<behavior>`.
Cover success and failure paths, especially idempotent starts, ring loops,
unavailable neighbors, deterministic partitions, model compatibility, and
atomic version updates. No numeric coverage threshold is configured; new logic
should have focused regression tests.

## Commit & Pull Request Guidelines

The repository has no commits yet, so no historical convention exists. Use
short imperative commits following Conventional Commits conventions such as `feat: Add Dirichlet partition validation`. Pull
requests should explain behavioral changes, list validation commands, identify
configuration changes, and include relevant logs or metric excerpts. Link the
research issue when applicable; screenshots are only needed for visual
deliverables.

## Security & Configuration

REST model exchange is trusted plaintext. Never commit secrets, `.env`, MNIST
data, model checkpoints, or `results/`. Treat downloaded model payloads as
trusted only within the isolated experiment network.
