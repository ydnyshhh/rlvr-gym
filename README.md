# RLVR-Gym

RLVR-Gym is a Python library for generating formal, verifiable reasoning and decision environments.

It is designed as a task factory: each environment family defines a distribution over related MDPs or POMDPs, and each sampled task instance yields:

- a latent world model
- a concrete objective
- a runnable environment
- structured observations
- canonical actions
- verifier suites
- configurable reward computation
- optional exact oracle support
- exportable task specifications and traces

RLVR-Gym is built for research workflows such as RLVR, online RL, offline RL, SFT data generation, rejection sampling, and benchmark construction.

## Core ideas

- Formal structure first: tasks come from explicit symbolic or algorithmic generators.
- Verification first: correctness is computed from state and rules rather than surface heuristics.
- Families over one-offs: each family defines a reusable distribution with reproducible difficulty control.
- Typed and modular: core interfaces are schema-driven and designed for extension.
- Research-ready exports: the same family can support online interaction, supervised targets, offline datasets, and benchmark splits.

## Included families

- `graph_planning`: weighted graph navigation with exact shortest-path verification
- `scheduling`: single-machine scheduling with deadlines, precedence constraints, and exact tardiness optimization

## Installation

RLVR-Gym uses `uv` for environment management and local development.

### Prerequisites

Before installing, make sure you have:

- Python 3.10 or newer
- `git`
- `uv`

You can verify your Python version with:

```bash
python --version
```

### Setup with `uv`

1. Clone the repository:

```bash
git clone https://github.com/ydnyshhh/rlvr-gym.git
cd rlvr-gym
```

2. Sync the project environment:

```bash
uv sync
```

This will:

- create a local virtual environment if needed
- install the package in the project environment
- resolve and install dependencies from the project configuration

3. Verify that the CLI works:

```bash
uv run rlvr-gym list-families
```

4. Run the test suite:

```bash
uv run python -B -m unittest discover -s tests -v
```

### Common first-run commands

After installation, these are good first checks:

List built-in families:

```bash
uv run rlvr-gym list-families
```

Sample a graph-planning task:

```bash
uv run rlvr-gym sample --family graph_planning --seed 7 --export task
```

Sample a scheduling oracle rollout:

```bash
uv run rlvr-gym sample --family scheduling --seed 11 --export oracle
```

### Troubleshooting

If the `rlvr-gym` command is not found:

- use `uv run rlvr-gym ...`
- make sure you are running commands from the repository root

If tests fail immediately because the environment is not active:

- rerun through `uv run ...`

## Quick start

```python
from rlvr_gym import FamilyConfig, export_task_spec, get_family, rollout_oracle

family = get_family("graph_planning")
task = family.sample_instance(seed=42, config=FamilyConfig(difficulty="medium"))

print(export_task_spec(task))
print(rollout_oracle(task))

env = family.create_env(seed=42, config=FamilyConfig(difficulty="medium"))
observation, info = env.reset()
print(observation)
print(info["valid_actions"])
```

## Environment model

Each family follows the same generation flow:

1. Sample generation parameters from config and difficulty.
2. Generate a latent symbolic world.
3. Derive a concrete objective.
4. Define observability and rendering.
5. Attach verifier configuration.
6. Attach reward configuration.
7. Optionally attach an oracle or solver.
8. Build a runnable runtime and exportable task spec.

This supports both:

- dynamic online generation at reset time
- deterministic offline generation for frozen benchmarks

## Runtime and exports

RLVR-Gym provides a lightweight Gym-like `reset()` / `step()` runtime through `RLVREnv`.

Available export paths:

- Online RL: interact directly with the runtime
- SFT: `export_sft_example(task)`
- Offline RL: `export_offline_transitions(task)`
- Oracle label views: `export_oracle_views(task)`
- Benchmark generation: `build_benchmark_splits(...)` and `write_benchmark_splits(...)`

## CLI

List available families:

```bash
uv run rlvr-gym list-families
```

Sample a task:

```bash
uv run rlvr-gym sample --family graph_planning --seed 7 --export task
```

Export an oracle rollout:

```bash
uv run rlvr-gym sample --family scheduling --seed 11 --export oracle
```

Export decomposed oracle labels:

```bash
uv run rlvr-gym sample --family scheduling --seed 11 --export labels
```

Generate deterministic benchmark splits:

```bash
uv run rlvr-gym benchmark --family scheduling --base-seed 123 --train-count 100 --validation-count 20 --test-count 20 --output-dir benchmarks
```

## Package layout

- [src/rlvr_gym/core/family.py](src/rlvr_gym/core/family.py): family abstraction and seeded task generation
- [src/rlvr_gym/core/runtime.py](src/rlvr_gym/core/runtime.py): runtime loop and trace recording
- [src/rlvr_gym/core/verifier.py](src/rlvr_gym/core/verifier.py): feasibility and quality verification
- [src/rlvr_gym/core/reward.py](src/rlvr_gym/core/reward.py): reward configuration and computation
- [src/rlvr_gym/core/exporters.py](src/rlvr_gym/core/exporters.py): task, rollout, SFT, offline RL, and benchmark exports
- [src/rlvr_gym/core/oracle.py](src/rlvr_gym/core/oracle.py): oracle interface and proof-style solution metadata
- [src/rlvr_gym/families/graph_planning.py](src/rlvr_gym/families/graph_planning.py): graph planning family
- [src/rlvr_gym/families/scheduling.py](src/rlvr_gym/families/scheduling.py): scheduling family

## Testing

```bash
uv run python -B -m unittest discover -s tests -v
```
