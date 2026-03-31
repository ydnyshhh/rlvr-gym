# RLVR-Gym

RLVR-Gym is a research-oriented Python package for generating formal, verifiable reasoning and decision environments. It is a task factory, not a trainer and not a static benchmark collection.

Status: RLVR-Gym is currently an alpha research framework. The core abstractions are in place and the initial families are intentionally narrow, serving as proof that the generation, verification, reward, oracle, and export stack works end to end. The project is not yet a broad benchmark suite.

Each environment family defines a distribution over related MDPs or POMDPs. A sampled task instance yields:

- a latent world model
- a concrete objective
- a runnable Gym-like environment API
- structured observations
- canonical actions
- verifier suites
- configurable reward computation
- optional exact oracle support
- exportable task specifications and traces

## Design goals

- Formal structure first: tasks come from explicit symbolic or algorithmic generators.
- Verification first: correctness is computable from state and rules.
- Families over one-offs: each family defines a reusable distribution with difficulty knobs.
- Typed and modular: strong schemas and clear interfaces drive extensibility.
- Research-ready: suitable for RLVR, online RL, offline RL, SFT, DPO-style data generation, rejection sampling, and benchmark creation.

## Included families

- `graph_planning`: weighted shortest-path navigation on generated graphs
- `scheduling`: single-machine scheduling with deadlines and precedence constraints

These are initial families rather than a claim of full environment breadth. They demonstrate the framework mechanics today; broader reasoning families are still a roadmap item.

## Quick start

Install and sync with `uv`:

```bash
uv sync
```

Run Python entrypoints with `uv`:

```bash
uv run python examples/basic_usage.py
```

```bash
uv run rlvr-gym list-families
```

Fallback editable install with `pip`:

```bash
python -m pip install -e .
```

```python
from rlvr_gym import FamilyConfig, export_task_spec, get_family, rollout_oracle

family = get_family("graph_planning")
task = family.sample_instance(seed=42, config=FamilyConfig(difficulty="medium"))

spec = export_task_spec(task)
oracle_rollout = rollout_oracle(task)

env = family.create_env(seed=42, config=FamilyConfig(difficulty="medium"))
observation, info = env.reset()
next_observation, reward, terminated, truncated, step_info = env.step(
    {"name": "move", "arguments": {"target": info["valid_actions"][0]["arguments"]["target"]}}
)
```

## CLI

List families:

```bash
uv run rlvr-gym list-families
```

or:

```bash
rlvr-gym list-families
```

Sample a task:

```bash
uv run rlvr-gym sample --family graph_planning --seed 7 --export task
```

or:

```bash
rlvr-gym sample --family graph_planning --seed 7 --export task
```

Export a deterministic benchmark split:

```bash
uv run rlvr-gym benchmark --family scheduling --base-seed 123 --train-count 100 --validation-count 20 --test-count 20 --output-dir benchmarks
```

or:

```bash
rlvr-gym benchmark --family scheduling --base-seed 123 --train-count 100 --validation-count 20 --test-count 20 --output-dir benchmarks
```

## Package layout

- [src/rlvr_gym/core/family.py](src/rlvr_gym/core/family.py): family abstraction and seeded task generation
- [src/rlvr_gym/core/runtime.py](src/rlvr_gym/core/runtime.py): Gym-like runtime API
- [src/rlvr_gym/core/verifier.py](src/rlvr_gym/core/verifier.py): weighted feasibility and quality verification
- [src/rlvr_gym/core/reward.py](src/rlvr_gym/core/reward.py): configurable reward engine with richer semantics
- [src/rlvr_gym/core/exporters.py](src/rlvr_gym/core/exporters.py): task, SFT, offline RL, and benchmark exports
- [src/rlvr_gym/families/graph_planning.py](src/rlvr_gym/families/graph_planning.py): graph planning family
- [src/rlvr_gym/families/scheduling.py](src/rlvr_gym/families/scheduling.py): scheduling family

## Export modes

- Online RL: use `RLVREnv.reset()` and `RLVREnv.step()`
- SFT: use `export_sft_example(task)`
- Offline RL: use `export_offline_transitions(task)`
- Benchmark splits: use `build_benchmark_splits(...)` and `write_benchmark_splits(...)`

`RLVREnv` is Gym-like and intentionally lightweight. It matches the familiar reset/step loop researchers expect, but it is not yet a formal Gymnasium adapter with dependency-level integration.

## Current limitations

- The built-in family set is still narrow and should be read as initial coverage, not benchmark breadth.
- The package favors a lightweight standard-library-first core and does not yet depend on heavier solver libraries.
- Richer families, more solver-backed domains, and benchmark breadth remain future work.

## Testing

```bash
uv run python -B -m unittest discover -s tests -v
```

or:

```bash
python -B -m unittest discover -s tests -v
```
