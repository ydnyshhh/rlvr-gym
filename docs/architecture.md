# Architecture

RLVR-Gym is organized around the idea that an environment family defines a distribution over formal task instances rather than a single benchmark item.

## Generation pipeline

1. Sample family generation parameters from difficulty and config.
2. Generate a latent symbolic world.
3. Derive a concrete task objective.
4. Choose observability and rendering behavior.
5. Attach verifier suites and reward configuration.
6. Optionally attach an oracle or solver.
7. Build a runnable runtime environment and exportable task spec.

## Core abstractions

`EnvironmentFamily`

- owns latent world generation
- defines transition dynamics
- exposes canonical actions and structured observations
- attaches family-specific verifier suites and optional oracles

`TaskInstance`

- bundles one sampled world, objective, runtime hooks, verifier suite, reward engine, and metadata

`RLVREnv`

- exposes `reset()` and `step()` in a Gym-style API
- records traces suitable for export and analysis

`VerifierSuite`

- supports action-level verification
- supports state-level verification
- supports goal-level verification
- supports trajectory-level verification
- separates feasibility checks from quality checks
- supports weighted composition and hard-fail semantics through the report layer

`RewardEngine`

- supports sparse, dense, and shaped reward modes
- also supports feasibility-focused, quality-aware, and hybrid modes
- consumes verifier outputs and transition hints

## Built-in families

### Deduction grid

- latent world: a hidden bijective assignment across entity and attribute categories plus a generated clue set
- objective: recover the full assignment by exact symbolic table updates and a final solution commit
- observation: exposes clues plus the agent-built deduction table, while hiding internal closure-derived pending updates and resolved assignments
- oracle: deterministic closure-driven deduction policy ending in a formally checked complete assignment
- propagation semantics: `propagate` performs one deterministic local closure update rather than collapsing the entire table in a single macro-step
- verification: action legality, table-update correctness, final assignment validity, trajectory efficiency

### Graph planning

- latent world: weighted graph with guaranteed start-goal connectivity
- objective: reach the goal with minimum path cost
- oracle: exact shortest path
- verification: legality, state consistency, goal completion, trajectory optimality

### Scheduling

- latent world: single-machine job set with deadlines and precedence DAG
- objective: schedule all jobs while minimizing total tardiness
- oracle: exact dynamic programming over job subsets
- verification: action readiness, schedule consistency, completion, precedence, optimality

## Research uses

- RLVR and online RL with on-reset procedural generation
- frozen offline datasets through oracle rollouts
- benchmark split generation with deterministic seeds
- reward shaping studies by changing reward mode and weights
- OOD generalization by altering difficulty and generator overrides
- pathology analysis through generated instances and verifier traces

## Benchmark cards

Per-family benchmark cards live under [benchmark_cards](benchmark_cards/README.md).
Generated benchmark artifacts live under [benchmark_artifacts](benchmark_artifacts/README.md).

They are intended to be the research-facing summary for each family, including:

- capability targets
- formal objectives
- verification guarantees
- oracle interpretation
- metrics
- recommended ID and OOD splits
- shortcut risks
- baseline heuristics

## Current maturity

RLVR-Gym currently proves the framework more than it proves environment breadth. The shipped families are intentionally narrow, while the core abstractions are designed to support a wider future suite of verifiable reasoning and decision domains.
