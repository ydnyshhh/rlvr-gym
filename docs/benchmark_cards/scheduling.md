# Scheduling Benchmark Card

## Artifact bundle

- Spec: [../benchmark_artifacts/scheduling/benchmark_spec.json](../benchmark_artifacts/scheduling/benchmark_spec.json)
- Split manifest: [../benchmark_artifacts/scheduling/split_manifest.json](../benchmark_artifacts/scheduling/split_manifest.json)
- Diversity summary: [../benchmark_artifacts/scheduling/diversity_summary.md](../benchmark_artifacts/scheduling/diversity_summary.md)
- Baseline table: [../benchmark_artifacts/scheduling/baseline_results.md](../benchmark_artifacts/scheduling/baseline_results.md)

## Capability

This family measures constrained sequential decision making under precedence structure, deadlines, and cumulative objective tradeoffs.

It is most useful for studying:

- planning with delayed quality signals
- legality under precedence constraints
- sequence construction under combinatorial action spaces
- feasibility versus schedule quality decomposition

## Formal objective

The agent is given a set of jobs with durations, deadlines, and precedence constraints.

At each step it must choose a legal ready job via the canonical `schedule` action. The task objective is to produce a complete valid schedule with minimum total tardiness.

## Verification guarantees

Step-level guarantees:

- the selected job must be ready and unscheduled
- the next schedule state must be time-consistent
- completed-job ordering must remain valid

Trajectory-level guarantees:

- all jobs must be scheduled exactly once
- all precedence constraints must be satisfied
- total tardiness is checked against an exact dynamic-programming oracle

## Oracle quality

Oracle quality is defined by exact optimal total tardiness.

The oracle returns:

- a valid complete job order
- the optimal objective value
- a proof-style certificate that the schedule was computed by exact subset dynamic programming

## Metrics to report

- completion rate
- invalid action rate
- feasibility score
- quality score
- total tardiness
- optimality gap versus oracle
- average number of scheduled jobs before failure or truncation
- fraction of oracle-optimal schedules

## Recommended ID split

Use disjoint seed sets while holding the generation regime fixed.

Recommended ID protocol:

- same difficulty bucket
- same job-count range
- same duration range
- same precedence probability
- disjoint deterministic seed sets for train, validation, and test

## Recommended OOD splits

Recommended OOD evaluations:

- more jobs than seen during training
- tighter deadlines
- denser precedence DAGs
- altered duration distributions
- changed reward modes from feasibility-heavy to quality-heavy evaluation

## Known shortcut risks

- favoring earliest-ready-job heuristics that look strong only on shallow DAGs
- overfitting to low-constraint regimes
- optimizing local tardiness greedily instead of global schedule quality
- exploiting reward shaping without learning long-horizon precedence reasoning

## Baseline heuristics

- random ready-job selection
- shortest-processing-time first among ready jobs
- earliest-deadline-first among ready jobs
- minimum slack heuristic among ready jobs
- exact oracle schedule
