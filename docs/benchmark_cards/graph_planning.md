# Graph Planning Benchmark Card

## Artifact bundle

- Spec: [../benchmark_artifacts/graph_planning/benchmark_spec.json](../benchmark_artifacts/graph_planning/benchmark_spec.json)
- Split manifest: [../benchmark_artifacts/graph_planning/split_manifest.json](../benchmark_artifacts/graph_planning/split_manifest.json)
- Diversity summary: [../benchmark_artifacts/graph_planning/diversity_summary.md](../benchmark_artifacts/graph_planning/diversity_summary.md)
- Baseline table: [../benchmark_artifacts/graph_planning/baseline_results.md](../benchmark_artifacts/graph_planning/baseline_results.md)

## Capability

This family measures sequential planning over structured graph worlds with exact local legality and globally verifiable path quality.

It is most useful for studying:

- shortest-path planning under structured action spaces
- trajectory construction rather than one-shot answer prediction
- feasibility versus optimality tradeoffs
- generalization across graph size, density, and edge-cost structure

## Formal objective

The agent is given a weighted graph, a start node, and a goal node.

At each step it must choose a legal outgoing edge by selecting a canonical `move` action. The task objective is to reach the goal while minimizing cumulative path cost.

## Verification guarantees

Step-level guarantees:

- the chosen action must correspond to a legal outgoing edge
- the next state must match the exact graph transition
- the path state and cost accounting must remain internally consistent

Trajectory-level guarantees:

- successful trajectories must terminate at the goal
- trajectory legality is checked against the graph structure
- optimality is checked against an exact shortest-path oracle

## Oracle quality

Oracle quality is defined by exact shortest-path optimality.

The oracle returns:

- a legal path
- the certified optimal path cost
- a proof-style certificate indicating that the shortest path was computed exactly

## Metrics to report

- success rate
- invalid action rate
- feasibility score
- quality score
- final path cost
- excess path cost over oracle optimum
- average steps to goal
- fraction of oracle-optimal trajectories

## Recommended ID split

Use a fixed generation configuration for train, validation, and test while changing only task seeds.

Recommended ID protocol:

- same difficulty bucket
- same graph size range
- same edge-density range
- same reward mode
- disjoint deterministic seed sets for train, validation, and test

## Recommended OOD splits

Recommended OOD evaluations:

- larger graphs than seen in training
- denser or sparser graphs than seen in training
- higher edge-cost variance
- directed graphs after training primarily on undirected graphs
- altered observability settings

## Known shortcut risks

- myopic goal-reaching without cost minimization
- exploiting dense shaping while ignoring optimality
- memorizing local motifs if graph distributions are too narrow
- overfitting to small graph diameter regimes

## Baseline heuristics

- random valid walk
- greedy lowest-edge-cost next move
- greedy next move with minimum remaining heuristic distance
- oracle shortest path
- one-step feasibility-only policy as an ablation
