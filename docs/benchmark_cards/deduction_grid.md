# Deduction Grid Benchmark Card

## Artifact bundle

- Spec: [../benchmark_artifacts/deduction_grid/benchmark_spec.json](../benchmark_artifacts/deduction_grid/benchmark_spec.json)
- Split manifest: [../benchmark_artifacts/deduction_grid/split_manifest.json](../benchmark_artifacts/deduction_grid/split_manifest.json)
- Diversity summary: [../benchmark_artifacts/deduction_grid/diversity_summary.md](../benchmark_artifacts/deduction_grid/diversity_summary.md)
- Baseline table: [../benchmark_artifacts/deduction_grid/baseline_results.md](../benchmark_artifacts/deduction_grid/baseline_results.md)

## Capability

This family measures formal relational deduction over a structured logic grid.

It is useful for studying:

- multi-step elimination and constraint propagation
- maintaining global consistency while making local symbolic updates
- separating legal deductions from unsupported guesses
- interpretable reasoning traces over an explicit symbolic table

## Formal objective

The latent world contains:

- a base entity category
- multiple attribute categories of equal size
- a hidden bijective assignment across categories
- a generated set of formal clues derived from that assignment

The agent interacts with a deduction table using canonical actions:

- `assert_pair`
- `rule_out_pair`
- `propagate`
- `commit_solution`

The public observation exposes the clue set, the current explicit table state, and recorded true/false facts. It does not expose the internally derived closure state, pending oracle deductions, or the hidden resolved assignment before commit.

`propagate` is a local deterministic closure action. It applies one verifier-checkable deduction-table update at a time rather than collapsing the full puzzle in a single step.

The task is solved when the agent commits a complete assignment that satisfies every clue and all bijection constraints.

## Verification guarantees

Step-level guarantees:

- asserted pairings must be entailed by the current closure state
- ruled-out pairings must be entailed eliminations
- propagation must apply exactly the currently derivable updates
- final commits must be structurally well formed

Trajectory-level guarantees:

- the deduction table must remain globally consistent
- the committed assignment must satisfy all clue semantics
- the committed assignment must satisfy bijection constraints in every category
- trajectory quality is measured against the deterministic oracle plan length

## Oracle quality

Oracle quality is defined relative to the built-in closure-driven deduction policy.

The oracle returns:

- a legal symbolic deduction trace
- a complete final assignment
- a proof-style certificate describing the deduction policy and puzzle size

The oracle is exact with respect to the family’s formal deduction semantics and final-solution verification.
The benchmark generator also rejects strongly propagation-collapsed puzzle instances, so trivial one- or two-step closure policies are less likely to dominate the benchmark product layer.

## Metrics to report

- success rate
- invalid action rate
- feasibility score
- quality score
- number of deduction steps
- objective gap in steps versus oracle
- final assignment validity rate
- clue-satisfaction rate on committed solutions

## Recommended ID split

Use fixed category-count and entity-count regimes with disjoint deterministic seeds.

Recommended ID protocol:

- same number of entities
- same number of relation categories
- same clue-generation bias
- disjoint train, validation, and test seed sets

## Recommended OOD splits

Recommended OOD evaluations:

- more entities than seen during training
- more relation categories than seen during training
- clue sets with a higher ratio of relational clues to direct clues
- more distractor clues
- longer oracle traces

## Known shortcut risks

- memorizing frequent local clue motifs instead of modeling full-table consistency
- overusing `propagate` without learning when specific assertions are justified
- exploiting direct clues while failing on relational clue chains
- learning category-specific surface patterns rather than general bijection reasoning

## Baseline heuristics

- random valid action
- propagate-first heuristic
- assert-first heuristic
- rule-out-first heuristic
- deterministic closure oracle
