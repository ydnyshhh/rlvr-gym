# Symbolic Transformation Benchmark Card

## Capability

This family measures formal step-by-step symbolic reasoning over structured expression trees.

It is most useful for studying:

- exact symbolic manipulation rather than free-form answer generation
- compositional reasoning depth
- interpretable trajectory construction
- separation between legal reasoning steps and superficial pattern imitation

The family currently includes two task modes:

- arithmetic simplification to canonical form
- boolean normalization to negation normal form

## Formal objective

The latent world is a symbolic expression tree and a target transformation objective.

The agent acts by issuing canonical `rewrite` actions that specify:

- which rewrite rule to apply
- which subtree path to apply it to

The task objective is to transform the source symbolic object into the target form by a sequence of legal local rewrites.

## Verification guarantees

Step-level guarantees:

- the rewrite rule must be valid for the selected subtree
- the subtree path must point to the correct location
- the resulting expression tree must exactly match the legal rewritten tree

Trajectory-level guarantees:

- the final expression must satisfy the target normal-form condition
- the final expression must remain semantically equivalent to the source expression
- trajectory quality is measured against the shortest oracle rewrite sequence

## Oracle quality

Oracle quality is defined by shortest-path optimality in the explicit rewrite graph.

The oracle returns:

- a legal rewrite sequence
- the minimum number of rewrite steps needed to reach the target form
- a proof-style certificate indicating that the sequence is shortest under the explicit rewrite system

## Metrics to report

- success rate
- invalid rewrite rate
- feasibility score
- quality score
- final structural distance to target
- rewrite-step count
- optimality gap in steps versus oracle
- fraction of oracle-minimal trajectories

Mode-specific useful metrics:

- arithmetic: canonical-form success and semantic-equivalence success
- boolean: NNF success and semantic-equivalence success

## Recommended ID split

Use fixed task-mode and generator regimes with disjoint seed sets.

Recommended ID protocol:

- same task mode
- same target-depth range
- same inverse-step corruption range
- same variable-count range
- disjoint deterministic seed sets for train, validation, and test

## Recommended OOD splits

Recommended OOD evaluations:

- deeper expression trees
- longer oracle rewrite sequences
- switching from arithmetic-only training to boolean evaluation, or vice versa
- more variables than seen in training
- altered rewrite-rule mixtures or target normal forms

## Known shortcut risks

- memorizing common surface rewrite motifs instead of modeling tree-local legality
- learning to imitate frequent rule IDs without understanding subtree selection
- overfitting to short trajectories with repeated identity-removal patterns
- exploiting local progress signals while failing on deeper compositional rewrites

## Baseline heuristics

- random valid rewrite
- greedy rewrite minimizing structural distance to target
- greedy rewrite minimizing expression size
- rule-priority heuristic for obvious local simplifications
- shortest-path oracle over the rewrite graph
