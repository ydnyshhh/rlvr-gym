# Sokoban Benchmark Card

## Capability

This family measures long-horizon spatial planning under delayed consequences. Agents must reason over exact deterministic box-pushing dynamics, avoid irreversible mistakes, and manage local decisions that affect future reachability.

## Formal objective

- World: a static board with walls and goals plus a dynamic player position and box positions
- Action space: `move_up`, `move_down`, `move_left`, `move_right`
- Success condition: every box occupies a goal cell
- Primary quality target: minimize primitive move count
- Secondary analysis target: compare push count to the oracle plan

## Verification guarantees

- Step legality: every primitive move is checked against exact Sokoban transition rules
- State validity: successor states must preserve wall, floor, player, and box consistency exactly
- Deadlock diagnostics: static dead-square detection flags known impossible box placements
- Goal verification: terminal success is exact and depends only on the final box-goal configuration
- Trajectory verification: completion, deadlock-free execution, and efficiency relative to the oracle are all reported

## Oracle quality

The oracle is an exact A* planner over primitive Sokoban states. It returns a certified feasible and optimal primitive-move plan for the sampled board, along with push count and search-expansion metadata.

## Metrics

- success rate
- feasibility score
- quality score
- invalid action rate
- move count
- push count
- move-count gap to oracle
- deadlock rate

## Recommended ID split

- moderate templates
- 2 boxes
- medium reverse-scramble depth
- bounded oracle solution length

## Recommended OOD splits

- more boxes: increase interaction between boxes and goal assignments
- longer plans: deeper reverse scrambles and longer oracle solutions
- larger warehouse templates: wider boards with more corridor structure
- deadlock-prone layouts: templates with more taboo cells and constrained geometry

## Known shortcut risks

- always-push heuristics can look good locally while creating irreversible deadlocks
- greedy distance-to-goal heuristics can fail in narrow corridors
- policies that ignore player positioning may underestimate the true planning burden

## Baseline heuristics

- random valid move
- push-when-possible
- greedy goal-progress heuristic
- deadlock-avoiding greedy
- oracle planner
