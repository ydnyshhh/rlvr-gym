from __future__ import annotations

import heapq
import random
from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from rlvr_gym.core.family import EnvironmentFamily
from rlvr_gym.core.oracle import Oracle, OracleSolution, ProofCertificate
from rlvr_gym.core.types import CanonicalAction, FamilyConfig, TaskObjective, TaskSpace, TransitionResult
from rlvr_gym.core.verifier import (
    BaseVerifier,
    StepContext,
    TrajectoryContext,
    VerificationKind,
    VerificationResult,
    VerificationScope,
    VerifierSuite,
)

Coord = tuple[int, int]

ACTION_DELTAS: dict[str, Coord] = {
    "move_up": (-1, 0),
    "move_down": (1, 0),
    "move_left": (0, -1),
    "move_right": (0, 1),
}


@dataclass(frozen=True)
class SokobanTemplate:
    name: str
    rows: tuple[str, ...]


@dataclass(frozen=True)
class SokobanWorld:
    template_name: str
    width: int
    height: int
    walls: tuple[Coord, ...]
    goals: tuple[Coord, ...]
    initial_boxes: tuple[Coord, ...]
    initial_player: Coord
    taboo_cells: tuple[Coord, ...]
    reverse_scramble_steps: int
    oracle_plan: tuple[str, ...] = ()
    oracle_move_count: int = 0
    oracle_push_count: int = 0
    solver_expansions: int = 0


@dataclass(frozen=True)
class SokobanState:
    player: Coord
    boxes: tuple[Coord, ...]
    solved: bool
    deadlock: bool
    move_count: int = 0
    push_count: int = 0


@dataclass(frozen=True)
class SolverResult:
    actions: tuple[str, ...]
    move_count: int
    push_count: int
    expanded_nodes: int


@dataclass(frozen=True)
class ReverseActionCandidate:
    action_type: str
    next_player: Coord
    next_boxes: tuple[Coord, ...]
    moved_box: bool


TEMPLATES: tuple[SokobanTemplate, ...] = (
    SokobanTemplate(
        name="compact_cross",
        rows=(
            "########",
            "#      #",
            "# ## # #",
            "#      #",
            "# # ## #",
            "#      #",
            "#      #",
            "########",
        ),
    ),
    SokobanTemplate(
        name="side_rooms",
        rows=(
            "########",
            "#   #  #",
            "#      #",
            "# ##   #",
            "#   ## #",
            "#      #",
            "#  #   #",
            "########",
        ),
    ),
    SokobanTemplate(
        name="warehouse_large",
        rows=(
            "#########",
            "#       #",
            "# ### # #",
            "#       #",
            "# # ### #",
            "#       #",
            "# ###   #",
            "#       #",
            "#########",
        ),
    ),
    SokobanTemplate(
        name="rooms_and_halls",
        rows=(
            "#########",
            "#   #   #",
            "#       #",
            "### # ###",
            "#       #",
            "# ###   #",
            "#   ### #",
            "#       #",
            "#########",
        ),
    ),
)


def _coord_add(left: Coord, right: Coord) -> Coord:
    return (left[0] + right[0], left[1] + right[1])


def _coord_sub(left: Coord, right: Coord) -> Coord:
    return (left[0] - right[0], left[1] - right[1])


def _sorted_coords(coords: set[Coord] | tuple[Coord, ...] | list[Coord]) -> tuple[Coord, ...]:
    return tuple(sorted(coords))


def _wall_set(world: SokobanWorld) -> set[Coord]:
    return set(world.walls)


def _goal_set(world: SokobanWorld) -> set[Coord]:
    return set(world.goals)


def _box_set(state: SokobanState) -> set[Coord]:
    return set(state.boxes)


def _all_floor_cells(width: int, height: int, walls: set[Coord]) -> set[Coord]:
    return {
        (row_index, column_index)
        for row_index in range(height)
        for column_index in range(width)
        if (row_index, column_index) not in walls
    }


def _render_ascii(
    width: int,
    height: int,
    walls: tuple[Coord, ...],
    goals: tuple[Coord, ...],
    player: Coord | None = None,
    boxes: tuple[Coord, ...] | None = None,
) -> str:
    goal_cells = set(goals)
    box_cells = set(boxes or ())
    wall_cells = set(walls)
    rows: list[str] = []
    for row_index in range(height):
        row_chars: list[str] = []
        for column_index in range(width):
            cell = (row_index, column_index)
            if cell in wall_cells:
                row_chars.append("#")
            elif player == cell and cell in goal_cells:
                row_chars.append("+")
            elif player == cell:
                row_chars.append("@")
            elif cell in box_cells and cell in goal_cells:
                row_chars.append("*")
            elif cell in box_cells:
                row_chars.append("$")
            elif cell in goal_cells:
                row_chars.append(".")
            else:
                row_chars.append(" ")
        rows.append("".join(row_chars))
    return "\n".join(rows)


def _parse_template(rows: tuple[str, ...]) -> tuple[int, int, tuple[Coord, ...], tuple[Coord, ...]]:
    height = len(rows)
    width = len(rows[0])
    walls: list[Coord] = []
    floors: list[Coord] = []
    for row_index, row in enumerate(rows):
        for column_index, char in enumerate(row):
            if char == "#":
                walls.append((row_index, column_index))
            else:
                floors.append((row_index, column_index))
    return width, height, _sorted_coords(walls), _sorted_coords(floors)


def _transform_rows(rows: tuple[str, ...], variant: str) -> tuple[str, ...]:
    if variant == "flip_horizontal":
        return tuple(row[::-1] for row in rows)
    if variant == "flip_vertical":
        return tuple(reversed(rows))
    if variant == "flip_both":
        return tuple(row[::-1] for row in reversed(rows))
    return rows


def _adjacent_floor_degree(cell: Coord, floors: set[Coord]) -> int:
    return sum(1 for delta in ACTION_DELTAS.values() if _coord_add(cell, delta) in floors)


def _is_static_corner(cell: Coord, walls: set[Coord]) -> bool:
    up = (cell[0] - 1, cell[1]) in walls
    down = (cell[0] + 1, cell[1]) in walls
    left = (cell[0], cell[1] - 1) in walls
    right = (cell[0], cell[1] + 1) in walls
    return (up or down) and (left or right)


def _goal_candidates(width: int, height: int, walls: tuple[Coord, ...], floors: tuple[Coord, ...]) -> list[Coord]:
    del width, height
    wall_cells = set(walls)
    floor_cells = set(floors)
    candidates = [
        cell
        for cell in floors
        if not _is_static_corner(cell, wall_cells) and _adjacent_floor_degree(cell, floor_cells) >= 2
    ]
    if len(candidates) >= 2:
        return candidates
    return [cell for cell in floors if _adjacent_floor_degree(cell, floor_cells) >= 2]


def _single_box_reachable_cells(width: int, height: int, walls: tuple[Coord, ...], goals: tuple[Coord, ...]) -> set[Coord]:
    floors = _all_floor_cells(width, height, set(walls))
    queue: deque[Coord] = deque(goals)
    reachable = set(goals)
    while queue:
        current = queue.popleft()
        for delta in ACTION_DELTAS.values():
            predecessor = _coord_sub(current, delta)
            support = _coord_sub(predecessor, delta)
            if predecessor in floors and support in floors and predecessor not in reachable:
                reachable.add(predecessor)
                queue.append(predecessor)
    return reachable


def _taboo_cells(width: int, height: int, walls: tuple[Coord, ...], goals: tuple[Coord, ...]) -> tuple[Coord, ...]:
    floor_cells = _all_floor_cells(width, height, set(walls))
    reachable = _single_box_reachable_cells(width, height, walls, goals)
    taboo = {cell for cell in floor_cells if cell not in goals and cell not in reachable}
    return _sorted_coords(taboo)


def _distance_to_goals(width: int, height: int, walls: tuple[Coord, ...], goals: tuple[Coord, ...]) -> dict[Coord, int]:
    wall_cells = set(walls)
    queue: deque[Coord] = deque(goals)
    distances = {goal: 0 for goal in goals}
    while queue:
        current = queue.popleft()
        for delta in ACTION_DELTAS.values():
            previous = _coord_sub(current, delta)
            if previous in wall_cells or previous in distances:
                continue
            if not (0 <= previous[0] < height and 0 <= previous[1] < width):
                continue
            distances[previous] = distances[current] + 1
            queue.append(previous)
    return distances


def _boxes_on_goals(boxes: tuple[Coord, ...], goals: tuple[Coord, ...]) -> int:
    goal_cells = set(goals)
    return sum(1 for box in boxes if box in goal_cells)


def _manhattan_distance(left: Coord, right: Coord) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _box_interaction_pair_count(boxes: tuple[Coord, ...]) -> int:
    count = 0
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if _manhattan_distance(left, right) <= 4:
                count += 1
    return count


def _box_interaction_component_count(boxes: tuple[Coord, ...]) -> int:
    if not boxes:
        return 0
    neighbors = {
        box: {other for other in boxes if other != box and _manhattan_distance(box, other) <= 4}
        for box in boxes
    }
    remaining = set(boxes)
    components = 0
    while remaining:
        components += 1
        start = remaining.pop()
        queue: deque[Coord] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return components


def _freeze_axis_blocked(
    box: Coord,
    axis: str,
    box_cells: set[Coord],
    wall_cells: set[Coord],
    goal_cells: set[Coord],
    cache: dict[tuple[Coord, str], bool],
    visiting: set[tuple[Coord, str]],
) -> bool:
    key = (box, axis)
    if key in cache:
        return cache[key]
    if key in visiting:
        return False
    visiting.add(key)
    if axis == "horizontal":
        neighbors = [(box[0], box[1] - 1), (box[0], box[1] + 1)]
        other_axis = "vertical"
    else:
        neighbors = [(box[0] - 1, box[1]), (box[0] + 1, box[1])]
        other_axis = "horizontal"

    blocked_sides: list[bool] = []
    for neighbor in neighbors:
        if neighbor in wall_cells:
            blocked_sides.append(True)
        elif neighbor in box_cells and neighbor not in goal_cells:
            blocked_sides.append(
                _freeze_axis_blocked(neighbor, other_axis, box_cells, wall_cells, goal_cells, cache, visiting)
            )
        else:
            blocked_sides.append(False)
    visiting.remove(key)
    cache[key] = blocked_sides[0] and blocked_sides[1]
    return cache[key]


def _freeze_deadlock_box(
    boxes: tuple[Coord, ...],
    goals: tuple[Coord, ...],
    walls: tuple[Coord, ...],
) -> Coord | None:
    box_cells = set(boxes)
    goal_cells = set(goals)
    wall_cells = set(walls)
    cache: dict[tuple[Coord, str], bool] = {}
    for box in boxes:
        if box in goal_cells:
            continue
        horizontal_blocked = _freeze_axis_blocked(box, "horizontal", box_cells, wall_cells, goal_cells, cache, set())
        vertical_blocked = _freeze_axis_blocked(box, "vertical", box_cells, wall_cells, goal_cells, cache, set())
        if horizontal_blocked and vertical_blocked:
            return box
    return None


def _detect_deadlock(
    boxes: tuple[Coord, ...],
    goals: tuple[Coord, ...],
    taboo_cells: tuple[Coord, ...],
    walls: tuple[Coord, ...],
) -> str | None:
    taboo = set(taboo_cells)
    goal_cells = set(goals)
    for box in boxes:
        if box not in goal_cells and box in taboo:
            return "static_dead_square"
    frozen_box = _freeze_deadlock_box(boxes, goals, walls)
    if frozen_box is not None:
        return "freeze_deadlock"
    return None


def _build_state(
    player: Coord,
    boxes: tuple[Coord, ...],
    walls: tuple[Coord, ...],
    goals: tuple[Coord, ...],
    taboo_cells: tuple[Coord, ...],
    move_count: int = 0,
    push_count: int = 0,
) -> SokobanState:
    solved = all(box in set(goals) for box in boxes)
    deadlock = False if solved else _detect_deadlock(boxes, goals, taboo_cells, walls) is not None
    return SokobanState(
        player=player,
        boxes=_sorted_coords(list(boxes)),
        solved=solved,
        deadlock=deadlock,
        move_count=move_count,
        push_count=push_count,
    )


def _heuristic(world: SokobanWorld, state: SokobanState) -> int:
    distances = _distance_to_goals(world.width, world.height, world.walls, world.goals)
    fallback = world.width + world.height
    return sum(distances.get(box, fallback) for box in state.boxes)


def _state_key(state: SokobanState) -> tuple[Coord, tuple[Coord, ...]]:
    return (state.player, state.boxes)


def _simulate_action(
    world: SokobanWorld,
    state: SokobanState,
    action_name: str,
    deadlock_terminal: bool,
) -> TransitionResult:
    if action_name not in ACTION_DELTAS:
        return TransitionResult(
            next_state=state,
            invalid_action=True,
            reward_hints={"goal_progress": -0.1},
            info={"invalid_action": True, "reason": "unknown_action"},
        )

    delta = ACTION_DELTAS[action_name]
    wall_cells = _wall_set(world)
    box_cells = _box_set(state)
    target = _coord_add(state.player, delta)

    if target in wall_cells:
        return TransitionResult(
            next_state=state,
            invalid_action=True,
            reward_hints={"goal_progress": -0.1},
            info={"invalid_action": True, "reason": "wall_collision"},
        )

    moved_box = False
    if target in box_cells:
        beyond = _coord_add(target, delta)
        if beyond in wall_cells or beyond in box_cells:
            return TransitionResult(
                next_state=state,
                invalid_action=True,
                reward_hints={"goal_progress": -0.1},
                info={"invalid_action": True, "reason": "blocked_box_push"},
            )
        next_boxes = set(box_cells)
        next_boxes.remove(target)
        next_boxes.add(beyond)
        next_player = target
        moved_box = True
    else:
        next_boxes = set(box_cells)
        next_player = target

    next_state = _build_state(
        player=next_player,
        boxes=_sorted_coords(next_boxes),
        walls=world.walls,
        goals=world.goals,
        taboo_cells=world.taboo_cells,
        move_count=state.move_count + 1,
        push_count=state.push_count + (1 if moved_box else 0),
    )
    previous_boxes_on_goals = _boxes_on_goals(state.boxes, world.goals)
    next_boxes_on_goals = _boxes_on_goals(next_state.boxes, world.goals)
    previous_heuristic = _heuristic(world, state)
    next_heuristic = _heuristic(world, next_state)
    deadlock_reason = _detect_deadlock(next_state.boxes, world.goals, world.taboo_cells, world.walls)
    terminated = next_state.solved or (next_state.deadlock and deadlock_terminal)
    termination_reason = "solved" if next_state.solved else "deadlock" if next_state.deadlock and deadlock_terminal else None

    return TransitionResult(
        next_state=next_state,
        terminated=terminated,
        success=next_state.solved,
        reward_hints={
            "goal_progress": float(next_boxes_on_goals - previous_boxes_on_goals) / max(1, len(world.goals)),
            "heuristic_progress": float(previous_heuristic - next_heuristic) / max(1, previous_heuristic),
            "deadlock_penalty": -1.0 if next_state.deadlock and not next_state.solved else 0.0,
        },
        info={
            "moved_box": moved_box,
            "deadlock_detected": next_state.deadlock,
            "deadlock_reason": deadlock_reason,
            "boxes_on_goals": next_boxes_on_goals,
            "goal_distance": next_heuristic,
            "move_count": next_state.move_count,
            "push_count": next_state.push_count,
            "termination_reason": termination_reason,
        },
    )


def _valid_action_previews(world: SokobanWorld, state: SokobanState, deadlock_terminal: bool) -> list[tuple[str, TransitionResult]]:
    previews: list[tuple[str, TransitionResult]] = []
    for action_name in ACTION_DELTAS:
        transition = _simulate_action(world, state, action_name, deadlock_terminal=deadlock_terminal)
        if not transition.invalid_action:
            previews.append((action_name, transition))
    return previews


def _reverse_action_candidates(world: SokobanWorld, state: SokobanState) -> list[ReverseActionCandidate]:
    wall_cells = _wall_set(world)
    box_cells = _box_set(state)
    floors = _all_floor_cells(world.width, world.height, wall_cells)
    candidates: list[ReverseActionCandidate] = []
    for delta in ACTION_DELTAS.values():
        adjacent = _coord_add(state.player, delta)
        if adjacent in floors and adjacent not in box_cells:
            candidates.append(
                ReverseActionCandidate(
                    action_type="reverse_move",
                    next_player=adjacent,
                    next_boxes=state.boxes,
                    moved_box=False,
                )
            )
        if adjacent in box_cells:
            behind = _coord_sub(state.player, delta)
            if behind in floors and behind not in box_cells:
                next_boxes = set(box_cells)
                next_boxes.remove(adjacent)
                next_boxes.add(state.player)
                candidates.append(
                    ReverseActionCandidate(
                        action_type="reverse_pull",
                        next_player=behind,
                        next_boxes=_sorted_coords(next_boxes),
                        moved_box=True,
                    )
                )
    return candidates


def _scramble_from_solved(
    world: SokobanWorld,
    rng: random.Random,
    steps: int,
    player_start: Coord,
) -> SokobanState:
    state = _build_state(player_start, world.goals, world.walls, world.goals, world.taboo_cells)
    previous_key: tuple[Coord, tuple[Coord, ...]] | None = None
    for _ in range(steps):
        candidates = _reverse_action_candidates(world, state)
        if not candidates:
            break
        non_backtracking = [candidate for candidate in candidates if (candidate.next_player, candidate.next_boxes) != previous_key]
        if non_backtracking:
            candidates = non_backtracking
        pull_candidates = [candidate for candidate in candidates if candidate.moved_box]
        if pull_candidates and rng.random() < 0.75:
            chosen = rng.choice(pull_candidates)
        else:
            chosen = rng.choice(candidates)
        previous_key = _state_key(state)
        state = _build_state(
            player=chosen.next_player,
            boxes=chosen.next_boxes,
            walls=world.walls,
            goals=world.goals,
            taboo_cells=world.taboo_cells,
            move_count=0,
            push_count=0,
        )
    return state


def _solve_sokoban(
    world: SokobanWorld,
    initial_state: SokobanState,
    expansion_limit: int,
) -> SolverResult | None:
    start_key = _state_key(initial_state)
    frontier: list[tuple[int, int, int, SokobanState]] = []
    heapq.heappush(frontier, (_heuristic(world, initial_state), 0, 0, initial_state))
    best_cost = {start_key: 0}
    parent: dict[tuple[Coord, tuple[Coord, ...]], tuple[Coord, tuple[Coord, ...]] | None] = {start_key: None}
    parent_action: dict[tuple[Coord, tuple[Coord, ...]], str] = {}
    parent_push: dict[tuple[Coord, tuple[Coord, ...]], int] = {start_key: 0}
    counter = 1
    expanded = 0

    while frontier:
        _, cost, _, state = heapq.heappop(frontier)
        state_key = _state_key(state)
        if cost != best_cost.get(state_key):
            continue
        expanded += 1
        if expanded > expansion_limit:
            return None
        if state.solved:
            actions: list[str] = []
            push_count = 0
            cursor = state_key
            while parent[cursor] is not None:
                actions.append(parent_action[cursor])
                push_count += parent_push[cursor]
                cursor = parent[cursor]
            actions.reverse()
            return SolverResult(
                actions=tuple(actions),
                move_count=len(actions),
                push_count=push_count,
                expanded_nodes=expanded,
            )

        for action_name, transition in _valid_action_previews(world, state, deadlock_terminal=False):
            next_state = transition.next_state
            if next_state.deadlock and not next_state.solved:
                continue
            next_key = _state_key(next_state)
            next_cost = cost + 1
            if next_cost >= best_cost.get(next_key, 10**9):
                continue
            best_cost[next_key] = next_cost
            parent[next_key] = state_key
            parent_action[next_key] = action_name
            parent_push[next_key] = 1 if transition.info.get("moved_box") else 0
            priority = next_cost + _heuristic(world, next_state)
            heapq.heappush(frontier, (priority, next_cost, counter, next_state))
            counter += 1
    return None


class SokobanActionVerifier(BaseVerifier):
    name = "sokoban_action_legality"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        passed = not context.transition.invalid_action
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.ACTION,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="Primitive Sokoban actions must be legal under exact box-pushing dynamics.",
            ),
        )


class SokobanStateVerifier(BaseVerifier):
    name = "sokoban_state_validity"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        if context.transition.invalid_action:
            return (
                VerificationResult(
                    name=self.name,
                    scope=VerificationScope.STATE,
                    passed=False,
                    score=0.0,
                    kind=VerificationKind.FEASIBILITY,
                    weight=2.0,
                    hard=True,
                    message="Invalid actions do not produce a valid Sokoban successor state.",
                ),
            )

        expected = _simulate_action(
            context.world,
            context.previous_state,
            context.action.name,
            bool(context.generation_params.get("deadlock_ends_episode", True)),
        )
        wall_cells = _wall_set(context.world)
        passed = (
            expected.next_state == context.next_state
            and not expected.invalid_action
            and context.next_state.player not in wall_cells
            and len(context.next_state.boxes) == len(set(context.next_state.boxes))
            and not any(box in wall_cells for box in context.next_state.boxes)
        )
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.STATE,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="State updates must exactly match the deterministic Sokoban transition function.",
            ),
        )


class SokobanDeadlockVerifier(BaseVerifier):
    name = "sokoban_deadlock_detection"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        if context.transition.invalid_action:
            return ()
        passed = not (context.next_state.deadlock and not context.next_state.solved)
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.STATE,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=1.5,
                hard=True,
                message="States flagged as static deadlocks should be treated as failed planning states.",
                metadata={"deadlock_reason": context.transition.info.get("deadlock_reason")},
            ),
        )


class SokobanGoalVerifier(BaseVerifier):
    name = "sokoban_goal_reached"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        if not context.transition.terminated:
            return ()
        if context.transition.success:
            passed = context.next_state.solved and all(box in _goal_set(context.world) for box in context.next_state.boxes)
        else:
            passed = not context.next_state.solved
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.GOAL,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="Success should occur exactly when every box occupies a goal cell.",
            ),
        )


class SokobanTrajectoryVerifier(BaseVerifier):
    def evaluate_trajectory(self, context: TrajectoryContext) -> tuple[VerificationResult, ...]:
        steps = context.trace.steps
        deadlock_free = all(not step.info.get("deadlock_detected") for step in steps)
        observed_moves = len(steps)
        observed_pushes = context.final_state.push_count
        oracle_moves = context.world.oracle_move_count
        oracle_pushes = context.world.oracle_push_count
        move_gap = abs(observed_moves - oracle_moves)
        push_gap = abs(observed_pushes - oracle_pushes)
        move_efficiency_score = 0.0
        push_efficiency_score = 0.0
        if context.success:
            move_efficiency_score = max(0.0, 1.0 - move_gap / max(1, oracle_moves))
            push_efficiency_score = max(0.0, 1.0 - push_gap / max(1, oracle_pushes))
        return (
            VerificationResult(
                name="sokoban_completion",
                scope=VerificationScope.TRAJECTORY,
                passed=context.success and context.final_state.solved,
                score=1.0 if context.success and context.final_state.solved else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="A successful trajectory must end with all boxes on goals.",
            ),
            VerificationResult(
                name="sokoban_deadlock_free_trajectory",
                scope=VerificationScope.TRAJECTORY,
                passed=deadlock_free,
                score=1.0 if deadlock_free else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=1.0,
                hard=False,
                message="Trajectory diagnostics record whether the agent ever entered a known deadlock state.",
            ),
            VerificationResult(
                name="sokoban_move_efficiency",
                scope=VerificationScope.TRAJECTORY,
                passed=context.success and observed_moves == oracle_moves,
                score=move_efficiency_score,
                kind=VerificationKind.QUALITY,
                weight=1.5,
                hard=False,
                message="Primary Sokoban quality tracks how closely primitive move count matches the solver-backed oracle.",
                metadata={
                    "primary_metric": True,
                    "observed_moves": observed_moves,
                    "oracle_moves": oracle_moves,
                    "move_gap": move_gap,
                },
            ),
            VerificationResult(
                name="sokoban_push_efficiency",
                scope=VerificationScope.TRAJECTORY,
                passed=context.success and observed_pushes == oracle_pushes,
                score=push_efficiency_score,
                kind=VerificationKind.QUALITY,
                weight=1.0,
                hard=False,
                message="Secondary Sokoban quality tracks how closely push count matches the oracle plan.",
                metadata={
                    "primary_metric": False,
                    "observed_pushes": observed_pushes,
                    "oracle_pushes": oracle_pushes,
                    "push_gap": push_gap,
                },
            ),
        )


class SokobanOracle(Oracle):
    def __init__(self, world: SokobanWorld) -> None:
        self.world = world

    def is_feasible(self) -> bool | None:
        return True

    def estimate_difficulty(self) -> float | None:
        return float(self.world.oracle_move_count + 2 * self.world.oracle_push_count + len(self.world.taboo_cells) / 8)

    def solve(self) -> OracleSolution:
        actions = tuple({"name": action_name, "arguments": {}} for action_name in self.world.oracle_plan)
        return OracleSolution(
            actions=actions,
            metadata={
                "template_name": self.world.template_name,
                "push_count": self.world.oracle_push_count,
                "solver_expansions": self.world.solver_expansions,
                "deadlock_detector": "static_dead_square+freeze_deadlock",
            },
            objective_value=self.world.oracle_move_count,
            feasible=True,
            optimal=True,
            difficulty_estimate=self.estimate_difficulty(),
            certificate=ProofCertificate(
                feasible=True,
                optimal=True,
                summary="Optimal primitive-move plan found with A* over exact Sokoban states using static and conservative freeze deadlock pruning.",
                witness={
                    "template_name": self.world.template_name,
                    "move_count": self.world.oracle_move_count,
                    "push_count": self.world.oracle_push_count,
                    "solver_expansions": self.world.solver_expansions,
                    "reverse_scramble_steps": self.world.reverse_scramble_steps,
                },
            ),
        )


class SokobanFamily(EnvironmentFamily):
    name = "sokoban"
    description = "Procedural Sokoban-style box-pushing planning with exact transitions, deadlock checks, and solver-backed oracles."

    def sample_generation_params(self, config: FamilyConfig, rng: random.Random) -> dict[str, Any]:
        difficulty_settings = {
            "easy": {
                "num_boxes": 1,
                "reverse_scramble_steps": (6, 10),
                "min_solution_length": 4,
                "max_solution_length": 18,
                "max_boxes_on_goals_at_start": 0,
                "min_interaction_pairs": 0,
                "template_pool": ["compact_cross", "side_rooms"],
                "solver_expansion_limit": 12000,
            },
            "medium": {
                "num_boxes": 2,
                "reverse_scramble_steps": (10, 16),
                "min_solution_length": 8,
                "max_solution_length": 36,
                "max_boxes_on_goals_at_start": 0,
                "min_interaction_pairs": 1,
                "template_pool": ["compact_cross", "side_rooms", "warehouse_large"],
                "solver_expansion_limit": 30000,
            },
            "hard": {
                "num_boxes": 3,
                "reverse_scramble_steps": (14, 22),
                "min_solution_length": 12,
                "max_solution_length": 60,
                "max_boxes_on_goals_at_start": 0,
                "min_interaction_pairs": 2,
                "template_pool": ["warehouse_large", "rooms_and_halls"],
                "solver_expansion_limit": 80000,
            },
        }
        settings = difficulty_settings.get(config.difficulty, difficulty_settings["medium"])
        return {
            "num_boxes": settings["num_boxes"],
            "reverse_scramble_steps": rng.randint(*settings["reverse_scramble_steps"]),
            "min_solution_length": settings["min_solution_length"],
            "max_solution_length": settings["max_solution_length"],
            "max_boxes_on_goals_at_start": settings["max_boxes_on_goals_at_start"],
            "min_interaction_pairs": settings["min_interaction_pairs"],
            "template_pool": list(settings["template_pool"]),
            "solver_expansion_limit": settings["solver_expansion_limit"],
            "deadlock_ends_episode": True,
            "observability": config.observability,
        }

    def sample_world(self, generation_params: dict[str, Any], rng: random.Random) -> SokobanWorld:
        template_names = set(generation_params.get("template_pool", []))
        candidate_templates = [template for template in TEMPLATES if template.name in template_names] or list(TEMPLATES)
        num_boxes = int(generation_params["num_boxes"])
        reverse_steps = int(generation_params["reverse_scramble_steps"])
        min_solution_length = int(generation_params["min_solution_length"])
        max_solution_length = int(generation_params["max_solution_length"])
        max_boxes_on_goals_at_start = int(generation_params.get("max_boxes_on_goals_at_start", 0))
        min_interaction_pairs = int(generation_params.get("min_interaction_pairs", 0))
        solver_expansion_limit = int(generation_params["solver_expansion_limit"])

        for _ in range(96):
            template = rng.choice(candidate_templates)
            rows = _transform_rows(template.rows, rng.choice(["identity", "flip_horizontal", "flip_vertical", "flip_both"]))
            width, height, walls, floors = _parse_template(rows)
            candidates = _goal_candidates(width, height, walls, floors)
            if len(candidates) < num_boxes + 1:
                continue
            goals = _sorted_coords(rng.sample(candidates, num_boxes))
            taboo = _taboo_cells(width, height, walls, goals)
            player_candidates = [cell for cell in floors if cell not in goals]
            if not player_candidates:
                continue
            player_start = rng.choice(player_candidates)
            candidate_world = SokobanWorld(
                template_name=template.name,
                width=width,
                height=height,
                walls=walls,
                goals=goals,
                initial_boxes=goals,
                initial_player=player_start,
                taboo_cells=taboo,
                reverse_scramble_steps=reverse_steps,
            )
            scrambled_state = _scramble_from_solved(candidate_world, rng, reverse_steps, player_start)
            if scrambled_state.boxes == candidate_world.goals:
                continue
            candidate_world = replace(
                candidate_world,
                initial_boxes=scrambled_state.boxes,
                initial_player=scrambled_state.player,
            )
            initial_state = _build_state(
                player=candidate_world.initial_player,
                boxes=candidate_world.initial_boxes,
                walls=candidate_world.walls,
                goals=candidate_world.goals,
                taboo_cells=candidate_world.taboo_cells,
            )
            if initial_state.deadlock or initial_state.solved:
                continue
            boxes_on_goals_at_start = _boxes_on_goals(initial_state.boxes, candidate_world.goals)
            interaction_pairs = _box_interaction_pair_count(initial_state.boxes)
            effective_min_interaction_pairs = min_interaction_pairs if num_boxes <= 1 else max(
                0,
                min_interaction_pairs - 1 if _ >= 48 else min_interaction_pairs,
            )
            if boxes_on_goals_at_start > max_boxes_on_goals_at_start:
                continue
            if interaction_pairs < effective_min_interaction_pairs:
                continue
            solver_result = _solve_sokoban(candidate_world, initial_state, expansion_limit=solver_expansion_limit)
            if solver_result is None:
                continue
            if not (min_solution_length <= solver_result.move_count <= max_solution_length):
                continue
            return replace(
                candidate_world,
                oracle_plan=solver_result.actions,
                oracle_move_count=solver_result.move_count,
                oracle_push_count=solver_result.push_count,
                solver_expansions=solver_result.expanded_nodes,
            )
        raise ValueError("Unable to generate a solvable Sokoban instance matching the requested difficulty.")

    def derive_objective(
        self,
        world: SokobanWorld,
        generation_params: dict[str, Any],
        rng: random.Random,
    ) -> TaskObjective:
        return TaskObjective(
            name="solve_sokoban",
            description="Push every box onto a goal cell using legal primitive Sokoban moves.",
            success_criteria={"all_boxes_on_goals": True},
            optimization_target="minimize_move_count",
            constraint_spec={
                "deterministic_push_dynamics": True,
                "primitive_actions_only": True,
                "deadlock_detection": ["static_dead_square", "freeze_deadlock"],
                "reported_quality_metrics": ["move_count_gap", "push_count_gap"],
            },
            quality_metric="move_count_gap",
        )

    def initial_state(
        self,
        world: SokobanWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> SokobanState:
        return _build_state(world.initial_player, world.initial_boxes, world.walls, world.goals, world.taboo_cells)

    def observe(
        self,
        world: SokobanWorld,
        state: SokobanState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "mode": generation_params["observability"],
            "board": {
                "height": world.height,
                "width": world.width,
                "walls": [list(cell) for cell in world.walls],
                "goals": [list(cell) for cell in world.goals],
            },
            "player_position": list(state.player),
            "box_positions": [list(cell) for cell in state.boxes],
            "num_boxes": len(state.boxes),
            "boxes_on_goals": _boxes_on_goals(state.boxes, world.goals),
            "solved": state.solved,
            "deadlock": state.deadlock,
            "move_count": state.move_count,
            "push_count": state.push_count,
            "deadlock_annotations": {"static_dead_squares": [list(cell) for cell in world.taboo_cells]},
            "ascii_board": _render_ascii(
                world.width,
                world.height,
                world.walls,
                world.goals,
                player=state.player,
                boxes=state.boxes,
            ),
        }

    def build_task_space(
        self,
        world: SokobanWorld,
        objective: TaskObjective,
        initial_state: SokobanState,
        generation_params: dict[str, Any],
    ) -> TaskSpace:
        return TaskSpace(
            observation_schema={
                "type": "object",
                "fields": {
                    "board": "grid_static_layout",
                    "player_position": "coord",
                    "box_positions": "list[coord]",
                    "ascii_board": "string",
                    "boxes_on_goals": "int",
                    "deadlock": "bool",
                },
                "observability": generation_params["observability"],
            },
            action_schema={
                "type": "canonical_object",
                "actions": [{"name": action_name, "arguments": {}} for action_name in ACTION_DELTAS],
            },
            runtime_api="gym_like",
            notes="Gym-like reset()/step() tuples over exact Sokoban primitive actions.",
        )

    def valid_actions(
        self,
        world: SokobanWorld,
        state: SokobanState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> list[CanonicalAction]:
        previews = _valid_action_previews(world, state, bool(generation_params.get("deadlock_ends_episode", True)))
        return [
            CanonicalAction(
                name=action_name,
                arguments={
                    "causes_push": bool(transition.info.get("moved_box")),
                    "boxes_on_goals_after": int(transition.info.get("boxes_on_goals", 0)),
                    "goal_distance_after": int(transition.info.get("goal_distance", 0)),
                    "introduces_deadlock": bool(transition.info.get("deadlock_detected")),
                },
            )
            for action_name, transition in previews
        ]

    def transition(
        self,
        world: SokobanWorld,
        state: SokobanState,
        action: CanonicalAction,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> TransitionResult:
        return _simulate_action(
            world,
            state,
            action.name,
            deadlock_terminal=bool(generation_params.get("deadlock_ends_episode", True)),
        )

    def build_verifier_suite(
        self,
        world: SokobanWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> VerifierSuite:
        return VerifierSuite(
            [
                SokobanActionVerifier(),
                SokobanStateVerifier(),
                SokobanDeadlockVerifier(),
                SokobanGoalVerifier(),
                SokobanTrajectoryVerifier(),
            ]
        )

    def build_oracle(
        self,
        world: SokobanWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> SokobanOracle:
        return SokobanOracle(world)

    def recommended_max_steps(self, generation_params: dict[str, Any]) -> int:
        return int(generation_params["max_solution_length"]) * 3 + 12

    def export_world(self, world: SokobanWorld) -> dict[str, Any]:
        boxes_on_goals_at_start = _boxes_on_goals(world.initial_boxes, world.goals)
        return {
            "template_name": world.template_name,
            "width": world.width,
            "height": world.height,
            "walls": [list(cell) for cell in world.walls],
            "goals": [list(cell) for cell in world.goals],
            "initial_boxes": [list(cell) for cell in world.initial_boxes],
            "initial_player": list(world.initial_player),
            "taboo_cells": [list(cell) for cell in world.taboo_cells],
            "reverse_scramble_steps": world.reverse_scramble_steps,
            "oracle_move_count": world.oracle_move_count,
            "oracle_push_count": world.oracle_push_count,
            "solver_expansions": world.solver_expansions,
            "boxes_on_goals_at_start": boxes_on_goals_at_start,
            "unsolved_boxes_at_start": len(world.initial_boxes) - boxes_on_goals_at_start,
            "box_interaction_pair_count": _box_interaction_pair_count(world.initial_boxes),
            "box_interaction_component_count": _box_interaction_component_count(world.initial_boxes),
            "static_board_ascii": _render_ascii(world.width, world.height, world.walls, world.goals),
        }

    def export_state(self, state: SokobanState) -> dict[str, Any]:
        return {
            "player": list(state.player),
            "boxes": [list(cell) for cell in state.boxes],
            "solved": state.solved,
            "deadlock": state.deadlock,
            "move_count": state.move_count,
            "push_count": state.push_count,
        }

    def task_metadata(
        self,
        world: SokobanWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        total_cells = world.width * world.height
        boxes_on_goals_at_start = _boxes_on_goals(world.initial_boxes, world.goals)
        return {
            "template_name": world.template_name,
            "board_height": world.height,
            "board_width": world.width,
            "num_boxes": len(world.initial_boxes),
            "num_goals": len(world.goals),
            "num_walls": len(world.walls),
            "wall_density": len(world.walls) / max(1, total_cells),
            "num_taboo_cells": len(world.taboo_cells),
            "oracle_steps": world.oracle_move_count,
            "oracle_push_count": world.oracle_push_count,
            "reverse_scramble_steps": world.reverse_scramble_steps,
            "boxes_on_goals_at_start": boxes_on_goals_at_start,
            "unsolved_boxes_at_start": len(world.initial_boxes) - boxes_on_goals_at_start,
            "box_interaction_pair_count": _box_interaction_pair_count(world.initial_boxes),
            "box_interaction_component_count": _box_interaction_component_count(world.initial_boxes),
            "primary_quality_metric": "move_count_gap",
            "secondary_quality_metric": "push_count_gap",
        }
