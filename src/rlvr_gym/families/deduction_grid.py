from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DeductionCategory:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class DeductionFact:
    category: str
    entity: str
    value: str


@dataclass(frozen=True)
class DeductionClue:
    clue_id: str
    clue_type: str
    positive: bool
    text: str
    entity: str | None = None
    category: str | None = None
    value: str | None = None
    left_category: str | None = None
    left_value: str | None = None
    right_category: str | None = None
    right_value: str | None = None


@dataclass(frozen=True)
class DeductionGridWorld:
    base_category: DeductionCategory
    relation_categories: tuple[DeductionCategory, ...]
    assignment: dict[str, dict[str, str]]
    clues: tuple[DeductionClue, ...]
    oracle_plan: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class DeductionGridState:
    known_true: tuple[DeductionFact, ...] = ()
    known_false: tuple[DeductionFact, ...] = ()
    committed_assignment: dict[str, dict[str, str]] | None = None
    solved: bool = False


@dataclass(frozen=True)
class DeductionAnalysis:
    contradiction: bool
    contradiction_reason: str
    possible: dict[str, dict[str, tuple[str, ...]]]
    entailed_true: tuple[DeductionFact, ...]
    entailed_false: tuple[DeductionFact, ...]
    pending_true: tuple[DeductionFact, ...]
    pending_false: tuple[DeductionFact, ...]
    resolved_assignment: dict[str, dict[str, str]]
    solved: bool
    num_resolved_pairs: int
    num_eliminated_pairs: int


CATEGORY_LIBRARY: tuple[DeductionCategory, ...] = (
    DeductionCategory("Person", ("Alice", "Bob", "Carol", "Dana", "Eli")),
    DeductionCategory("House", ("Red", "Blue", "Green", "Yellow", "White")),
    DeductionCategory("Pet", ("Cat", "Dog", "Fish", "Bird", "Horse")),
    DeductionCategory("Drink", ("Tea", "Coffee", "Milk", "Juice", "Water")),
    DeductionCategory("Meal", ("Pasta", "Soup", "Salad", "Rice", "Tacos")),
    DeductionCategory("Vehicle", ("Bike", "Car", "Train", "Boat", "Scooter")),
    DeductionCategory("Hobby", ("Chess", "Music", "Reading", "Painting", "Running")),
    DeductionCategory("Time", ("Morning", "Noon", "Evening", "Night", "Dawn")),
)


def _relation_lookup(world: DeductionGridWorld) -> dict[str, DeductionCategory]:
    return {category.name: category for category in world.relation_categories}


def _entities(world: DeductionGridWorld) -> tuple[str, ...]:
    return world.base_category.values


def _fact_key(fact: DeductionFact) -> tuple[str, str, str]:
    return (fact.category, fact.entity, fact.value)


def _sorted_facts(facts: set[DeductionFact] | tuple[DeductionFact, ...] | list[DeductionFact]) -> tuple[DeductionFact, ...]:
    return tuple(sorted(facts, key=_fact_key))


def _fact_to_dict(fact: DeductionFact) -> dict[str, str]:
    return {"category": fact.category, "entity": fact.entity, "value": fact.value}


def _possible_snapshot(possible: dict[str, dict[str, set[str]]], relation_categories: tuple[DeductionCategory, ...]) -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        category.name: {
            entity: tuple(sorted(possible[category.name][entity]))
            for entity in possible[category.name]
        }
        for category in relation_categories
    }


def _render_clue_text(clue: DeductionClue) -> str:
    return clue.text


def _make_direct_clue(index: int, entity: str, category: str, value: str, positive: bool) -> DeductionClue:
    text = (
        f"{entity} is matched with {value} in {category}."
        if positive
        else f"{entity} is not matched with {value} in {category}."
    )
    return DeductionClue(
        clue_id=f"clue_{index}",
        clue_type="entity_value",
        positive=positive,
        text=text,
        entity=entity,
        category=category,
        value=value,
    )


def _make_link_clue(
    index: int,
    left_category: str,
    left_value: str,
    right_category: str,
    right_value: str,
    positive: bool,
) -> DeductionClue:
    text = (
        f"The {left_value} in {left_category} is matched with the {right_value} in {right_category}."
        if positive
        else f"The {left_value} in {left_category} is not matched with the {right_value} in {right_category}."
    )
    return DeductionClue(
        clue_id=f"clue_{index}",
        clue_type="linked_values",
        positive=positive,
        text=text,
        left_category=left_category,
        left_value=left_value,
        right_category=right_category,
        right_value=right_value,
    )


def _init_possible(base_category: DeductionCategory, relation_categories: tuple[DeductionCategory, ...]) -> dict[str, dict[str, set[str]]]:
    return {
        category.name: {entity: set(category.values) for entity in base_category.values}
        for category in relation_categories
    }


def _force_true(
    possible: dict[str, dict[str, set[str]]],
    relation_lookup: dict[str, DeductionCategory],
    entities: tuple[str, ...],
    fact: DeductionFact,
) -> tuple[bool, str | None]:
    options = possible[fact.category][fact.entity]
    if fact.value not in options:
        return False, f"{fact.entity} cannot be matched with {fact.value} in {fact.category}."
    changed = False
    if options != {fact.value}:
        possible[fact.category][fact.entity] = {fact.value}
        options = possible[fact.category][fact.entity]
        changed = True
    for other_entity in entities:
        if other_entity == fact.entity:
            continue
        other_options = possible[fact.category][other_entity]
        if fact.value in other_options:
            other_options.remove(fact.value)
            changed = True
            if not other_options:
                return False, f"{other_entity} has no remaining values in {fact.category}."
    for candidate in relation_lookup[fact.category].values:
        if candidate != fact.value and candidate in options:
            options.remove(candidate)
            changed = True
    if not options:
        return False, f"{fact.entity} has no remaining values in {fact.category}."
    return changed, None


def _force_false(
    possible: dict[str, dict[str, set[str]]],
    fact: DeductionFact,
) -> tuple[bool, str | None]:
    options = possible[fact.category][fact.entity]
    if fact.value not in options:
        return False, None
    options.remove(fact.value)
    if not options:
        return False, f"{fact.entity} has no remaining values in {fact.category}."
    return True, None


def _analyze_state(
    base_category: DeductionCategory,
    relation_categories: tuple[DeductionCategory, ...],
    clues: tuple[DeductionClue, ...],
    state: DeductionGridState,
) -> DeductionAnalysis:
    relation_lookup = {category.name: category for category in relation_categories}
    entities = base_category.values
    possible = _init_possible(base_category, relation_categories)
    contradiction = False
    contradiction_reason = ""

    def apply_true(fact: DeductionFact) -> bool:
        nonlocal contradiction, contradiction_reason
        changed, reason = _force_true(possible, relation_lookup, entities, fact)
        if reason is not None:
            contradiction = True
            contradiction_reason = reason
            return False
        return changed

    def apply_false(fact: DeductionFact) -> bool:
        nonlocal contradiction, contradiction_reason
        changed, reason = _force_false(possible, fact)
        if reason is not None:
            contradiction = True
            contradiction_reason = reason
            return False
        return changed

    changed = True
    while changed and not contradiction:
        changed = False
        for fact in state.known_true:
            changed = apply_true(fact) or changed
        for fact in state.known_false:
            changed = apply_false(fact) or changed
        for clue in clues:
            if clue.clue_type == "entity_value":
                fact = DeductionFact(category=str(clue.category), entity=str(clue.entity), value=str(clue.value))
                changed = (apply_true(fact) if clue.positive else apply_false(fact)) or changed
                continue
            left_category = str(clue.left_category)
            left_value = str(clue.left_value)
            right_category = str(clue.right_category)
            right_value = str(clue.right_value)
            for entity in entities:
                left_fact = DeductionFact(category=left_category, entity=entity, value=left_value)
                right_fact = DeductionFact(category=right_category, entity=entity, value=right_value)
                left_possible = left_value in possible[left_category][entity]
                right_possible = right_value in possible[right_category][entity]
                if clue.positive:
                    if left_possible and not right_possible:
                        changed = apply_false(left_fact) or changed
                    if right_possible and not left_possible:
                        changed = apply_false(right_fact) or changed
                    if possible[left_category][entity] == {left_value}:
                        changed = apply_true(right_fact) or changed
                    if possible[right_category][entity] == {right_value}:
                        changed = apply_true(left_fact) or changed
                else:
                    if possible[left_category][entity] == {left_value}:
                        changed = apply_false(right_fact) or changed
                    if possible[right_category][entity] == {right_value}:
                        changed = apply_false(left_fact) or changed
        for category in relation_categories:
            for entity in entities:
                options = possible[category.name][entity]
                if not options:
                    contradiction = True
                    contradiction_reason = f"{entity} has no remaining values in {category.name}."
                    break
                if len(options) == 1:
                    changed = apply_true(
                        DeductionFact(category=category.name, entity=entity, value=next(iter(options)))
                    ) or changed
            if contradiction:
                break
            for value in category.values:
                candidate_entities = [entity for entity in entities if value in possible[category.name][entity]]
                if not candidate_entities:
                    contradiction = True
                    contradiction_reason = f"{value} has no supporting entity in {category.name}."
                    break
                if len(candidate_entities) == 1:
                    changed = apply_true(
                        DeductionFact(category=category.name, entity=candidate_entities[0], value=value)
                    ) or changed
            if contradiction:
                break

    entailed_true: set[DeductionFact] = set()
    entailed_false: set[DeductionFact] = set()
    resolved_assignment: dict[str, dict[str, str]] = {}
    num_resolved_pairs = 0
    num_eliminated_pairs = 0
    for category in relation_categories:
        category_assignment: dict[str, str] = {}
        for entity in entities:
            options = possible[category.name][entity]
            num_eliminated_pairs += len(category.values) - len(options)
            if len(options) == 1:
                resolved_value = next(iter(options))
                entailed_true.add(DeductionFact(category=category.name, entity=entity, value=resolved_value))
                category_assignment[entity] = resolved_value
                num_resolved_pairs += 1
            for value in category.values:
                if value not in options:
                    entailed_false.add(DeductionFact(category=category.name, entity=entity, value=value))
        if category_assignment:
            resolved_assignment[category.name] = category_assignment

    known_true_set = set(state.known_true)
    known_false_set = set(state.known_false)
    pending_true = _sorted_facts(set(entailed_true) - known_true_set)
    pending_false = _sorted_facts(set(entailed_false) - known_false_set)
    solved = (
        not contradiction
        and num_resolved_pairs == len(entities) * len(relation_categories)
        and len(resolved_assignment) == len(relation_categories)
    )
    return DeductionAnalysis(
        contradiction=contradiction,
        contradiction_reason=contradiction_reason,
        possible=_possible_snapshot(possible, relation_categories),
        entailed_true=_sorted_facts(entailed_true),
        entailed_false=_sorted_facts(entailed_false),
        pending_true=pending_true,
        pending_false=pending_false,
        resolved_assignment=resolved_assignment,
        solved=solved,
        num_resolved_pairs=num_resolved_pairs,
        num_eliminated_pairs=num_eliminated_pairs,
    )


def _fact_from_action(action: CanonicalAction) -> DeductionFact | None:
    category = action.arguments.get("category")
    entity = action.arguments.get("entity")
    value = action.arguments.get("value")
    if not isinstance(category, str) or not isinstance(entity, str) or not isinstance(value, str):
        return None
    return DeductionFact(category=category, entity=entity, value=value)


def _normalize_assignment(raw_assignment: Any, world: DeductionGridWorld) -> dict[str, dict[str, str]] | None:
    if not isinstance(raw_assignment, dict):
        return None
    normalized: dict[str, dict[str, str]] = {}
    for category in world.relation_categories:
        category_payload = raw_assignment.get(category.name)
        if not isinstance(category_payload, dict):
            return None
        normalized[category.name] = {}
        for entity in _entities(world):
            value = category_payload.get(entity)
            if not isinstance(value, str):
                return None
            normalized[category.name][entity] = value
    return normalized


def _assignment_satisfies_world(world: DeductionGridWorld, assignment: dict[str, dict[str, str]]) -> bool:
    entities = _entities(world)
    for category in world.relation_categories:
        if category.name not in assignment:
            return False
        values = [assignment[category.name].get(entity) for entity in entities]
        if any(not isinstance(value, str) for value in values):
            return False
        if sorted(values) != sorted(category.values):
            return False
    for clue in world.clues:
        if clue.clue_type == "entity_value":
            actual = assignment[str(clue.category)][str(clue.entity)]
            if clue.positive and actual != clue.value:
                return False
            if not clue.positive and actual == clue.value:
                return False
            continue
        left_entities = [entity for entity in entities if assignment[str(clue.left_category)][entity] == clue.left_value]
        right_entities = [entity for entity in entities if assignment[str(clue.right_category)][entity] == clue.right_value]
        if len(left_entities) != 1 or len(right_entities) != 1:
            return False
        same_entity = left_entities[0] == right_entities[0]
        if clue.positive and not same_entity:
            return False
        if not clue.positive and same_entity:
            return False
    return True


def _build_oracle_plan(
    base_category: DeductionCategory,
    relation_categories: tuple[DeductionCategory, ...],
    clues: tuple[DeductionClue, ...],
) -> tuple[dict[str, Any], ...]:
    state = DeductionGridState()
    actions: list[dict[str, Any]] = []
    budget = len(base_category.values) * len(relation_categories) * 3 + 8
    for _ in range(budget):
        analysis = _analyze_state(base_category, relation_categories, clues, state)
        if analysis.contradiction:
            raise ValueError(f"Contradictory puzzle during oracle planning: {analysis.contradiction_reason}")
        if _commit_ready(analysis):
            actions.append({"name": "commit_solution", "arguments": {"assignment": analysis.resolved_assignment}})
            return tuple(actions)
        if analysis.pending_true:
            fact = analysis.pending_true[0]
            actions.append({"name": "assert_pair", "arguments": _fact_to_dict(fact)})
            state = DeductionGridState(
                known_true=_sorted_facts(set(state.known_true) | {fact}),
                known_false=state.known_false,
            )
            continue
        if analysis.pending_false:
            actions.append(
                {
                    "name": "propagate",
                    "arguments": {
                        "num_true_updates": len(analysis.pending_true),
                        "num_false_updates": len(analysis.pending_false),
                    },
                }
            )
            state = DeductionGridState(
                known_true=_sorted_facts(set(state.known_true) | set(analysis.pending_true)),
                known_false=_sorted_facts(set(state.known_false) | set(analysis.pending_false)),
            )
            continue
        raise ValueError("Unable to derive an oracle plan for the deduction grid puzzle.")
    raise ValueError("Oracle planning exceeded the expected step budget.")


def _build_candidate_clues(
    relation_categories: tuple[DeductionCategory, ...],
    assignment: dict[str, dict[str, str]],
    entities: tuple[str, ...],
    rng: random.Random,
    prefer_relational: bool,
) -> list[DeductionClue]:
    positive_links: list[DeductionClue] = []
    negative_links: list[DeductionClue] = []
    positive_direct: list[DeductionClue] = []
    negative_direct: list[DeductionClue] = []
    clue_index = 0

    category_pairs = list(itertools.combinations(relation_categories, 2))
    rng.shuffle(category_pairs)
    for left_category, right_category in category_pairs:
        shuffled_entities = list(entities)
        rng.shuffle(shuffled_entities)
        for entity in shuffled_entities:
            left_value = assignment[left_category.name][entity]
            right_value = assignment[right_category.name][entity]
            positive_links.append(
                _make_link_clue(clue_index, left_category.name, left_value, right_category.name, right_value, True)
            )
            clue_index += 1
            wrong_values = [value for value in right_category.values if value != right_value]
            rng.shuffle(wrong_values)
            for wrong_value in wrong_values[: max(1, len(wrong_values) // 2)]:
                negative_links.append(
                    _make_link_clue(clue_index, left_category.name, left_value, right_category.name, wrong_value, False)
                )
                clue_index += 1

    shuffled_categories = list(relation_categories)
    rng.shuffle(shuffled_categories)
    for category in shuffled_categories:
        shuffled_entities = list(entities)
        rng.shuffle(shuffled_entities)
        for entity in shuffled_entities:
            actual_value = assignment[category.name][entity]
            positive_direct.append(_make_direct_clue(clue_index, entity, category.name, actual_value, True))
            clue_index += 1
            wrong_values = [value for value in category.values if value != actual_value]
            rng.shuffle(wrong_values)
            for wrong_value in wrong_values[: max(1, len(wrong_values) // 2)]:
                negative_direct.append(_make_direct_clue(clue_index, entity, category.name, wrong_value, False))
                clue_index += 1

    if prefer_relational:
        return positive_links + negative_links + positive_direct + negative_direct
    return positive_direct + negative_direct + positive_links + negative_links


def _select_clues(
    base_category: DeductionCategory,
    relation_categories: tuple[DeductionCategory, ...],
    candidate_clues: list[DeductionClue],
    rng: random.Random,
    num_distractor_clues: int,
) -> tuple[DeductionClue, ...]:
    selected: list[DeductionClue] = []
    remaining = list(candidate_clues)
    while remaining:
        baseline = _analyze_state(base_category, relation_categories, tuple(selected), DeductionGridState())
        if baseline.solved:
            break
        best_index = None
        best_score = -1
        for index, clue in enumerate(remaining):
            analysis = _analyze_state(base_category, relation_categories, tuple(selected + [clue]), DeductionGridState())
            if analysis.contradiction:
                continue
            score = analysis.num_resolved_pairs * 1000 + analysis.num_eliminated_pairs
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selected.append(remaining.pop(best_index))

    solved_analysis = _analyze_state(base_category, relation_categories, tuple(selected), DeductionGridState())
    if not solved_analysis.solved:
        for clue in remaining:
            selected.append(clue)
            solved_analysis = _analyze_state(base_category, relation_categories, tuple(selected), DeductionGridState())
            if solved_analysis.solved:
                break
    if not solved_analysis.solved:
        raise ValueError("Unable to generate a uniquely solvable deduction-grid puzzle.")

    rng.shuffle(remaining)
    added = 0
    for clue in remaining:
        if added >= num_distractor_clues:
            break
        analysis = _analyze_state(base_category, relation_categories, tuple(selected + [clue]), DeductionGridState())
        if analysis.contradiction:
            continue
        selected.append(clue)
        added += 1
    return tuple(selected)


def _progress_score(previous: DeductionAnalysis, current: DeductionAnalysis, total_true_pairs: int) -> float:
    resolved_gain = current.num_resolved_pairs - previous.num_resolved_pairs
    eliminated_gain = current.num_eliminated_pairs - previous.num_eliminated_pairs
    return (resolved_gain + 0.25 * eliminated_gain) / max(1, total_true_pairs)


def _observation_table(world: DeductionGridWorld, analysis: DeductionAnalysis) -> dict[str, Any]:
    table: dict[str, Any] = {}
    relation_lookup = _relation_lookup(world)
    for category_name, rows in analysis.possible.items():
        category_values = relation_lookup[category_name].values
        table[category_name] = {}
        for entity, possible_values in rows.items():
            table[category_name][entity] = {
                "possible_values": list(possible_values),
                "resolved": len(possible_values) == 1,
                "ruled_out_values": [value for value in category_values if value not in possible_values],
            }
    return table


def _commit_ready(analysis: DeductionAnalysis) -> bool:
    return analysis.solved and not analysis.pending_true and not analysis.pending_false


class DeductionActionVerifier(BaseVerifier):
    name = "deduction_action_validity"

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
                message="Each action must be a legal symbolic deduction move under the current closure state.",
            ),
        )


class DeductionStateUpdateVerifier(BaseVerifier):
    name = "deduction_state_update_correctness"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        previous_analysis = _analyze_state(
            context.world.base_category,
            context.world.relation_categories,
            context.world.clues,
            context.previous_state,
        )
        passed = False
        if context.transition.invalid_action:
            passed = context.next_state == context.previous_state
        elif context.action.name == "assert_pair":
            fact = _fact_from_action(context.action)
            passed = (
                fact is not None
                and fact in set(previous_analysis.pending_true)
                and fact in set(context.next_state.known_true)
                and set(context.next_state.known_true) == set(context.previous_state.known_true) | {fact}
                and set(context.next_state.known_false) == set(context.previous_state.known_false)
            )
        elif context.action.name == "rule_out_pair":
            fact = _fact_from_action(context.action)
            passed = (
                fact is not None
                and fact in set(previous_analysis.pending_false)
                and fact in set(context.next_state.known_false)
                and set(context.next_state.known_false) == set(context.previous_state.known_false) | {fact}
                and set(context.next_state.known_true) == set(context.previous_state.known_true)
            )
        elif context.action.name == "propagate":
            passed = (
                set(context.next_state.known_true) == set(context.previous_state.known_true) | set(previous_analysis.pending_true)
                and set(context.next_state.known_false) == set(context.previous_state.known_false) | set(previous_analysis.pending_false)
            )
        elif context.action.name == "commit_solution":
            assignment = _normalize_assignment(context.action.arguments.get("assignment"), context.world)
            passed = (
                assignment is not None
                and context.next_state.committed_assignment == assignment
                and context.transition.terminated
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
                message="The symbolic deduction table must update exactly as required by the chosen operation.",
            ),
        )


class DeductionGoalVerifier(BaseVerifier):
    name = "deduction_goal_reached"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        if not context.transition.terminated:
            return ()
        assignment = context.next_state.committed_assignment or {}
        passed = context.transition.success and _assignment_satisfies_world(context.world, assignment)
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.GOAL,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="Final success requires a complete bijective assignment satisfying all clues.",
            ),
        )


class DeductionTrajectoryVerifier(BaseVerifier):
    def evaluate_trajectory(self, context: TrajectoryContext) -> tuple[VerificationResult, ...]:
        final_analysis = _analyze_state(
            context.world.base_category,
            context.world.relation_categories,
            context.world.clues,
            DeductionGridState(
                known_true=context.final_state.known_true,
                known_false=context.final_state.known_false,
            ),
        )
        committed_assignment = context.final_state.committed_assignment or {}
        assignment_valid = bool(committed_assignment) and _assignment_satisfies_world(context.world, committed_assignment)
        observed_steps = len(context.trace.steps) + 1
        oracle_steps = len(context.world.oracle_plan)
        efficient = context.success and observed_steps == oracle_steps
        return (
            VerificationResult(
                name="deduction_table_consistency",
                scope=VerificationScope.TRAJECTORY,
                passed=not final_analysis.contradiction,
                score=1.0 if not final_analysis.contradiction else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="The evolving deduction table must remain globally consistent under bijection and clue constraints.",
            ),
            VerificationResult(
                name="deduction_solution_validity",
                scope=VerificationScope.TRAJECTORY,
                passed=assignment_valid,
                score=1.0 if assignment_valid else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="The committed solution must satisfy all clue semantics and bijection constraints.",
            ),
            VerificationResult(
                name="deduction_trajectory_efficiency",
                scope=VerificationScope.TRAJECTORY,
                passed=efficient,
                score=1.0 if efficient else max(0.0, 1.0 - max(0, observed_steps - oracle_steps) / max(1, oracle_steps)),
                kind=VerificationKind.QUALITY,
                weight=1.0,
                hard=False,
                message="Quality measures how closely the trajectory matches the oracle deduction length.",
                metadata={"observed_steps": observed_steps, "oracle_steps": oracle_steps},
            ),
        )


class DeductionGridOracle(Oracle):
    def __init__(self, world: DeductionGridWorld) -> None:
        self.world = world

    def is_feasible(self) -> bool | None:
        return True

    def estimate_difficulty(self) -> float | None:
        return float(
            len(self.world.oracle_plan)
            + len(self.world.clues) / 2
            + len(self.world.base_category.values) * len(self.world.relation_categories)
        )

    def solve(self) -> OracleSolution:
        return OracleSolution(
            actions=self.world.oracle_plan,
            metadata={
                "base_category": self.world.base_category.name,
                "num_entities": len(self.world.base_category.values),
                "num_relation_categories": len(self.world.relation_categories),
                "num_clues": len(self.world.clues),
            },
            objective_value=len(self.world.oracle_plan),
            feasible=True,
            optimal=True,
            difficulty_estimate=self.estimate_difficulty(),
            certificate=ProofCertificate(
                feasible=True,
                optimal=True,
                summary="Oracle plan is produced by replaying the deterministic closure-derived deduction policy to a unique full assignment.",
                witness={
                    "num_actions": len(self.world.oracle_plan),
                    "num_clues": len(self.world.clues),
                    "num_entities": len(self.world.base_category.values),
                    "num_relation_categories": len(self.world.relation_categories),
                },
            ),
        )


class DeductionGridFamily(EnvironmentFamily):
    name = "deduction_grid"
    description = "Logic-grid deduction puzzles with exact symbolic table updates, propagation, and commit verification."

    def sample_generation_params(self, config: FamilyConfig, rng: random.Random) -> dict[str, Any]:
        difficulty_specs = {
            "easy": {"num_entities": 3, "num_relation_categories": 2, "num_distractor_clues": (0, 1), "prefer_relational": False},
            "medium": {"num_entities": 4, "num_relation_categories": 3, "num_distractor_clues": (1, 2), "prefer_relational": True},
            "hard": {"num_entities": 5, "num_relation_categories": 4, "num_distractor_clues": (2, 3), "prefer_relational": True},
        }
        spec = difficulty_specs.get(config.difficulty, difficulty_specs["medium"])
        return {
            "num_entities": spec["num_entities"],
            "num_relation_categories": spec["num_relation_categories"],
            "num_distractor_clues": rng.randint(*spec["num_distractor_clues"]),
            "prefer_relational": spec["prefer_relational"],
            "observability": config.observability,
        }

    def sample_world(self, generation_params: dict[str, Any], rng: random.Random) -> DeductionGridWorld:
        num_entities = int(generation_params["num_entities"])
        num_relation_categories = int(generation_params["num_relation_categories"])
        base_template = CATEGORY_LIBRARY[0]
        base_category = DeductionCategory(base_template.name, base_template.values[:num_entities])
        candidate_relations = [DeductionCategory(category.name, category.values[:num_entities]) for category in CATEGORY_LIBRARY[1:]]
        relation_categories = tuple(rng.sample(candidate_relations, num_relation_categories))
        entities = base_category.values
        assignment: dict[str, dict[str, str]] = {}
        for category in relation_categories:
            shuffled_values = list(category.values)
            rng.shuffle(shuffled_values)
            assignment[category.name] = {entity: shuffled_values[index] for index, entity in enumerate(entities)}
        candidate_clues = _build_candidate_clues(
            relation_categories,
            assignment,
            entities,
            rng,
            prefer_relational=bool(generation_params["prefer_relational"]),
        )
        clues = _select_clues(
            base_category,
            relation_categories,
            candidate_clues,
            rng,
            num_distractor_clues=int(generation_params["num_distractor_clues"]),
        )
        oracle_plan = _build_oracle_plan(base_category, relation_categories, clues)
        return DeductionGridWorld(
            base_category=base_category,
            relation_categories=relation_categories,
            assignment=assignment,
            clues=clues,
            oracle_plan=oracle_plan,
        )

    def derive_objective(
        self,
        world: DeductionGridWorld,
        generation_params: dict[str, Any],
        rng: random.Random,
    ) -> TaskObjective:
        return TaskObjective(
            name="solve_deduction_grid",
            description="Recover the hidden bijective assignment by making legal symbolic deduction updates to the grid and then committing a complete solution.",
            success_criteria={
                "complete_assignment": True,
                "satisfy_all_clues": True,
                "respect_bijection_constraints": True,
            },
            optimization_target="minimize_number_of_deduction_steps",
            constraint_spec={
                "legal_actions": ["assert_pair", "rule_out_pair", "propagate", "commit_solution"],
                "base_category": world.base_category.name,
                "relation_categories": [category.name for category in world.relation_categories],
            },
            quality_metric="deduction_steps",
        )

    def initial_state(
        self,
        world: DeductionGridWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> DeductionGridState:
        return DeductionGridState()

    def observe(
        self,
        world: DeductionGridWorld,
        state: DeductionGridState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        analysis = _analyze_state(world.base_category, world.relation_categories, world.clues, state)
        return {
            "mode": generation_params["observability"],
            "base_category": world.base_category.name,
            "entities": list(world.base_category.values),
            "relation_categories": [
                {"name": category.name, "values": list(category.values)}
                for category in world.relation_categories
            ],
            "clues": [
                {
                    "clue_id": clue.clue_id,
                    "clue_type": clue.clue_type,
                    "positive": clue.positive,
                    "text": _render_clue_text(clue),
                }
                for clue in world.clues
            ],
            "deduction_table": _observation_table(world, analysis),
            "known_true": [_fact_to_dict(fact) for fact in state.known_true],
            "known_false": [_fact_to_dict(fact) for fact in state.known_false],
            "pending_true": [_fact_to_dict(fact) for fact in analysis.pending_true],
            "pending_false": [_fact_to_dict(fact) for fact in analysis.pending_false],
            "resolved_assignment": analysis.resolved_assignment,
            "ready_to_commit": _commit_ready(analysis),
            "committed_assignment": state.committed_assignment,
            "solved": state.solved,
        }

    def build_task_space(
        self,
        world: DeductionGridWorld,
        objective: TaskObjective,
        initial_state: DeductionGridState,
        generation_params: dict[str, Any],
    ) -> TaskSpace:
        return TaskSpace(
            observation_schema={
                "type": "object",
                "fields": {
                    "clues": "list[structured_clue]",
                    "deduction_table": "category -> entity -> possible_values",
                    "known_true": "list[deduction_fact]",
                    "known_false": "list[deduction_fact]",
                    "resolved_assignment": "partial_or_complete_assignment",
                },
                "observability": generation_params["observability"],
            },
            action_schema={
                "type": "canonical_object",
                "actions": [
                    {"name": "assert_pair", "arguments": {"category": "string", "entity": "string", "value": "string"}},
                    {"name": "rule_out_pair", "arguments": {"category": "string", "entity": "string", "value": "string"}},
                    {"name": "propagate", "arguments": {}},
                    {"name": "commit_solution", "arguments": {"assignment": "category -> entity -> value"}},
                ],
            },
            runtime_api="gym_like",
            notes="Gym-like reset()/step() tuples over a formal deduction table with exact propagation semantics.",
        )

    def valid_actions(
        self,
        world: DeductionGridWorld,
        state: DeductionGridState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> list[CanonicalAction]:
        analysis = _analyze_state(world.base_category, world.relation_categories, world.clues, state)
        actions: list[CanonicalAction] = []
        actions.extend(CanonicalAction(name="assert_pair", arguments=_fact_to_dict(fact)) for fact in analysis.pending_true)
        actions.extend(CanonicalAction(name="rule_out_pair", arguments=_fact_to_dict(fact)) for fact in analysis.pending_false)
        if analysis.pending_true or analysis.pending_false:
            actions.append(
                CanonicalAction(
                    name="propagate",
                    arguments={
                        "num_true_updates": len(analysis.pending_true),
                        "num_false_updates": len(analysis.pending_false),
                    },
                )
            )
        if _commit_ready(analysis):
            actions.append(CanonicalAction(name="commit_solution", arguments={"assignment": analysis.resolved_assignment}))
        return actions

    def transition(
        self,
        world: DeductionGridWorld,
        state: DeductionGridState,
        action: CanonicalAction,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> TransitionResult:
        analysis = _analyze_state(world.base_category, world.relation_categories, world.clues, state)
        total_true_pairs = len(world.base_category.values) * len(world.relation_categories)

        if action.name == "assert_pair":
            fact = _fact_from_action(action)
            if fact is None or fact not in set(analysis.pending_true):
                return TransitionResult(
                    next_state=state,
                    invalid_action=True,
                    reward_hints={"deduction_progress": -0.25},
                    info={"invalid_action": True, "reason": "assertion_not_entailed"},
                )
            next_state = DeductionGridState(
                known_true=_sorted_facts(set(state.known_true) | {fact}),
                known_false=state.known_false,
            )
            next_analysis = _analyze_state(world.base_category, world.relation_categories, world.clues, next_state)
            return TransitionResult(
                next_state=next_state,
                reward_hints={"deduction_progress": _progress_score(analysis, next_analysis, total_true_pairs)},
                info={"applied_operation": "assert_pair", "fact": _fact_to_dict(fact)},
            )

        if action.name == "rule_out_pair":
            fact = _fact_from_action(action)
            if fact is None or fact not in set(analysis.pending_false):
                return TransitionResult(
                    next_state=state,
                    invalid_action=True,
                    reward_hints={"deduction_progress": -0.25},
                    info={"invalid_action": True, "reason": "elimination_not_entailed"},
                )
            next_state = DeductionGridState(
                known_true=state.known_true,
                known_false=_sorted_facts(set(state.known_false) | {fact}),
            )
            next_analysis = _analyze_state(world.base_category, world.relation_categories, world.clues, next_state)
            return TransitionResult(
                next_state=next_state,
                reward_hints={"deduction_progress": _progress_score(analysis, next_analysis, total_true_pairs)},
                info={"applied_operation": "rule_out_pair", "fact": _fact_to_dict(fact)},
            )

        if action.name == "propagate":
            if not analysis.pending_true and not analysis.pending_false:
                return TransitionResult(
                    next_state=state,
                    invalid_action=True,
                    reward_hints={"deduction_progress": -0.1},
                    info={"invalid_action": True, "reason": "no_pending_propagation"},
                )
            next_state = DeductionGridState(
                known_true=_sorted_facts(set(state.known_true) | set(analysis.pending_true)),
                known_false=_sorted_facts(set(state.known_false) | set(analysis.pending_false)),
            )
            next_analysis = _analyze_state(world.base_category, world.relation_categories, world.clues, next_state)
            return TransitionResult(
                next_state=next_state,
                reward_hints={"deduction_progress": _progress_score(analysis, next_analysis, total_true_pairs)},
                info={
                    "applied_operation": "propagate",
                    "num_true_updates": len(analysis.pending_true),
                    "num_false_updates": len(analysis.pending_false),
                },
            )

        if action.name == "commit_solution":
            assignment = _normalize_assignment(action.arguments.get("assignment"), world)
            if assignment is None:
                return TransitionResult(
                    next_state=state,
                    invalid_action=True,
                    reward_hints={"deduction_progress": -0.5},
                    info={"invalid_action": True, "reason": "malformed_commit_assignment"},
                )
            if not _commit_ready(analysis):
                return TransitionResult(
                    next_state=state,
                    invalid_action=True,
                    reward_hints={"deduction_progress": -0.5},
                    info={"invalid_action": True, "reason": "commit_before_table_complete"},
                )
            success = _assignment_satisfies_world(world, assignment)
            next_state = DeductionGridState(
                known_true=state.known_true,
                known_false=state.known_false,
                committed_assignment=assignment,
                solved=success,
            )
            return TransitionResult(
                next_state=next_state,
                terminated=True,
                success=success,
                reward_hints={"deduction_progress": 1.0 if success else -1.0},
                info={
                    "applied_operation": "commit_solution",
                    "commit_valid": success,
                    "resolved_ready": _commit_ready(analysis),
                },
            )

        return TransitionResult(
            next_state=state,
            invalid_action=True,
            reward_hints={"deduction_progress": -0.25},
            info={"invalid_action": True, "reason": "unknown_action"},
        )

    def build_verifier_suite(
        self,
        world: DeductionGridWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> VerifierSuite:
        return VerifierSuite(
            [
                DeductionActionVerifier(),
                DeductionStateUpdateVerifier(),
                DeductionGoalVerifier(),
                DeductionTrajectoryVerifier(),
            ]
        )

    def build_oracle(
        self,
        world: DeductionGridWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> DeductionGridOracle:
        return DeductionGridOracle(world)

    def recommended_max_steps(self, generation_params: dict[str, Any]) -> int:
        total_pairs = int(generation_params["num_entities"]) * int(generation_params["num_relation_categories"])
        return max(24, total_pairs * 4)

    def task_metadata(
        self,
        world: DeductionGridWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        clue_type_histogram: dict[str, int] = {}
        for clue in world.clues:
            key = f"{clue.clue_type}_{'positive' if clue.positive else 'negative'}"
            clue_type_histogram[key] = clue_type_histogram.get(key, 0) + 1
        return {
            "num_entities": len(world.base_category.values),
            "num_relation_categories": len(world.relation_categories),
            "num_clues": len(world.clues),
            "oracle_steps": len(world.oracle_plan),
            "base_category": world.base_category.name,
            "relation_categories": [category.name for category in world.relation_categories],
            "clue_type_histogram": clue_type_histogram,
        }
