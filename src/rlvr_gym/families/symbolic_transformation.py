from __future__ import annotations

import itertools
import random
from collections import deque
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
class SymbolicExpression:
    op: str
    value: int | str | bool | None = None
    children: tuple["SymbolicExpression", ...] = ()


@dataclass(frozen=True)
class RewritePlanStep:
    rule_id: str
    path: tuple[int, ...]


@dataclass(frozen=True)
class RewriteCandidate:
    rule_id: str
    path: tuple[int, ...]
    result: SymbolicExpression
    description: str


@dataclass(frozen=True)
class SymbolicTransformationWorld:
    task_type: str
    source_expression: SymbolicExpression
    target_expression: SymbolicExpression
    variables: tuple[str, ...]
    allowed_rule_ids: tuple[str, ...]
    oracle_plan: tuple[RewritePlanStep, ...]


@dataclass(frozen=True)
class SymbolicTransformationState:
    current_expression: SymbolicExpression
    applied_rules: tuple[str, ...]


def _const(value: int) -> SymbolicExpression:
    return SymbolicExpression(op="const", value=value)


def _var(name: str) -> SymbolicExpression:
    return SymbolicExpression(op="var", value=name)


def _bool_const(value: bool) -> SymbolicExpression:
    return SymbolicExpression(op="bool", value=value)


def _make(op: str, left: SymbolicExpression, right: SymbolicExpression) -> SymbolicExpression:
    return SymbolicExpression(op=op, children=(left, right))


def _negate(expr: SymbolicExpression) -> SymbolicExpression:
    return SymbolicExpression(op="not", children=(expr,))


def _expr_key(expr: SymbolicExpression) -> tuple[Any, ...]:
    child_keys = tuple(_expr_key(child) for child in expr.children)
    return (expr.op, expr.value, child_keys)


def _expr_to_dict(expr: SymbolicExpression) -> dict[str, Any]:
    return {
        "op": expr.op,
        "value": expr.value,
        "children": [_expr_to_dict(child) for child in expr.children],
    }


def _expr_to_string(expr: SymbolicExpression) -> str:
    if expr.op == "const":
        return str(expr.value)
    if expr.op == "bool":
        return "true" if expr.value else "false"
    if expr.op == "var":
        return str(expr.value)
    if expr.op == "not":
        child = expr.children[0]
        return f"~{_expr_to_string(child)}" if child.op in {"var", "bool"} else f"~({_expr_to_string(child)})"
    symbols = {"add": "+", "mul": "*", "and": "&", "or": "|"}
    left, right = expr.children
    return f"({_expr_to_string(left)} {symbols[expr.op]} {_expr_to_string(right)})"


def _count_nodes(expr: SymbolicExpression) -> int:
    return 1 + sum(_count_nodes(child) for child in expr.children)


def _replace_at_path(expr: SymbolicExpression, path: tuple[int, ...], replacement: SymbolicExpression) -> SymbolicExpression:
    if not path:
        return replacement
    child_index = path[0]
    children = list(expr.children)
    children[child_index] = _replace_at_path(children[child_index], path[1:], replacement)
    return SymbolicExpression(op=expr.op, value=expr.value, children=tuple(children))


def _iter_paths(expr: SymbolicExpression, prefix: tuple[int, ...] = ()) -> list[tuple[tuple[int, ...], SymbolicExpression]]:
    paths = [(prefix, expr)]
    for index, child in enumerate(expr.children):
        paths.extend(_iter_paths(child, prefix + (index,)))
    return paths


def _path_label(path: tuple[int, ...]) -> str:
    return "root" if not path else ".".join(str(index) for index in path)


def _is_const(expr: SymbolicExpression, value: int | None = None) -> bool:
    return expr.op == "const" and (value is None or expr.value == value)


def _is_bool_atom(expr: SymbolicExpression) -> bool:
    return expr.op in {"var", "bool"} or (expr.op == "not" and expr.children[0].op in {"var", "bool"})


def _canonicalize_arithmetic(expr: SymbolicExpression) -> SymbolicExpression:
    if expr.op in {"const", "var"}:
        return expr
    left = _canonicalize_arithmetic(expr.children[0])
    right = _canonicalize_arithmetic(expr.children[1])
    if expr.op == "add":
        if left.op == "add":
            return _canonicalize_arithmetic(_make("add", left.children[0], _make("add", left.children[1], right)))
        if _is_const(left, 0):
            return right
        if _is_const(right, 0):
            return left
        if _is_const(left) and _is_const(right):
            return _const(int(left.value) + int(right.value))
        if _expr_key(left) > _expr_key(right):
            return _make("add", right, left)
        return _make("add", left, right)
    if left.op == "mul":
        return _canonicalize_arithmetic(_make("mul", left.children[0], _make("mul", left.children[1], right)))
    if _is_const(left, 0) or _is_const(right, 0):
        return _const(0)
    if _is_const(left, 1):
        return right
    if _is_const(right, 1):
        return left
    if _is_const(left) and _is_const(right):
        return _const(int(left.value) * int(right.value))
    if _expr_key(left) > _expr_key(right):
        return _make("mul", right, left)
    return _make("mul", left, right)


def _canonicalize_boolean(expr: SymbolicExpression) -> SymbolicExpression:
    if expr.op in {"var", "bool"}:
        return expr
    if expr.op == "not":
        child = _canonicalize_boolean(expr.children[0])
        if child.op == "not":
            return _canonicalize_boolean(child.children[0])
        if child.op == "and":
            return _canonicalize_boolean(_make("or", _negate(child.children[0]), _negate(child.children[1])))
        if child.op == "or":
            return _canonicalize_boolean(_make("and", _negate(child.children[0]), _negate(child.children[1])))
        return _negate(child)
    left = _canonicalize_boolean(expr.children[0])
    right = _canonicalize_boolean(expr.children[1])
    if left.op == expr.op:
        return _canonicalize_boolean(_make(expr.op, left.children[0], _make(expr.op, left.children[1], right)))
    if _expr_key(left) > _expr_key(right):
        return SymbolicExpression(op=expr.op, children=(right, left))
    return SymbolicExpression(op=expr.op, children=(left, right))


def _is_arithmetic_normal_form(expr: SymbolicExpression) -> bool:
    return expr == _canonicalize_arithmetic(expr)


def _is_boolean_nnf(expr: SymbolicExpression) -> bool:
    return expr == _canonicalize_boolean(expr) and (
        expr.op in {"var", "bool"}
        or (expr.op == "not" and expr.children[0].op in {"var", "bool"})
        or (expr.op in {"and", "or"} and all(_is_boolean_nnf(child) for child in expr.children))
    )


def _enumerate_forward_rewrites(expr: SymbolicExpression, task_type: str) -> list[RewriteCandidate]:
    candidates: list[RewriteCandidate] = []
    for path, node in _iter_paths(expr):
        local: list[tuple[str, SymbolicExpression, str]] = []
        if task_type == "arithmetic_simplify":
            if node.op == "add":
                left, right = node.children
                if left.op == "add":
                    local.append(("assoc_add", _make("add", left.children[0], _make("add", left.children[1], right)), "Reassociate addition to the right."))
                if _is_const(left, 0):
                    local.append(("remove_add_zero_left", right, "Remove additive identity on the left."))
                if _is_const(right, 0):
                    local.append(("remove_add_zero_right", left, "Remove additive identity on the right."))
                if _is_const(left) and _is_const(right):
                    local.append(("fold_add_constants", _const(int(left.value) + int(right.value)), "Fold a constant addition."))
                if _expr_key(left) > _expr_key(right):
                    local.append(("sort_add_operands", _make("add", right, left), "Swap addition operands into canonical order."))
            elif node.op == "mul":
                left, right = node.children
                if left.op == "mul":
                    local.append(("assoc_mul", _make("mul", left.children[0], _make("mul", left.children[1], right)), "Reassociate multiplication to the right."))
                if _is_const(left, 0) or _is_const(right, 0):
                    local.append(("collapse_mul_zero", _const(0), "Collapse multiplication by zero."))
                if _is_const(left, 1):
                    local.append(("remove_mul_one_left", right, "Remove multiplicative identity on the left."))
                if _is_const(right, 1):
                    local.append(("remove_mul_one_right", left, "Remove multiplicative identity on the right."))
                if _is_const(left) and _is_const(right):
                    local.append(("fold_mul_constants", _const(int(left.value) * int(right.value)), "Fold a constant multiplication."))
                if _expr_key(left) > _expr_key(right):
                    local.append(("sort_mul_operands", _make("mul", right, left), "Swap multiplication operands into canonical order."))
        else:
            if node.op == "not":
                child = node.children[0]
                if child.op == "not":
                    local.append(("double_negation", child.children[0], "Eliminate a double negation."))
                if child.op == "and":
                    local.append(("de_morgan_and", _make("or", _negate(child.children[0]), _negate(child.children[1])), "Push negation through conjunction."))
                if child.op == "or":
                    local.append(("de_morgan_or", _make("and", _negate(child.children[0]), _negate(child.children[1])), "Push negation through disjunction."))
            elif node.op in {"and", "or"}:
                left, right = node.children
                if left.op == node.op:
                    local.append((f"assoc_{node.op}", SymbolicExpression(op=node.op, children=(left.children[0], SymbolicExpression(op=node.op, children=(left.children[1], right)))), f"Reassociate {node.op} to the right."))
                if _expr_key(left) > _expr_key(right):
                    local.append((f"sort_{node.op}_operands", SymbolicExpression(op=node.op, children=(right, left)), f"Swap {node.op} operands into canonical order."))
        for rule_id, rewritten, description in local:
            candidates.append(
                RewriteCandidate(
                    rule_id=rule_id,
                    path=path,
                    result=_replace_at_path(expr, path, rewritten),
                    description=description,
                )
            )
    return sorted(candidates, key=lambda candidate: (len(candidate.path), candidate.path, candidate.rule_id, _expr_to_string(candidate.result)))


def _enumerate_inverse_rewrites(expr: SymbolicExpression, task_type: str) -> list[RewriteCandidate]:
    candidates: list[RewriteCandidate] = []
    for path, node in _iter_paths(expr):
        local: list[tuple[str, SymbolicExpression, str]] = []
        if task_type == "arithmetic_simplify":
            local.append(("introduce_add_zero_left", _make("add", _const(0), node), "Introduce an additive identity on the left."))
            local.append(("introduce_add_zero_right", _make("add", node, _const(0)), "Introduce an additive identity on the right."))
            local.append(("introduce_mul_one_left", _make("mul", _const(1), node), "Introduce a multiplicative identity on the left."))
            local.append(("introduce_mul_one_right", _make("mul", node, _const(1)), "Introduce a multiplicative identity on the right."))
            if node.op == "const" and int(node.value) >= 2:
                split = max(1, int(node.value) // 2)
                local.append(("split_add_constant", _make("add", _const(split), _const(int(node.value) - split)), "Split a constant into an addition."))
            if node.op in {"add", "mul"}:
                left, right = node.children
                local.append((f"unsort_{node.op}_operands", SymbolicExpression(op=node.op, children=(right, left)), f"Undo canonical operand ordering for {node.op}."))
                if right.op == node.op:
                    local.append((f"left_assoc_{node.op}", SymbolicExpression(op=node.op, children=(SymbolicExpression(op=node.op, children=(left, right.children[0])), right.children[1])), f"Undo right association for {node.op}."))
        else:
            local.append(("introduce_double_negation", _negate(_negate(node)), "Introduce a double negation."))
            if node.op == "or" and all(child.op == "not" for child in node.children):
                local.append(("inverse_de_morgan_and", _negate(_make("and", node.children[0].children[0], node.children[1].children[0])), "Combine negated disjuncts into a negated conjunction."))
            if node.op == "and" and all(child.op == "not" for child in node.children):
                local.append(("inverse_de_morgan_or", _negate(_make("or", node.children[0].children[0], node.children[1].children[0])), "Combine negated conjuncts into a negated disjunction."))
            if node.op in {"and", "or"}:
                left, right = node.children
                local.append((f"unsort_{node.op}_operands", SymbolicExpression(op=node.op, children=(right, left)), f"Undo canonical operand ordering for {node.op}."))
                if right.op == node.op:
                    local.append((f"left_assoc_{node.op}", SymbolicExpression(op=node.op, children=(SymbolicExpression(op=node.op, children=(left, right.children[0])), right.children[1])), f"Undo right association for {node.op}."))
        for rule_id, rewritten, description in local:
            candidates.append(
                RewriteCandidate(
                    rule_id=rule_id,
                    path=path,
                    result=_replace_at_path(expr, path, rewritten),
                    description=description,
                )
            )
    return sorted(candidates, key=lambda candidate: (len(candidate.path), candidate.path, candidate.rule_id, _expr_to_string(candidate.result)))


def _find_rewrite_sequence(start: SymbolicExpression, target: SymbolicExpression, task_type: str) -> tuple[RewritePlanStep, ...] | None:
    if start == target:
        return ()
    queue: deque[tuple[SymbolicExpression, tuple[RewritePlanStep, ...]]] = deque([(start, ())])
    visited = {start}
    while queue:
        current, plan = queue.popleft()
        for candidate in _enumerate_forward_rewrites(current, task_type):
            next_expression = candidate.result
            if next_expression in visited:
                continue
            next_plan = plan + (RewritePlanStep(rule_id=candidate.rule_id, path=candidate.path),)
            if next_expression == target:
                return next_plan
            visited.add(next_expression)
            if len(visited) > 10000:
                return None
            queue.append((next_expression, next_plan))
    return None


def _structural_distance(left: SymbolicExpression, right: SymbolicExpression) -> int:
    if left == right:
        return 0
    if left.op != right.op or left.value != right.value or len(left.children) != len(right.children):
        return 1 + _count_nodes(left) + _count_nodes(right)
    return 1 + sum(_structural_distance(left_child, right_child) for left_child, right_child in zip(left.children, right.children))


def _eval_arithmetic(expr: SymbolicExpression, assignment: dict[str, int]) -> int:
    if expr.op == "const":
        return int(expr.value)
    if expr.op == "var":
        return assignment[str(expr.value)]
    if expr.op == "add":
        return _eval_arithmetic(expr.children[0], assignment) + _eval_arithmetic(expr.children[1], assignment)
    return _eval_arithmetic(expr.children[0], assignment) * _eval_arithmetic(expr.children[1], assignment)


def _eval_boolean(expr: SymbolicExpression, assignment: dict[str, bool]) -> bool:
    if expr.op == "bool":
        return bool(expr.value)
    if expr.op == "var":
        return assignment[str(expr.value)]
    if expr.op == "not":
        return not _eval_boolean(expr.children[0], assignment)
    if expr.op == "and":
        return _eval_boolean(expr.children[0], assignment) and _eval_boolean(expr.children[1], assignment)
    return _eval_boolean(expr.children[0], assignment) or _eval_boolean(expr.children[1], assignment)


def _semantically_equivalent(world: SymbolicTransformationWorld, expr: SymbolicExpression) -> bool:
    if world.task_type == "arithmetic_simplify":
        domains = list(itertools.product([0, 1, 2], repeat=len(world.variables))) or [()]
        for values in domains:
            assignment = {name: value for name, value in zip(world.variables, values)}
            if _eval_arithmetic(expr, assignment) != _eval_arithmetic(world.source_expression, assignment):
                return False
        return True
    domains = list(itertools.product([False, True], repeat=len(world.variables))) or [()]
    for values in domains:
        assignment = {name: value for name, value in zip(world.variables, values)}
        if _eval_boolean(expr, assignment) != _eval_boolean(world.source_expression, assignment):
            return False
    return True


def _generate_arithmetic_target(rng: random.Random, depth: int, variables: tuple[str, ...]) -> SymbolicExpression:
    if depth <= 0 or rng.random() < 0.4:
        return _const(rng.randint(0, 4)) if rng.random() < 0.5 else _var(rng.choice(variables))
    op = rng.choice(["add", "mul"])
    return _canonicalize_arithmetic(_make(op, _generate_arithmetic_target(rng, depth - 1, variables), _generate_arithmetic_target(rng, depth - 1, variables)))


def _generate_boolean_target(rng: random.Random, depth: int, variables: tuple[str, ...]) -> SymbolicExpression:
    if depth <= 0 or rng.random() < 0.35:
        literal = _var(rng.choice(variables))
        return literal if rng.random() < 0.5 else _negate(literal)
    op = rng.choice(["and", "or"])
    return _canonicalize_boolean(SymbolicExpression(op=op, children=(_generate_boolean_target(rng, depth - 1, variables), _generate_boolean_target(rng, depth - 1, variables))))


def _rule_inventory(task_type: str) -> tuple[str, ...]:
    if task_type == "arithmetic_simplify":
        return (
            "assoc_add",
            "assoc_mul",
            "collapse_mul_zero",
            "fold_add_constants",
            "fold_mul_constants",
            "remove_add_zero_left",
            "remove_add_zero_right",
            "remove_mul_one_left",
            "remove_mul_one_right",
            "sort_add_operands",
            "sort_mul_operands",
        )
    return (
        "assoc_and",
        "assoc_or",
        "de_morgan_and",
        "de_morgan_or",
        "double_negation",
        "sort_and_operands",
        "sort_or_operands",
    )


class SymbolicActionVerifier(BaseVerifier):
    name = "symbolic_rewrite_action_validity"

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
                message="The selected rewrite rule and tree path must identify a legal local rewrite.",
            ),
        )


class SymbolicRewriteResultVerifier(BaseVerifier):
    name = "symbolic_rewrite_result_correctness"

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
                    message="Invalid rewrites do not produce a correct legal successor state.",
                ),
            )
        path = tuple(context.action.arguments.get("path", ()))
        rule_id = context.action.arguments.get("rule_id")
        legal_map = {
            (candidate.rule_id, candidate.path): candidate.result
            for candidate in _enumerate_forward_rewrites(context.previous_state.current_expression, context.world.task_type)
        }
        expected = legal_map.get((rule_id, path))
        passed = expected == context.next_state.current_expression
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.STATE,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="The resulting symbolic tree must exactly match the legal rewritten tree.",
            ),
        )


class SymbolicGoalVerifier(BaseVerifier):
    name = "symbolic_goal_reached"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        if not context.transition.terminated:
            return ()
        passed = context.next_state.current_expression == context.world.target_expression and context.transition.success
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.GOAL,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="Termination should coincide with reaching the target symbolic form.",
            ),
        )


class SymbolicTrajectoryVerifier(BaseVerifier):
    def evaluate_trajectory(self, context: TrajectoryContext) -> tuple[VerificationResult, ...]:
        final_expression = context.final_state.current_expression
        semantic_ok = _semantically_equivalent(context.world, final_expression)
        if context.world.task_type == "arithmetic_simplify":
            normal_form_ok = _is_arithmetic_normal_form(final_expression)
        else:
            normal_form_ok = _is_boolean_nnf(final_expression)
        observed_steps = len(context.trace.steps) + 1
        oracle_steps = len(context.world.oracle_plan)
        efficient = context.success and observed_steps == oracle_steps
        return (
            VerificationResult(
                name="symbolic_semantic_equivalence",
                scope=VerificationScope.TRAJECTORY,
                passed=semantic_ok,
                score=1.0 if semantic_ok else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="The final symbolic object must remain semantically equivalent to the source expression.",
            ),
            VerificationResult(
                name="symbolic_target_normal_form",
                scope=VerificationScope.TRAJECTORY,
                passed=normal_form_ok,
                score=1.0 if normal_form_ok else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="The final symbolic object must satisfy the target normal-form constraint.",
            ),
            VerificationResult(
                name="symbolic_transformation_efficiency",
                scope=VerificationScope.TRAJECTORY,
                passed=efficient,
                score=1.0 if efficient else max(0.0, 1.0 - max(0, observed_steps - oracle_steps) / max(1, oracle_steps)),
                kind=VerificationKind.QUALITY,
                weight=1.0,
                hard=False,
                message="Quality tracks how closely the trajectory matches the shortest oracle rewrite sequence.",
                metadata={"observed_steps": observed_steps, "oracle_steps": oracle_steps},
            ),
        )


class SymbolicTransformationOracle(Oracle):
    def __init__(self, world: SymbolicTransformationWorld) -> None:
        self.world = world

    def is_feasible(self) -> bool | None:
        return True

    def estimate_difficulty(self) -> float | None:
        return float(len(self.world.oracle_plan) + _count_nodes(self.world.source_expression) / 2)

    def solve(self) -> OracleSolution:
        actions = tuple(
            {
                "name": "rewrite",
                "arguments": {"rule_id": step.rule_id, "path": list(step.path), "path_str": _path_label(step.path)},
            }
            for step in self.world.oracle_plan
        )
        return OracleSolution(
            actions=actions,
            metadata={"task_type": self.world.task_type, "target_expression": _expr_to_string(self.world.target_expression)},
            objective_value=len(self.world.oracle_plan),
            feasible=True,
            optimal=True,
            difficulty_estimate=self.estimate_difficulty(),
            certificate=ProofCertificate(
                feasible=True,
                optimal=True,
                summary="Shortest rewrite sequence found over the explicit symbolic rewrite graph.",
                witness={
                    "task_type": self.world.task_type,
                    "num_steps": len(self.world.oracle_plan),
                    "target_expression": _expr_to_string(self.world.target_expression),
                },
            ),
        )


class SymbolicTransformationFamily(EnvironmentFamily):
    name = "symbolic_transformation"
    description = "Formal symbolic tree transformation via explicit local rewrite rules."

    def sample_generation_params(self, config: FamilyConfig, rng: random.Random) -> dict[str, Any]:
        difficulty_ranges = {
            "easy": {"depth": (1, 2), "inverse_steps": (2, 4), "variables": 2},
            "medium": {"depth": (2, 3), "inverse_steps": (4, 6), "variables": 3},
            "hard": {"depth": (3, 4), "inverse_steps": (6, 8), "variables": 3},
        }
        spec = difficulty_ranges.get(config.difficulty, difficulty_ranges["medium"])
        return {
            "task_type": rng.choice(["arithmetic_simplify", "boolean_nnf"]),
            "target_depth": rng.randint(*spec["depth"]),
            "inverse_steps": rng.randint(*spec["inverse_steps"]),
            "num_variables": spec["variables"],
            "observability": config.observability,
        }

    def sample_world(self, generation_params: dict[str, Any], rng: random.Random) -> SymbolicTransformationWorld:
        task_type = str(generation_params["task_type"])
        variable_pool = ("x", "y", "z", "p", "q", "r")
        variables = tuple(variable_pool[: int(generation_params["num_variables"])])
        for _ in range(64):
            if task_type == "arithmetic_simplify":
                target = _generate_arithmetic_target(rng, int(generation_params["target_depth"]), variables)
                if target.op in {"const", "var"}:
                    target = _canonicalize_arithmetic(_make("add", target, _const(rng.randint(0, 3))))
            else:
                target = _generate_boolean_target(rng, int(generation_params["target_depth"]), variables)
            current = target
            for _ in range(int(generation_params["inverse_steps"])):
                inverse_candidates = _enumerate_inverse_rewrites(current, task_type)
                if not inverse_candidates:
                    break
                current = rng.choice(inverse_candidates).result
            oracle_plan = _find_rewrite_sequence(current, target, task_type)
            if oracle_plan is None or len(oracle_plan) == 0:
                continue
            return SymbolicTransformationWorld(
                task_type=task_type,
                source_expression=current,
                target_expression=target,
                variables=variables,
                allowed_rule_ids=_rule_inventory(task_type),
                oracle_plan=oracle_plan,
            )
        raise ValueError("Unable to generate a solvable symbolic transformation task.")

    def derive_objective(
        self,
        world: SymbolicTransformationWorld,
        generation_params: dict[str, Any],
        rng: random.Random,
    ) -> TaskObjective:
        if world.task_type == "arithmetic_simplify":
            return TaskObjective(
                name="simplify_symbolic_expression",
                description="Rewrite the arithmetic expression into its canonical symbolic form using legal local transformations.",
                success_criteria={"target_expression": _expr_to_string(world.target_expression), "target_normal_form": "canonical_arithmetic"},
                optimization_target="minimize_number_of_rewrite_steps",
                constraint_spec={"equivalent_to_source": True, "legal_local_rewrites_only": True},
                quality_metric="rewrite_steps",
            )
        return TaskObjective(
            name="normalize_formula_to_nnf",
            description="Rewrite the boolean formula into negation normal form using legal local transformations.",
            success_criteria={"target_expression": _expr_to_string(world.target_expression), "target_normal_form": "nnf"},
            optimization_target="minimize_number_of_rewrite_steps",
            constraint_spec={"equivalent_to_source": True, "legal_local_rewrites_only": True},
            quality_metric="rewrite_steps",
        )

    def initial_state(
        self,
        world: SymbolicTransformationWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> SymbolicTransformationState:
        return SymbolicTransformationState(current_expression=world.source_expression, applied_rules=())

    def observe(
        self,
        world: SymbolicTransformationWorld,
        state: SymbolicTransformationState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "mode": generation_params["observability"],
            "task_type": world.task_type,
            "current_expression": _expr_to_dict(state.current_expression),
            "current_expression_pretty": _expr_to_string(state.current_expression),
            "target_expression": _expr_to_dict(world.target_expression),
            "target_expression_pretty": _expr_to_string(world.target_expression),
            "applied_rules": list(state.applied_rules),
            "allowed_rule_ids": list(world.allowed_rule_ids),
            "num_nodes": _count_nodes(state.current_expression),
        }

    def build_task_space(
        self,
        world: SymbolicTransformationWorld,
        objective: TaskObjective,
        initial_state: SymbolicTransformationState,
        generation_params: dict[str, Any],
    ) -> TaskSpace:
        return TaskSpace(
            observation_schema={
                "type": "object",
                "fields": {
                    "current_expression": "symbolic_tree",
                    "target_expression": "symbolic_tree",
                    "applied_rules": "list[string]",
                    "allowed_rule_ids": "list[string]",
                },
                "observability": generation_params["observability"],
            },
            action_schema={
                "type": "canonical_object",
                "actions": [{"name": "rewrite", "arguments": {"rule_id": "string", "path": "list[int]"}}],
            },
            runtime_api="gym_like",
            notes="Gym-like reset()/step() tuples over formal symbolic tree rewrites.",
        )

    def valid_actions(
        self,
        world: SymbolicTransformationWorld,
        state: SymbolicTransformationState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> list[CanonicalAction]:
        return [
            CanonicalAction(
                name="rewrite",
                arguments={
                    "rule_id": candidate.rule_id,
                    "path": list(candidate.path),
                    "path_str": _path_label(candidate.path),
                    "preview": _expr_to_string(candidate.result),
                },
            )
            for candidate in _enumerate_forward_rewrites(state.current_expression, world.task_type)
        ]

    def transition(
        self,
        world: SymbolicTransformationWorld,
        state: SymbolicTransformationState,
        action: CanonicalAction,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> TransitionResult:
        legal_candidates = {
            (candidate.rule_id, candidate.path): candidate
            for candidate in _enumerate_forward_rewrites(state.current_expression, world.task_type)
        }
        path = tuple(action.arguments.get("path", ()))
        rule_id = action.arguments.get("rule_id")
        if action.name != "rewrite" or (rule_id, path) not in legal_candidates:
            return TransitionResult(
                next_state=state,
                invalid_action=True,
                reward_hints={"rewrite_progress": -0.25},
                info={"invalid_action": True, "reason": "unknown_rule_or_illegal_path"},
            )
        candidate = legal_candidates[(rule_id, path)]
        next_state = SymbolicTransformationState(
            current_expression=candidate.result,
            applied_rules=state.applied_rules + (candidate.rule_id,),
        )
        previous_distance = _structural_distance(state.current_expression, world.target_expression)
        next_distance = _structural_distance(next_state.current_expression, world.target_expression)
        success = next_state.current_expression == world.target_expression
        return TransitionResult(
            next_state=next_state,
            terminated=success,
            success=success,
            reward_hints={"rewrite_progress": (previous_distance - next_distance) / max(1, previous_distance)},
            info={
                "applied_rule_id": candidate.rule_id,
                "applied_path": list(candidate.path),
                "applied_path_str": _path_label(candidate.path),
                "result_expression_pretty": _expr_to_string(candidate.result),
                "distance_to_target": next_distance,
            },
        )

    def build_verifier_suite(
        self,
        world: SymbolicTransformationWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> VerifierSuite:
        return VerifierSuite(
            [
                SymbolicActionVerifier(),
                SymbolicRewriteResultVerifier(),
                SymbolicGoalVerifier(),
                SymbolicTrajectoryVerifier(),
            ]
        )

    def build_oracle(
        self,
        world: SymbolicTransformationWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> SymbolicTransformationOracle:
        return SymbolicTransformationOracle(world)

    def recommended_max_steps(self, generation_params: dict[str, Any]) -> int:
        return max(8, int(generation_params["inverse_steps"]) * 2 + 4)

    def task_metadata(
        self,
        world: SymbolicTransformationWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "task_type": world.task_type,
            "source_num_nodes": _count_nodes(world.source_expression),
            "target_num_nodes": _count_nodes(world.target_expression),
            "oracle_steps": len(world.oracle_plan),
        }
