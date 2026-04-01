from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlvr_gym.core.runtime import RLVREnv
from rlvr_gym.core.types import FamilyConfig
from rlvr_gym.core.utils import as_primitive, stable_hash_seed
from rlvr_gym.families.symbolic_transformation import (
    _count_nodes,
    _enumerate_forward_rewrites,
    _structural_distance,
)
from rlvr_gym.registry import get_family


BENCHMARK_SPECS: dict[str, dict[str, Any]] = {
    "deduction_grid": {
        "version": "v1",
        "description": "Frozen deduction-grid benchmark with ID and puzzle-size OOD splits.",
        "objective_gap_name": "deduction_step_gap",
        "splits": {
            "train": {"group": "id", "count": 24, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "validation": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "test": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "ood_more_entities": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"num_entities": 5, "num_relation_categories": 3, "num_distractor_clues": 2}},
            "ood_more_categories": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"num_entities": 4, "num_relation_categories": 4, "num_distractor_clues": 2}},
        },
        "evaluation_splits": ["test", "ood_more_entities", "ood_more_categories"],
        "baselines": ["random_valid", "propagate_first", "assert_first", "rule_out_first", "oracle"],
    },
    "graph_planning": {
        "version": "v1",
        "description": "Frozen graph-planning benchmark with ID and graph-structure OOD splits.",
        "objective_gap_name": "path_cost_gap",
        "splits": {
            "train": {"group": "id", "count": 24, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "validation": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "test": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "ood_larger_graphs": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"num_nodes": 12, "directed": False, "edge_probability": 0.55}},
            "ood_directed_shift": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"num_nodes": 10, "directed": True, "edge_probability": 0.45}},
        },
        "evaluation_splits": ["test", "ood_larger_graphs", "ood_directed_shift"],
        "baselines": ["random_valid", "greedy_low_cost", "greedy_goal_distance", "oracle"],
    },
    "scheduling": {
        "version": "v1",
        "description": "Frozen scheduling benchmark with ID and constraint-shift OOD splits.",
        "objective_gap_name": "tardiness_gap",
        "splits": {
            "train": {"group": "id", "count": 24, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "validation": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "test": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "ood_more_jobs": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"num_jobs": 9}},
            "ood_tighter_constraints": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"num_jobs": 8, "prerequisite_probability": 0.45, "duration_range": [2, 5]}},
        },
        "evaluation_splits": ["test", "ood_more_jobs", "ood_tighter_constraints"],
        "baselines": ["random_ready", "shortest_processing_time", "earliest_deadline", "minimum_slack", "oracle"],
    },
    "sokoban": {
        "version": "v1",
        "description": "Frozen Sokoban benchmark with ID and long-horizon planning OOD splits.",
        "objective_gap_name": "move_count_gap",
        "secondary_objective_gap_name": "push_count_gap",
        "splits": {
            "train": {"group": "id", "count": 24, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "validation": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "test": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {}},
            "ood_more_boxes": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"num_boxes": 3, "min_solution_length": 14}},
            "ood_longer_plans": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"reverse_scramble_steps": 20, "min_solution_length": 18, "template_pool": ["warehouse_large", "rooms_and_halls"]}},
        },
        "evaluation_splits": ["test", "ood_more_boxes", "ood_longer_plans"],
        "baselines": ["random_valid", "push_when_possible", "greedy_goal_progress", "deadlock_avoiding_greedy", "oracle"],
    },
    "symbolic_transformation": {
        "version": "v1",
        "description": "Frozen symbolic rewrite benchmark with mode-specific ID splits and deeper OOD splits.",
        "objective_gap_name": "rewrite_step_gap",
        "splits": {
            "arithmetic_train": {"group": "id", "count": 24, "difficulty": "medium", "observability": "full", "generation_overrides": {"task_type": "arithmetic_simplify"}},
            "arithmetic_validation": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {"task_type": "arithmetic_simplify"}},
            "arithmetic_test": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {"task_type": "arithmetic_simplify"}},
            "boolean_train": {"group": "id", "count": 24, "difficulty": "medium", "observability": "full", "generation_overrides": {"task_type": "boolean_nnf"}},
            "boolean_validation": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {"task_type": "boolean_nnf"}},
            "boolean_test": {"group": "id", "count": 12, "difficulty": "medium", "observability": "full", "generation_overrides": {"task_type": "boolean_nnf"}},
            "ood_arithmetic_deeper": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"task_type": "arithmetic_simplify", "target_depth": 4, "inverse_steps": 8}},
            "ood_boolean_deeper": {"group": "ood", "count": 12, "difficulty": "hard", "observability": "full", "generation_overrides": {"task_type": "boolean_nnf", "target_depth": 4, "inverse_steps": 8}},
        },
        "evaluation_splits": ["arithmetic_test", "boolean_test", "ood_arithmetic_deeper", "ood_boolean_deeper"],
        "baselines": ["random_valid", "greedy_target_distance", "greedy_expression_size", "rule_priority", "oracle"],
    },
}


def _config_from_split(split_spec: dict[str, Any]) -> FamilyConfig:
    return FamilyConfig(
        difficulty=split_spec["difficulty"],
        observability=split_spec["observability"],
        generation_overrides=dict(split_spec.get("generation_overrides", {})),
    )


def _seed_for(family_name: str, version: str, split_name: str, index: int) -> int:
    return stable_hash_seed("benchmark_artifacts", family_name, version, split_name, index)


def _graph_baseline_action(name: str, task: Any, observation: dict[str, Any], valid_actions: list[dict[str, Any]], rng: Random) -> dict[str, Any]:
    if name == "random_valid":
        return rng.choice(valid_actions)
    neighbors = {neighbor["target"]: neighbor["cost"] for neighbor in observation["neighbors"]}
    if name == "greedy_low_cost":
        return min(valid_actions, key=lambda action: (neighbors[action["arguments"]["target"]], action["arguments"]["target"]))
    return min(
        valid_actions,
        key=lambda action: (
            task.world.distance_to_goal[action["arguments"]["target"]],
            neighbors[action["arguments"]["target"]],
            action["arguments"]["target"],
        ),
    )


def _scheduling_baseline_action(name: str, observation: dict[str, Any], valid_actions: list[dict[str, Any]], rng: Random) -> dict[str, Any]:
    if name == "random_ready":
        return rng.choice(valid_actions)
    ready = {job["job_id"]: job for job in observation["ready_jobs"]}
    if name == "shortest_processing_time":
        return min(valid_actions, key=lambda action: (ready[action["arguments"]["job_id"]]["duration"], ready[action["arguments"]["job_id"]]["deadline"], action["arguments"]["job_id"]))
    if name == "earliest_deadline":
        return min(valid_actions, key=lambda action: (ready[action["arguments"]["job_id"]]["deadline"], ready[action["arguments"]["job_id"]]["duration"], action["arguments"]["job_id"]))
    return min(
        valid_actions,
        key=lambda action: (
            ready[action["arguments"]["job_id"]]["deadline"] - observation["current_time"] - ready[action["arguments"]["job_id"]]["duration"],
            ready[action["arguments"]["job_id"]]["deadline"],
            action["arguments"]["job_id"],
        ),
    )


def _sokoban_baseline_action(name: str, valid_actions: list[dict[str, Any]], rng: Random) -> dict[str, Any]:
    if name == "random_valid":
        return rng.choice(valid_actions)
    if name == "push_when_possible":
        priority = lambda action: (
            0 if action["arguments"].get("causes_push") else 1,
            -action["arguments"].get("boxes_on_goals_after", 0),
            action["arguments"].get("goal_distance_after", 0),
            action["name"],
        )
        return min(valid_actions, key=priority)
    if name == "greedy_goal_progress":
        return min(
            valid_actions,
            key=lambda action: (
                -action["arguments"].get("boxes_on_goals_after", 0),
                action["arguments"].get("goal_distance_after", 0),
                0 if action["arguments"].get("causes_push") else 1,
                action["name"],
            ),
        )
    safe_actions = [action for action in valid_actions if not action["arguments"].get("introduces_deadlock")]
    candidates = safe_actions or valid_actions
    return min(
        candidates,
        key=lambda action: (
            action["arguments"].get("goal_distance_after", 0),
            0 if action["arguments"].get("causes_push") else 1,
            -action["arguments"].get("boxes_on_goals_after", 0),
            action["name"],
        ),
    )


def _deduction_action_sort_key(action: dict[str, Any]) -> tuple[Any, ...]:
    arguments = action.get("arguments", {})
    assignment = arguments.get("assignment")
    assignment_key = json.dumps(assignment, sort_keys=True) if isinstance(assignment, dict) else ""
    return (
        action.get("name", ""),
        str(arguments.get("category", "")),
        str(arguments.get("entity", "")),
        str(arguments.get("value", "")),
        assignment_key,
    )


def _deduction_baseline_action(name: str, valid_actions: list[dict[str, Any]], rng: Random) -> dict[str, Any]:
    if name == "random_valid":
        return rng.choice(valid_actions)
    if name == "propagate_first":
        priority = {"propagate": 0, "assert_pair": 1, "rule_out_pair": 2, "commit_solution": 3}
    elif name == "assert_first":
        priority = {"assert_pair": 0, "propagate": 1, "rule_out_pair": 2, "commit_solution": 3}
    else:
        priority = {"rule_out_pair": 0, "propagate": 1, "assert_pair": 2, "commit_solution": 3}
    return min(valid_actions, key=lambda action: (priority.get(action["name"], 4), _deduction_action_sort_key(action)))


def _symbolic_candidates(task: Any, env: RLVREnv) -> list[Any]:
    return _enumerate_forward_rewrites(env.state.current_expression, task.world.task_type)


def _symbolic_baseline_action(name: str, task: Any, env: RLVREnv, rng: Random) -> dict[str, Any]:
    candidates = _symbolic_candidates(task, env)
    if name == "random_valid":
        chosen = rng.choice(candidates)
    elif name == "greedy_target_distance":
        chosen = min(
            candidates,
            key=lambda candidate: (
                _structural_distance(candidate.result, task.world.target_expression),
                _count_nodes(candidate.result),
                candidate.rule_id,
                candidate.path,
            ),
        )
    elif name == "greedy_expression_size":
        chosen = min(
            candidates,
            key=lambda candidate: (
                _count_nodes(candidate.result),
                _structural_distance(candidate.result, task.world.target_expression),
                candidate.rule_id,
                candidate.path,
            ),
        )
    else:
        priority = {
            "remove_add_zero_left": 0,
            "remove_add_zero_right": 0,
            "remove_mul_one_left": 0,
            "remove_mul_one_right": 0,
            "collapse_mul_zero": 0,
            "fold_add_constants": 1,
            "fold_mul_constants": 1,
            "double_negation": 0,
            "de_morgan_and": 1,
            "de_morgan_or": 1,
            "sort_add_operands": 2,
            "sort_mul_operands": 2,
            "sort_and_operands": 2,
            "sort_or_operands": 2,
            "assoc_add": 3,
            "assoc_mul": 3,
            "assoc_and": 3,
            "assoc_or": 3,
        }
        chosen = min(
            candidates,
            key=lambda candidate: (
                priority.get(candidate.rule_id, 5),
                _structural_distance(candidate.result, task.world.target_expression),
                _count_nodes(candidate.result),
                candidate.rule_id,
                candidate.path,
            ),
        )
    return {
        "name": "rewrite",
        "arguments": {"rule_id": chosen.rule_id, "path": list(chosen.path), "path_str": "root" if not chosen.path else ".".join(str(part) for part in chosen.path)},
    }


def _evaluate_policy(family_name: str, task: Any, baseline_name: str) -> dict[str, Any]:
    rng = Random(stable_hash_seed("baseline", family_name, task.task_id, baseline_name))
    env = RLVREnv(task)
    observation, info = env.reset()
    if baseline_name == "oracle":
        planned_actions = list(task.oracle.solve().actions)
    else:
        planned_actions = None
    action_index = 0
    terminated = False
    truncated = False
    while True:
        valid_actions = info.get("valid_actions", [])
        if terminated or truncated or not valid_actions and planned_actions is None:
            break
        if planned_actions is not None:
            if action_index >= len(planned_actions):
                break
            action = planned_actions[action_index]
            action_index += 1
        elif family_name == "graph_planning":
            action = _graph_baseline_action(baseline_name, task, observation, valid_actions, rng)
        elif family_name == "sokoban":
            action = _sokoban_baseline_action(baseline_name, valid_actions, rng)
        elif family_name == "scheduling":
            action = _scheduling_baseline_action(baseline_name, observation, valid_actions, rng)
        elif family_name == "deduction_grid":
            action = _deduction_baseline_action(baseline_name, valid_actions, rng)
        else:
            action = _symbolic_baseline_action(baseline_name, task, env, rng)
        observation, _, terminated, truncated, info = env.step(action)
    trace = env.trace.to_dict() if env.trace else {"steps": []}
    final_step = trace["steps"][-1] if trace["steps"] else None
    verification = final_step["info"]["verification"] if final_step else {"feasibility_score": 0.0, "quality_score": 0.0, "passed": False}
    success = bool(final_step and final_step["terminated"] and verification["passed"])
    num_steps = len(trace["steps"])
    total_reward = sum(step["reward"] for step in trace["steps"])
    invalid_action_rate = 0.0 if num_steps == 0 else sum(1 for step in trace["steps"] if step["info"].get("invalid_action")) / num_steps
    objective_gap: float | None = None
    secondary_objective_gap: float | None = None
    if family_name == "graph_planning":
        final_cost = getattr(env.state, "total_cost", None)
        if final_cost is not None:
            objective_gap = float(final_cost - task.world.shortest_cost)
    elif family_name == "sokoban":
        final_moves = getattr(env.state, "move_count", None)
        final_pushes = getattr(env.state, "push_count", None)
        if final_moves is not None:
            objective_gap = float(final_moves - task.world.oracle_move_count)
        if final_pushes is not None:
            secondary_objective_gap = float(final_pushes - task.world.oracle_push_count)
    elif family_name == "scheduling":
        final_tardiness = getattr(env.state, "total_tardiness", None)
        if final_tardiness is not None:
            objective_gap = float(final_tardiness - task.world.optimal_total_tardiness)
    else:
        objective_gap = float(num_steps - len(task.world.oracle_plan))
    return {
        "success": 1.0 if success else 0.0,
        "feasibility_score": float(verification.get("feasibility_score", 0.0)),
        "quality_score": float(verification.get("quality_score", 0.0)),
        "invalid_action_rate": float(invalid_action_rate),
        "steps": float(num_steps),
        "total_reward": float(total_reward),
        "objective_gap": objective_gap,
        "secondary_objective_gap": secondary_objective_gap,
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _aggregate_baseline_results(records: list[dict[str, Any]]) -> dict[str, float]:
    gaps = [record["objective_gap"] for record in records if record["objective_gap"] is not None]
    secondary_gaps = [record["secondary_objective_gap"] for record in records if record["secondary_objective_gap"] is not None]
    return {
        "success_rate": _mean([record["success"] for record in records]),
        "mean_feasibility_score": _mean([record["feasibility_score"] for record in records]),
        "mean_quality_score": _mean([record["quality_score"] for record in records]),
        "mean_invalid_action_rate": _mean([record["invalid_action_rate"] for record in records]),
        "mean_steps": _mean([record["steps"] for record in records]),
        "mean_total_reward": _mean([record["total_reward"] for record in records]),
        "mean_objective_gap": _mean(gaps),
        "mean_secondary_objective_gap": _mean(secondary_gaps),
    }


def _manifest_record(seed: int, split_name: str, task: Any) -> dict[str, Any]:
    oracle_solution = task.oracle.solve() if task.oracle is not None else None
    return {
        "split": split_name,
        "seed": seed,
        "task_id": task.task_id,
        "generation_seed": task.generation_seed,
        "objective_name": task.objective.name,
        "generation_params": as_primitive(task.generation_params),
        "metadata": as_primitive(task.metadata),
        "oracle_objective_value": oracle_solution.objective_value if oracle_solution is not None else None,
        "oracle_steps": len(oracle_solution.actions) if oracle_solution is not None else None,
        "oracle_difficulty_estimate": oracle_solution.difficulty_estimate if oracle_solution is not None else None,
    }


def _family_summary_row(family_name: str, split_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"split": split_name, "count": len(records)}
    if family_name == "graph_planning":
        row.update(
            {
                "num_nodes_mean": round(_mean([record["metadata"]["num_nodes"] for record in records]), 2),
                "num_edges_mean": round(_mean([record["metadata"]["num_edges"] for record in records]), 2),
                "optimal_cost_mean": round(_mean([record["metadata"]["optimal_cost"] for record in records]), 2),
                "oracle_steps_mean": round(_mean([record["oracle_steps"] for record in records]), 2),
            }
        )
    elif family_name == "scheduling":
        row.update(
            {
                "num_jobs_mean": round(_mean([record["metadata"]["num_jobs"] for record in records]), 2),
                "num_constraints_mean": round(_mean([record["metadata"]["num_constraints"] for record in records]), 2),
                "optimal_tardiness_mean": round(_mean([record["metadata"]["optimal_total_tardiness"] for record in records]), 2),
                "oracle_steps_mean": round(_mean([record["oracle_steps"] for record in records]), 2),
            }
        )
    elif family_name == "sokoban":
        row.update(
            {
                "num_boxes_mean": round(_mean([record["metadata"]["num_boxes"] for record in records]), 2),
                "board_height_mean": round(_mean([record["metadata"]["board_height"] for record in records]), 2),
                "board_width_mean": round(_mean([record["metadata"]["board_width"] for record in records]), 2),
                "oracle_steps_mean": round(_mean([record["metadata"]["oracle_steps"] for record in records]), 2),
                "oracle_push_count_mean": round(_mean([record["metadata"]["oracle_push_count"] for record in records]), 2),
                "boxes_on_goals_at_start_mean": round(_mean([record["metadata"]["boxes_on_goals_at_start"] for record in records]), 2),
                "unsolved_boxes_at_start_mean": round(_mean([record["metadata"]["unsolved_boxes_at_start"] for record in records]), 2),
                "box_interaction_pair_count_mean": round(_mean([record["metadata"]["box_interaction_pair_count"] for record in records]), 2),
                "box_interaction_component_count_mean": round(_mean([record["metadata"]["box_interaction_component_count"] for record in records]), 2),
            }
        )
    elif family_name == "deduction_grid":
        row.update(
            {
                "num_entities_mean": round(_mean([record["metadata"]["num_entities"] for record in records]), 2),
                "num_relation_categories_mean": round(_mean([record["metadata"]["num_relation_categories"] for record in records]), 2),
                "num_clues_mean": round(_mean([record["metadata"]["num_clues"] for record in records]), 2),
                "oracle_steps_mean": round(_mean([record["oracle_steps"] for record in records]), 2),
            }
        )
    else:
        task_types: dict[str, int] = {}
        for record in records:
            task_types[record["metadata"]["task_type"]] = task_types.get(record["metadata"]["task_type"], 0) + 1
        row.update(
            {
                "source_nodes_mean": round(_mean([record["metadata"]["source_num_nodes"] for record in records]), 2),
                "target_nodes_mean": round(_mean([record["metadata"]["target_num_nodes"] for record in records]), 2),
                "oracle_steps_mean": round(_mean([record["metadata"]["oracle_steps"] for record in records]), 2),
                "task_type_histogram": task_types,
            }
        )
    return row


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(json.dumps(value, sort_keys=True) if isinstance(value, dict) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(as_primitive(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def build_family_artifacts(family_name: str, spec: dict[str, Any], output_root: Path) -> None:
    family = get_family(family_name)
    version = spec["version"]
    split_manifest: dict[str, list[dict[str, Any]]] = {}
    tasks_by_split: dict[str, list[tuple[int, Any]]] = {}

    for split_name, split_spec in spec["splits"].items():
        config = _config_from_split(split_spec)
        tasks: list[tuple[int, Any]] = []
        records: list[dict[str, Any]] = []
        for index in range(split_spec["count"]):
            seed = _seed_for(family_name, version, split_name, index)
            task = family.sample_instance(seed=seed, config=config)
            tasks.append((seed, task))
            records.append(_manifest_record(seed, split_name, task))
        tasks_by_split[split_name] = tasks
        split_manifest[split_name] = records

    summary_rows = [
        _family_summary_row(family_name, split_name, records)
        for split_name, records in split_manifest.items()
    ]

    baseline_rows: list[dict[str, Any]] = []
    for split_name in spec["evaluation_splits"]:
        split_tasks = tasks_by_split[split_name]
        for baseline_name in spec["baselines"]:
            results = [_evaluate_policy(family_name, task, baseline_name) for _, task in split_tasks]
            aggregate = _aggregate_baseline_results(results)
            baseline_rows.append({"split": split_name, "baseline": baseline_name, **aggregate})

    family_dir = output_root / family_name
    _write_json(
        family_dir / "benchmark_spec.json",
        {"family_name": family_name, **spec},
    )
    _write_json(
        family_dir / "split_manifest.json",
        {"family_name": family_name, "version": version, "splits": split_manifest},
    )

    summary_headers = list(summary_rows[0].keys()) if summary_rows else ["split", "count"]
    summary_text = "\n".join(
        [
            f"# {family_name} Diversity Summary",
            "",
            f"Benchmark version: `{version}`",
            "",
            _markdown_table(summary_rows, summary_headers),
        ]
    )
    _write_text(family_dir / "diversity_summary.md", summary_text)

    baseline_headers = [
        "split",
        "baseline",
        "success_rate",
        "mean_feasibility_score",
        "mean_quality_score",
        "mean_invalid_action_rate",
        "mean_steps",
        "mean_total_reward",
        "mean_objective_gap",
        "mean_secondary_objective_gap",
    ]
    baseline_text = "\n".join(
        [
            f"# {family_name} Baseline Results",
            "",
            f"Benchmark version: `{version}`",
            "",
            f"Objective gap column corresponds to `{spec['objective_gap_name']}`.",
            (
                f"Secondary objective gap column corresponds to `{spec['secondary_objective_gap_name']}`."
                if "secondary_objective_gap_name" in spec
                else "Secondary objective gap column is unused for this family."
            ),
            "",
            _markdown_table(baseline_rows, baseline_headers),
        ]
    )
    _write_text(family_dir / "baseline_results.md", baseline_text)


def build_index(output_root: Path) -> None:
    lines = [
        "# Benchmark Artifacts",
        "",
        "This directory contains frozen benchmark specs, split manifests, diversity summaries, and baseline tables for each built-in family.",
        "",
        "## Families",
        "",
        "- [deduction_grid](deduction_grid/benchmark_spec.json)",
        "- [graph_planning](graph_planning/benchmark_spec.json)",
        "- [scheduling](scheduling/benchmark_spec.json)",
        "- [sokoban](sokoban/benchmark_spec.json)",
        "- [symbolic_transformation](symbolic_transformation/benchmark_spec.json)",
    ]
    _write_text(output_root / "README.md", "\n".join(lines))


def main() -> None:
    output_root = ROOT / "docs" / "benchmark_artifacts"
    for family_name, spec in BENCHMARK_SPECS.items():
        build_family_artifacts(family_name, spec, output_root)
    build_index(output_root)


if __name__ == "__main__":
    main()
