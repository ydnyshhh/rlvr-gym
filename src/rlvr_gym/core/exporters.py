from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rlvr_gym.core.runtime import RLVREnv
from rlvr_gym.core.types import TaskInstance
from rlvr_gym.core.utils import as_primitive, stable_hash_seed


def export_task_spec(task: TaskInstance) -> dict[str, Any]:
    return task.family.export_task_spec(task)


def rollout_oracle(task: TaskInstance) -> dict[str, Any]:
    if task.oracle is None:
        raise ValueError("Task instance has no oracle attached.")
    solution = task.oracle.solve()
    env = RLVREnv(task)
    env.reset()
    completed = False
    for action in solution.actions:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            completed = True
            break
    return {
        "task_spec": export_task_spec(task),
        "oracle_solution": as_primitive(solution.to_dict()),
        "completed": completed,
        "trace": env.trace.to_dict() if env.trace else None,
    }


def export_sft_example(task: TaskInstance) -> dict[str, Any]:
    if task.oracle is None:
        raise ValueError("SFT export requires an oracle.")
    solution = task.oracle.solve()
    task_spec = export_task_spec(task)
    prompt = (
        "Solve the following formal decision problem. "
        "Return a JSON object with an 'actions' field containing the canonical action sequence.\n\n"
        f"{json.dumps(task_spec, indent=2, sort_keys=True)}"
    )
    assistant_payload = {
        "actions": list(solution.actions),
        "objective_value": solution.objective_value,
        "metadata": solution.metadata,
    }
    return {
        "family_name": task.family_name,
        "task_id": task.task_id,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(as_primitive(assistant_payload), sort_keys=True)},
        ],
        "task_spec": task_spec,
    }


def export_offline_transitions(task: TaskInstance) -> dict[str, Any]:
    rollout = rollout_oracle(task)
    trace = rollout["trace"] or {}
    transitions: list[dict[str, Any]] = []
    previous_observation = trace.get("initial_observation")
    for step in trace.get("steps", []):
        transitions.append(
            {
                "observation": previous_observation,
                "action": step["action"],
                "reward": step["reward"],
                "next_observation": step["observation"],
                "terminated": step["terminated"],
                "truncated": step["truncated"],
                "info": step["info"],
            }
        )
        previous_observation = step["observation"]
    return {
        "family_name": task.family_name,
        "task_id": task.task_id,
        "task_spec": rollout["task_spec"],
        "oracle_solution": rollout["oracle_solution"],
        "transitions": transitions,
    }


def build_benchmark_splits(
    family: Any,
    split_counts: dict[str, int],
    base_seed: int,
    config: Any | None = None,
    include_oracle: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name, count in split_counts.items():
        records: list[dict[str, Any]] = []
        for index in range(count):
            seed = stable_hash_seed(family.name, split_name, base_seed, index)
            task = family.sample_instance(seed=seed, config=config)
            record = export_task_spec(task)
            if include_oracle and task.oracle is not None:
                record["oracle_solution"] = as_primitive(task.oracle.solve().to_dict())
            record["benchmark_split"] = split_name
            record["benchmark_index"] = index
            records.append(record)
        splits[split_name] = records
    return splits


def write_benchmark_splits(
    output_dir: str | Path,
    splits: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for split_name, records in splits.items():
        split_path = target / f"{split_name}.jsonl"
        with split_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(as_primitive(record), sort_keys=True))
                handle.write("\n")
        written[split_name] = str(split_path)
    return written
