from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlvr_gym.core.exporters import build_benchmark_splits, export_offline_transitions, export_task_spec, rollout_oracle
from rlvr_gym.core.runtime import RLVREnv
from rlvr_gym.core.types import FamilyConfig
from rlvr_gym.registry import get_family


class RLVRGymTests(unittest.TestCase):
    def test_graph_task_generation_is_deterministic(self) -> None:
        family = get_family("graph_planning")
        config = FamilyConfig(difficulty="medium")
        task_a = family.sample_instance(seed=123, config=config)
        task_b = family.sample_instance(seed=123, config=config)
        self.assertEqual(export_task_spec(task_a), export_task_spec(task_b))

    def test_graph_oracle_rollout_reaches_verified_optimum(self) -> None:
        family = get_family("graph_planning")
        task = family.sample_instance(seed=7, config=FamilyConfig(difficulty="medium"))
        rollout = rollout_oracle(task)
        self.assertTrue(rollout["completed"])
        final_step = rollout["trace"]["steps"][-1]
        self.assertTrue(final_step["terminated"])
        verification_results = {result["name"]: result for result in final_step["info"]["verification"]["results"]}
        self.assertTrue(verification_results["graph_goal_reached"]["passed"])
        self.assertTrue(verification_results["graph_trajectory_optimality"]["passed"])

    def test_scheduling_oracle_exports_offline_transitions(self) -> None:
        family = get_family("scheduling")
        task = family.sample_instance(seed=11, config=FamilyConfig(difficulty="medium"))
        dataset = export_offline_transitions(task)
        self.assertEqual(len(dataset["transitions"]), len(task.world.jobs))
        self.assertTrue(dataset["transitions"][-1]["terminated"])
        self.assertEqual(task.oracle.solve().objective_value, task.world.optimal_total_tardiness)

    def test_benchmark_split_generation_is_reproducible(self) -> None:
        family = get_family("scheduling")
        config = FamilyConfig(difficulty="easy")
        split_counts = {"train": 3, "validation": 1}
        splits_a = build_benchmark_splits(family=family, split_counts=split_counts, base_seed=55, config=config)
        splits_b = build_benchmark_splits(family=family, split_counts=split_counts, base_seed=55, config=config)
        self.assertEqual(splits_a, splits_b)

    def test_runtime_respects_max_steps_truncation(self) -> None:
        family = get_family("scheduling")
        task = family.sample_instance(seed=101, config=FamilyConfig(difficulty="easy", max_steps=1))
        env = RLVREnv(task)
        _, info = env.reset()
        action = info["valid_actions"][0]
        _, _, terminated, truncated, step_info = env.step(action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(step_info["termination_reason"], "max_steps")


if __name__ == "__main__":
    unittest.main()
