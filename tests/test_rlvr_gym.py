from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rlvr_gym.core.exporters import build_benchmark_splits, export_offline_transitions, export_task_spec, rollout_oracle
from rlvr_gym.core.reward import RewardEngine
from rlvr_gym.core.runtime import RLVREnv
from rlvr_gym.core.types import FamilyConfig, RewardConfig, RewardMode, TransitionResult
from rlvr_gym.core.verifier import VerificationKind, VerificationReport, VerificationResult, VerificationScope
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
        self.assertTrue(final_step["info"]["verification"]["passed"])
        self.assertFalse(final_step["info"]["verification"]["hard_failed"])
        self.assertEqual(final_step["info"]["verification"]["kind_scores"]["quality"], 1.0)
        verification_results = {result["name"]: result for result in final_step["info"]["verification"]["results"]}
        self.assertTrue(verification_results["graph_goal_reached"]["passed"])
        self.assertTrue(verification_results["graph_trajectory_optimality"]["passed"])
        self.assertEqual(verification_results["graph_trajectory_optimality"]["kind"], "quality")

    def test_scheduling_oracle_exports_offline_transitions(self) -> None:
        family = get_family("scheduling")
        task = family.sample_instance(seed=11, config=FamilyConfig(difficulty="medium"))
        dataset = export_offline_transitions(task)
        self.assertEqual(len(dataset["transitions"]), len(task.world.jobs))
        self.assertTrue(dataset["transitions"][-1]["terminated"])
        self.assertEqual(task.oracle.solve().objective_value, task.world.optimal_total_tardiness)
        self.assertTrue(dataset["oracle_solution"]["optimal"])
        self.assertTrue(dataset["oracle_solution"]["certificate"]["optimal"])

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

    def test_verification_report_separates_feasibility_and_quality(self) -> None:
        report = VerificationReport(
            results=(
                VerificationResult(
                    name="legal",
                    scope=VerificationScope.ACTION,
                    passed=True,
                    score=1.0,
                    kind=VerificationKind.FEASIBILITY,
                    hard=True,
                    weight=2.0,
                ),
                VerificationResult(
                    name="suboptimal",
                    scope=VerificationScope.TRAJECTORY,
                    passed=False,
                    score=0.25,
                    kind=VerificationKind.QUALITY,
                    hard=False,
                    weight=1.0,
                ),
            )
        )
        self.assertTrue(report.passed)
        self.assertFalse(report.hard_failed)
        self.assertEqual(report.feasibility_score, 1.0)
        self.assertEqual(report.quality_score, 0.25)

    def test_hybrid_reward_mode_uses_feasibility_quality_and_hints(self) -> None:
        engine = RewardEngine(
            RewardConfig(
                mode=RewardMode.HYBRID,
                step_penalty=0.0,
                success_reward=0.0,
                failure_penalty=0.0,
                feasibility_weight=1.0,
                quality_weight=1.0,
                shaped_hint_weight=1.0,
                terminal_quality_weight=0.0,
                hard_failure_penalty=0.0,
            )
        )
        report = VerificationReport(
            results=(
                VerificationResult(
                    name="feasible",
                    scope=VerificationScope.STATE,
                    passed=True,
                    score=1.0,
                    kind=VerificationKind.FEASIBILITY,
                ),
                VerificationResult(
                    name="quality",
                    scope=VerificationScope.TRAJECTORY,
                    passed=True,
                    score=0.5,
                    kind=VerificationKind.QUALITY,
                ),
            )
        )
        reward = engine.compute_step(
            TransitionResult(next_state={"ok": True}, reward_hints={"proxy": 0.25}),
            report,
        )
        self.assertAlmostEqual(reward, 1.75)


if __name__ == "__main__":
    unittest.main()
