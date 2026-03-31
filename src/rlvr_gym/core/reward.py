from __future__ import annotations

from rlvr_gym.core.types import RewardConfig, RewardMode, TransitionResult
from rlvr_gym.core.verifier import VerificationReport


class RewardEngine:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute_step(self, transition: TransitionResult, report: VerificationReport) -> float:
        reward = self.config.step_penalty
        if transition.invalid_action:
            reward += self.config.invalid_action_penalty
        if self.config.mode in {RewardMode.DENSE, RewardMode.SHAPED}:
            reward += self.config.dense_verifier_weight * report.normalized_score
        if self.config.mode is RewardMode.SHAPED:
            reward += self.config.shaped_hint_weight * sum(transition.reward_hints.values())
        if transition.terminated or transition.truncated:
            reward += self.config.success_reward if transition.success else self.config.failure_penalty
        return reward
