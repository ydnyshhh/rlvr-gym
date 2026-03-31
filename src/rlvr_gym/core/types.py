from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RewardMode(str, Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    SHAPED = "shaped"


@dataclass(frozen=True)
class RewardConfig:
    mode: RewardMode = RewardMode.SHAPED
    success_reward: float = 1.0
    failure_penalty: float = -0.5
    invalid_action_penalty: float = -1.0
    step_penalty: float = -0.01
    dense_verifier_weight: float = 0.2
    shaped_hint_weight: float = 1.0


@dataclass(frozen=True)
class FamilyConfig:
    difficulty: str = "medium"
    observability: str = "full"
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    attach_oracle: bool = True
    max_steps: int | None = None
    generation_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalAction:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class TaskObjective:
    name: str
    description: str
    success_criteria: dict[str, Any]
    optimization_target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "success_criteria": dict(self.success_criteria),
            "optimization_target": self.optimization_target,
        }


@dataclass(frozen=True)
class TransitionResult:
    next_state: Any
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    invalid_action: bool = False
    reward_hints: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransitionRecord:
    step_index: int
    action: dict[str, Any]
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class EpisodeTrace:
    task_id: str
    initial_observation: Any
    steps: list[TransitionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "initial_observation": self.initial_observation,
            "steps": [
                {
                    "step_index": step.step_index,
                    "action": step.action,
                    "observation": step.observation,
                    "reward": step.reward,
                    "terminated": step.terminated,
                    "truncated": step.truncated,
                    "info": step.info,
                }
                for step in self.steps
            ],
        }


@dataclass
class TaskInstance:
    family: Any
    family_name: str
    task_id: str
    generation_seed: int
    config: FamilyConfig
    generation_params: dict[str, Any]
    world: Any
    objective: TaskObjective
    initial_state: Any
    initial_observation: Any
    verifier_suite: Any
    reward_engine: Any
    oracle: Any | None
    max_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)
