from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from rlvr_gym.core.types import EpisodeTrace, TaskInstance, TransitionRecord
from rlvr_gym.core.utils import as_primitive
from rlvr_gym.core.verifier import StepContext, TrajectoryContext


class RLVREnv:
    """Gym-style runtime for sampled RLVR task instances."""

    def __init__(self, task: TaskInstance) -> None:
        self.task = task
        self.trace: EpisodeTrace | None = None
        self._state: Any | None = None
        self._done = False
        self._started = False

    @property
    def state(self) -> Any:
        if not self._started:
            raise RuntimeError("Environment has not been reset yet.")
        return self._state

    def reset(self) -> tuple[Any, dict[str, Any]]:
        self._state = copy.deepcopy(self.task.initial_state)
        self.trace = EpisodeTrace(
            task_id=self.task.task_id,
            initial_observation=copy.deepcopy(self.task.initial_observation),
        )
        self._done = False
        self._started = True
        info = {
            "task_id": self.task.task_id,
            "family_name": self.task.family_name,
            "objective": self.task.objective.to_dict(),
            "valid_actions": [action.to_dict() for action in self.task.family.valid_actions(
                self.task.world,
                self._state,
                self.task.objective,
                self.task.generation_params,
            )],
        }
        return copy.deepcopy(self.task.initial_observation), info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Cannot step a finished environment.")

        assert self.trace is not None
        previous_state = self._state
        canonical_action = self.task.family.coerce_action(action)
        transition = self.task.family.transition(
            self.task.world,
            previous_state,
            canonical_action,
            self.task.objective,
            self.task.generation_params,
        )
        step_index = len(self.trace.steps)
        truncated = transition.truncated or (step_index + 1 >= self.task.max_steps and not transition.terminated)
        if truncated and not transition.truncated:
            transition = replace(
                transition,
                truncated=True,
                info={**transition.info, "termination_reason": "max_steps"},
            )

        observation = self.task.family.observe(
            self.task.world,
            transition.next_state,
            self.task.objective,
            self.task.generation_params,
        )
        step_report = self.task.verifier_suite.evaluate_step(
            StepContext(
                world=self.task.world,
                objective=self.task.objective,
                generation_params=self.task.generation_params,
                previous_state=previous_state,
                action=canonical_action,
                transition=transition,
                next_state=transition.next_state,
                trace=self.trace,
            )
        )
        combined_report = step_report
        if transition.terminated or transition.truncated:
            combined_report = combined_report.extend(
                self.task.verifier_suite.evaluate_trajectory(
                    TrajectoryContext(
                        world=self.task.world,
                        objective=self.task.objective,
                        generation_params=self.task.generation_params,
                        final_state=transition.next_state,
                        trace=self.trace,
                        success=transition.success,
                        truncated=transition.truncated,
                    )
                )
            )
        reward = self.task.reward_engine.compute_step(transition, combined_report)
        record = TransitionRecord(
            step_index=step_index,
            action=canonical_action.to_dict(),
            observation=as_primitive(observation),
            reward=reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            info={
                **as_primitive(transition.info),
                "verification": combined_report.to_dict(),
            },
        )
        self.trace.steps.append(record)
        self._state = transition.next_state
        self._done = transition.terminated or transition.truncated
        info = {
            "task_id": self.task.task_id,
            "verification": combined_report.to_dict(),
            "valid_actions": [
                action.to_dict()
                for action in self.task.family.valid_actions(
                    self.task.world,
                    self._state,
                    self.task.objective,
                    self.task.generation_params,
                )
            ],
            **as_primitive(transition.info),
        }
        return observation, reward, transition.terminated, transition.truncated, info
