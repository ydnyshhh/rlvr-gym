from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from rlvr_gym.core.reward import RewardEngine
from rlvr_gym.core.types import CanonicalAction, FamilyConfig, TaskInstance, TaskObjective
from rlvr_gym.core.utils import as_primitive, merge_overrides, stable_hash_seed
from rlvr_gym.core.verifier import VerifierSuite


class EnvironmentFamily(ABC):
    name: str = "base"
    description: str = "Abstract environment family."

    @abstractmethod
    def sample_generation_params(self, config: FamilyConfig, rng: random.Random) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def sample_world(self, generation_params: dict[str, Any], rng: random.Random) -> Any:
        raise NotImplementedError

    @abstractmethod
    def derive_objective(
        self,
        world: Any,
        generation_params: dict[str, Any],
        rng: random.Random,
    ) -> TaskObjective:
        raise NotImplementedError

    @abstractmethod
    def initial_state(self, world: Any, objective: TaskObjective, generation_params: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def observe(self, world: Any, state: Any, objective: TaskObjective, generation_params: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def valid_actions(
        self,
        world: Any,
        state: Any,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> list[CanonicalAction]:
        raise NotImplementedError

    @abstractmethod
    def transition(
        self,
        world: Any,
        state: Any,
        action: CanonicalAction,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_verifier_suite(
        self,
        world: Any,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> VerifierSuite:
        raise NotImplementedError

    def build_oracle(self, world: Any, objective: TaskObjective, generation_params: dict[str, Any]) -> Any | None:
        return None

    def export_world(self, world: Any) -> dict[str, Any]:
        return as_primitive(world)

    def export_state(self, state: Any) -> dict[str, Any]:
        return as_primitive(state)

    def export_task_spec(self, task: TaskInstance) -> dict[str, Any]:
        return {
            "family_name": task.family_name,
            "task_id": task.task_id,
            "generation_seed": task.generation_seed,
            "generation_params": as_primitive(task.generation_params),
            "objective": task.objective.to_dict(),
            "world": self.export_world(task.world),
            "initial_state": self.export_state(task.initial_state),
            "initial_observation": as_primitive(task.initial_observation),
            "metadata": as_primitive(task.metadata),
        }

    def recommended_max_steps(self, generation_params: dict[str, Any]) -> int:
        return 32

    def task_metadata(self, world: Any, objective: TaskObjective, generation_params: dict[str, Any]) -> dict[str, Any]:
        return {}

    def coerce_action(self, action: CanonicalAction | Mapping[str, Any] | str) -> CanonicalAction:
        if isinstance(action, CanonicalAction):
            return action
        if isinstance(action, str):
            return CanonicalAction(name=action, arguments={})
        if isinstance(action, Mapping):
            name = action.get("name", action.get("type"))
            if not name:
                raise ValueError("Action mappings must include a 'name' or 'type' field.")
            arguments = dict(action.get("arguments", {}))
            for key, value in action.items():
                if key not in {"name", "type", "arguments"}:
                    arguments[key] = value
            return CanonicalAction(name=str(name), arguments=arguments)
        raise TypeError(f"Unsupported action format: {type(action)!r}")

    def sample_instance(self, seed: int, config: FamilyConfig | None = None) -> TaskInstance:
        resolved_config = config or FamilyConfig()
        generation_seed = stable_hash_seed(self.name, seed, resolved_config.difficulty)
        rng = random.Random(generation_seed)
        generation_params = merge_overrides(
            self.sample_generation_params(resolved_config, rng),
            resolved_config.generation_overrides,
        )
        world = self.sample_world(generation_params, rng)
        objective = self.derive_objective(world, generation_params, rng)
        initial_state = self.initial_state(world, objective, generation_params)
        initial_observation = self.observe(world, initial_state, objective, generation_params)
        verifier_suite = self.build_verifier_suite(world, objective, generation_params)
        reward_engine = RewardEngine(resolved_config.reward_config)
        oracle = self.build_oracle(world, objective, generation_params) if resolved_config.attach_oracle else None
        task_id = f"{self.name}-{stable_hash_seed(seed, generation_params, objective.description):016x}"
        max_steps = resolved_config.max_steps or self.recommended_max_steps(generation_params)
        metadata = {
            "difficulty": resolved_config.difficulty,
            "observability": resolved_config.observability,
            **self.task_metadata(world, objective, generation_params),
        }
        return TaskInstance(
            family=self,
            family_name=self.name,
            task_id=task_id,
            generation_seed=generation_seed,
            config=resolved_config,
            generation_params=as_primitive(generation_params),
            world=world,
            objective=objective,
            initial_state=initial_state,
            initial_observation=initial_observation,
            verifier_suite=verifier_suite,
            reward_engine=reward_engine,
            oracle=oracle,
            max_steps=max_steps,
            metadata=metadata,
        )

    def create_env(self, seed: int, config: FamilyConfig | None = None) -> Any:
        from rlvr_gym.core.runtime import RLVREnv

        return RLVREnv(self.sample_instance(seed=seed, config=config))
