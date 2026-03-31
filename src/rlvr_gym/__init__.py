"""RLVR-Gym: procedural verifiable reasoning and decision environments."""

from rlvr_gym.core.exporters import (
    build_benchmark_splits,
    export_offline_transitions,
    export_oracle_views,
    export_sft_example,
    export_task_spec,
    rollout_oracle,
    write_benchmark_splits,
)
from rlvr_gym.core.family import EnvironmentFamily
from rlvr_gym.core.oracle import ProofCertificate
from rlvr_gym.core.runtime import RLVREnv
from rlvr_gym.core.types import CanonicalAction, FamilyConfig, RewardConfig, RewardMode, TaskSpace
from rlvr_gym.registry import FAMILY_REGISTRY, get_family, list_families

__all__ = [
    "CanonicalAction",
    "EnvironmentFamily",
    "FamilyConfig",
    "FAMILY_REGISTRY",
    "ProofCertificate",
    "RLVREnv",
    "RewardConfig",
    "RewardMode",
    "TaskSpace",
    "build_benchmark_splits",
    "export_offline_transitions",
    "export_oracle_views",
    "export_sft_example",
    "export_task_spec",
    "get_family",
    "list_families",
    "rollout_oracle",
    "write_benchmark_splits",
]
