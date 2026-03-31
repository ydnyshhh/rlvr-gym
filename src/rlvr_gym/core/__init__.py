"""Core abstractions for RLVR-Gym."""

from rlvr_gym.core.family import EnvironmentFamily
from rlvr_gym.core.oracle import Oracle, OracleSolution
from rlvr_gym.core.reward import RewardEngine
from rlvr_gym.core.runtime import RLVREnv
from rlvr_gym.core.types import CanonicalAction, FamilyConfig, RewardConfig, RewardMode
from rlvr_gym.core.verifier import VerificationReport, VerificationResult, VerificationScope, VerifierSuite

__all__ = [
    "CanonicalAction",
    "EnvironmentFamily",
    "FamilyConfig",
    "Oracle",
    "OracleSolution",
    "RLVREnv",
    "RewardConfig",
    "RewardEngine",
    "RewardMode",
    "VerificationReport",
    "VerificationResult",
    "VerificationScope",
    "VerifierSuite",
]
