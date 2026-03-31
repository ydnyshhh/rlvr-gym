"""Core abstractions for RLVR-Gym."""

from rlvr_gym.core.family import EnvironmentFamily
from rlvr_gym.core.oracle import Oracle, OracleSolution, ProofCertificate
from rlvr_gym.core.reward import RewardEngine
from rlvr_gym.core.runtime import RLVREnv
from rlvr_gym.core.types import CanonicalAction, FamilyConfig, RewardConfig, RewardMode, TaskSpace
from rlvr_gym.core.verifier import (
    VerificationKind,
    VerificationReport,
    VerificationResult,
    VerificationScope,
    VerifierSuite,
)

__all__ = [
    "CanonicalAction",
    "EnvironmentFamily",
    "FamilyConfig",
    "Oracle",
    "OracleSolution",
    "ProofCertificate",
    "RLVREnv",
    "RewardConfig",
    "RewardEngine",
    "RewardMode",
    "TaskSpace",
    "VerificationKind",
    "VerificationReport",
    "VerificationResult",
    "VerificationScope",
    "VerifierSuite",
]
