from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class VerificationScope(str, Enum):
    ACTION = "action"
    STATE = "state"
    GOAL = "goal"
    TRAJECTORY = "trajectory"


@dataclass(frozen=True)
class VerificationResult:
    name: str
    scope: VerificationScope
    passed: bool
    score: float
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope.value,
            "passed": self.passed,
            "score": self.score,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[VerificationResult, ...] = ()

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results) if self.results else True

    @property
    def normalized_score(self) -> float:
        if not self.results:
            return 1.0
        total = sum(max(0.0, min(1.0, result.score)) for result in self.results)
        return total / len(self.results)

    def extend(self, other: VerificationReport) -> VerificationReport:
        return VerificationReport(results=self.results + other.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "normalized_score": self.normalized_score,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class StepContext:
    world: Any
    objective: Any
    generation_params: dict[str, Any]
    previous_state: Any
    action: Any
    transition: Any
    next_state: Any
    trace: Any


@dataclass(frozen=True)
class TrajectoryContext:
    world: Any
    objective: Any
    generation_params: dict[str, Any]
    final_state: Any
    trace: Any
    success: bool
    truncated: bool


class BaseVerifier:
    name: str = "base"

    def evaluate_step(self, context: StepContext) -> Sequence[VerificationResult]:
        return ()

    def evaluate_trajectory(self, context: TrajectoryContext) -> Sequence[VerificationResult]:
        return ()


class VerifierSuite:
    def __init__(self, verifiers: Iterable[BaseVerifier] | None = None) -> None:
        self.verifiers = list(verifiers or [])

    def evaluate_step(self, context: StepContext) -> VerificationReport:
        results: list[VerificationResult] = []
        for verifier in self.verifiers:
            results.extend(verifier.evaluate_step(context))
        return VerificationReport(results=tuple(results))

    def evaluate_trajectory(self, context: TrajectoryContext) -> VerificationReport:
        results: list[VerificationResult] = []
        for verifier in self.verifiers:
            results.extend(verifier.evaluate_trajectory(context))
        return VerificationReport(results=tuple(results))
