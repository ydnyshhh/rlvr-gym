from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence


class VerificationScope(str, Enum):
    ACTION = "action"
    STATE = "state"
    GOAL = "goal"
    TRAJECTORY = "trajectory"


class VerificationKind(str, Enum):
    FEASIBILITY = "feasibility"
    QUALITY = "quality"


@dataclass(frozen=True)
class VerificationResult:
    name: str
    scope: VerificationScope
    passed: bool
    score: float
    kind: VerificationKind = VerificationKind.FEASIBILITY
    weight: float = 1.0
    hard: bool = False
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope.value,
            "passed": self.passed,
            "score": self.score,
            "kind": self.kind.value,
            "weight": self.weight,
            "hard": self.hard,
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[VerificationResult, ...] = ()

    @property
    def hard_failed(self) -> bool:
        return any(result.hard and not result.passed for result in self.results)

    @property
    def feasibility_passed(self) -> bool:
        feasibility_results = [result for result in self.results if result.kind is VerificationKind.FEASIBILITY]
        return all(result.passed for result in feasibility_results) if feasibility_results else True

    @property
    def passed(self) -> bool:
        return self.feasibility_passed and not self.hard_failed

    @staticmethod
    def _weighted_average(results: Sequence[VerificationResult]) -> float:
        if not results:
            return 1.0
        total_weight = sum(max(0.0, result.weight) for result in results)
        if total_weight <= 0.0:
            total_weight = float(len(results))
            return sum(max(0.0, min(1.0, result.score)) for result in results) / total_weight
        return sum(
            max(0.0, min(1.0, result.score)) * max(0.0, result.weight)
            for result in results
        ) / total_weight

    @property
    def weighted_score(self) -> float:
        return self._weighted_average(self.results)

    @property
    def normalized_score(self) -> float:
        return self.weighted_score

    @property
    def feasibility_score(self) -> float:
        return self._weighted_average(
            [result for result in self.results if result.kind is VerificationKind.FEASIBILITY]
        )

    @property
    def quality_score(self) -> float:
        return self._weighted_average(
            [result for result in self.results if result.kind is VerificationKind.QUALITY]
        )

    @property
    def scope_scores(self) -> dict[str, float]:
        return {
            scope.value: self._weighted_average([result for result in self.results if result.scope is scope])
            for scope in VerificationScope
        }

    @property
    def kind_scores(self) -> dict[str, float]:
        return {
            kind.value: self._weighted_average([result for result in self.results if result.kind is kind])
            for kind in VerificationKind
        }

    def extend(self, other: VerificationReport) -> VerificationReport:
        return VerificationReport(results=self.results + other.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "hard_failed": self.hard_failed,
            "normalized_score": self.normalized_score,
            "weighted_score": self.weighted_score,
            "feasibility_score": self.feasibility_score,
            "quality_score": self.quality_score,
            "scope_scores": self.scope_scores,
            "kind_scores": self.kind_scores,
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
