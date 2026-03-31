from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProofCertificate:
    feasible: bool | None = None
    optimal: bool | None = None
    summary: str = ""
    witness: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "optimal": self.optimal,
            "summary": self.summary,
            "witness": dict(self.witness),
        }


@dataclass(frozen=True)
class OracleSolution:
    actions: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    objective_value: float | int | None = None
    feasible: bool | None = None
    optimal: bool | None = None
    difficulty_estimate: float | None = None
    certificate: ProofCertificate = field(default_factory=ProofCertificate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "metadata": dict(self.metadata),
            "objective_value": self.objective_value,
            "feasible": self.feasible,
            "optimal": self.optimal,
            "difficulty_estimate": self.difficulty_estimate,
            "certificate": self.certificate.to_dict(),
        }


class Oracle(ABC):
    def is_feasible(self) -> bool | None:
        return None

    def estimate_difficulty(self) -> float | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "oracle_type": self.__class__.__name__,
            "feasible": self.is_feasible(),
            "difficulty_estimate": self.estimate_difficulty(),
        }

    @abstractmethod
    def solve(self) -> OracleSolution:
        raise NotImplementedError
