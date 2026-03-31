from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OracleSolution:
    actions: tuple[Any, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    objective_value: float | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "metadata": dict(self.metadata),
            "objective_value": self.objective_value,
        }


class Oracle(ABC):
    @abstractmethod
    def solve(self) -> OracleSolution:
        raise NotImplementedError
