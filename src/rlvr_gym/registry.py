from __future__ import annotations

from rlvr_gym.families.graph_planning import GraphPlanningFamily
from rlvr_gym.families.scheduling import SchedulingFamily

FAMILY_REGISTRY = {
    "graph_planning": GraphPlanningFamily(),
    "scheduling": SchedulingFamily(),
}


def get_family(name: str):
    try:
        return FAMILY_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown family: {name!r}. Available: {sorted(FAMILY_REGISTRY)}") from exc


def list_families() -> list[dict[str, str]]:
    return [
        {"name": family.name, "description": family.description}
        for family in FAMILY_REGISTRY.values()
    ]
