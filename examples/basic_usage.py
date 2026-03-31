from __future__ import annotations

import json

from rlvr_gym import (
    FamilyConfig,
    build_benchmark_splits,
    export_offline_transitions,
    export_task_spec,
    get_family,
    rollout_oracle,
)


def main() -> None:
    graph_family = get_family("graph_planning")
    graph_task = graph_family.sample_instance(seed=42, config=FamilyConfig(difficulty="medium"))
    print("Graph task spec:")
    print(json.dumps(export_task_spec(graph_task), indent=2, sort_keys=True))

    scheduling_family = get_family("scheduling")
    scheduling_task = scheduling_family.sample_instance(seed=99, config=FamilyConfig(difficulty="medium"))
    print("\nScheduling oracle rollout:")
    print(json.dumps(rollout_oracle(scheduling_task), indent=2, sort_keys=True))

    print("\nOffline RL transitions:")
    print(json.dumps(export_offline_transitions(scheduling_task), indent=2, sort_keys=True))

    splits = build_benchmark_splits(
        family=graph_family,
        split_counts={"train": 2, "validation": 1, "test": 1},
        base_seed=123,
        config=FamilyConfig(difficulty="easy"),
        include_oracle=True,
    )
    print("\nDeterministic benchmark splits:")
    print(json.dumps(splits, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
