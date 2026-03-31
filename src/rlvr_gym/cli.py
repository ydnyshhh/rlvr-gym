from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr_gym.core.exporters import (
    build_benchmark_splits,
    export_offline_transitions,
    export_sft_example,
    export_task_spec,
    rollout_oracle,
    write_benchmark_splits,
)
from rlvr_gym.core.types import FamilyConfig, RewardConfig, RewardMode
from rlvr_gym.core.utils import as_primitive
from rlvr_gym.registry import get_family, list_families


def _build_config(args: argparse.Namespace) -> FamilyConfig:
    reward_config = RewardConfig(mode=RewardMode(args.reward_mode))
    return FamilyConfig(
        difficulty=args.difficulty,
        observability=args.observability,
        reward_config=reward_config,
        attach_oracle=not getattr(args, "no_oracle", False),
        max_steps=args.max_steps,
    )


def _emit(payload) -> None:
    print(json.dumps(as_primitive(payload), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(prog="rlvr-gym", description="RLVR-Gym CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-families", help="List built-in environment families.")

    sample_parser = subparsers.add_parser("sample", help="Sample and export a single task instance.")
    sample_parser.add_argument("--family", required=True)
    sample_parser.add_argument("--seed", required=True, type=int)
    sample_parser.add_argument("--difficulty", default="medium")
    sample_parser.add_argument("--observability", default="full")
    sample_parser.add_argument("--reward-mode", default="shaped", choices=[mode.value for mode in RewardMode])
    sample_parser.add_argument("--max-steps", type=int)
    sample_parser.add_argument("--no-oracle", action="store_true")
    sample_parser.add_argument("--export", default="task", choices=["task", "oracle", "sft", "offline"])

    benchmark_parser = subparsers.add_parser("benchmark", help="Generate deterministic benchmark splits.")
    benchmark_parser.add_argument("--family", required=True)
    benchmark_parser.add_argument("--base-seed", required=True, type=int)
    benchmark_parser.add_argument("--difficulty", default="medium")
    benchmark_parser.add_argument("--observability", default="full")
    benchmark_parser.add_argument("--reward-mode", default="shaped", choices=[mode.value for mode in RewardMode])
    benchmark_parser.add_argument("--max-steps", type=int)
    benchmark_parser.add_argument("--train-count", type=int, default=0)
    benchmark_parser.add_argument("--validation-count", type=int, default=0)
    benchmark_parser.add_argument("--test-count", type=int, default=0)
    benchmark_parser.add_argument("--output-dir")
    benchmark_parser.add_argument("--include-oracle", action="store_true")

    args = parser.parse_args()

    if args.command == "list-families":
        _emit({"families": list_families()})
        return

    if args.command == "sample":
        family = get_family(args.family)
        config = _build_config(args)
        task = family.sample_instance(seed=args.seed, config=config)
        if args.export == "task":
            _emit(export_task_spec(task))
        elif args.export == "oracle":
            _emit(rollout_oracle(task))
        elif args.export == "sft":
            _emit(export_sft_example(task))
        elif args.export == "offline":
            _emit(export_offline_transitions(task))
        return

    if args.command == "benchmark":
        family = get_family(args.family)
        config = _build_config(args)
        split_counts = {
            "train": args.train_count,
            "validation": args.validation_count,
            "test": args.test_count,
        }
        splits = build_benchmark_splits(
            family=family,
            split_counts={name: count for name, count in split_counts.items() if count > 0},
            base_seed=args.base_seed,
            config=config,
            include_oracle=args.include_oracle,
        )
        if args.output_dir:
            written = write_benchmark_splits(Path(args.output_dir), splits)
            _emit({"splits": splits, "written_files": written})
        else:
            _emit({"splits": splits})


if __name__ == "__main__":
    main()
