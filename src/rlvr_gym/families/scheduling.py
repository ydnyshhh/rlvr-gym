from __future__ import annotations

import functools
import random
from dataclasses import dataclass
from typing import Any

from rlvr_gym.core.family import EnvironmentFamily
from rlvr_gym.core.oracle import Oracle, OracleSolution
from rlvr_gym.core.types import CanonicalAction, FamilyConfig, TaskObjective, TransitionResult
from rlvr_gym.core.verifier import (
    BaseVerifier,
    StepContext,
    TrajectoryContext,
    VerificationResult,
    VerificationScope,
    VerifierSuite,
)


@dataclass(frozen=True)
class SchedulingJob:
    job_id: str
    duration: int
    deadline: int
    prerequisites: tuple[str, ...]


@dataclass(frozen=True)
class ScheduledJob:
    job_id: str
    start: int
    finish: int
    tardiness: int


@dataclass(frozen=True)
class SchedulingWorld:
    jobs: tuple[SchedulingJob, ...]
    precedence_edges: tuple[tuple[str, str], ...]
    optimal_order: tuple[str, ...]
    optimal_total_tardiness: int


@dataclass(frozen=True)
class SchedulingState:
    current_time: int
    completed_jobs: tuple[str, ...]
    schedule: tuple[ScheduledJob, ...]
    total_tardiness: int


def _solve_optimal_schedule(jobs: tuple[SchedulingJob, ...]) -> tuple[tuple[str, ...], int]:
    index_by_job = {job.job_id: index for index, job in enumerate(jobs)}
    durations = tuple(job.duration for job in jobs)
    deadlines = tuple(job.deadline for job in jobs)
    prerequisite_masks = []
    for job in jobs:
        mask = 0
        for prerequisite in job.prerequisites:
            mask |= 1 << index_by_job[prerequisite]
        prerequisite_masks.append(mask)
    all_done_mask = (1 << len(jobs)) - 1
    time_by_mask = {mask: 0 for mask in range(all_done_mask + 1)}
    for mask in range(all_done_mask + 1):
        total = 0
        for index, duration in enumerate(durations):
            if mask & (1 << index):
                total += duration
        time_by_mask[mask] = total

    @functools.lru_cache(maxsize=None)
    def solve(mask: int) -> tuple[int, tuple[str, ...]]:
        if mask == all_done_mask:
            return 0, ()
        current_time = time_by_mask[mask]
        best_cost = None
        best_order: tuple[str, ...] = ()
        ready_indices = [
            index
            for index in range(len(jobs))
            if not mask & (1 << index) and prerequisite_masks[index] & ~mask == 0
        ]
        for index in ready_indices:
            finish = current_time + durations[index]
            tardiness = max(0, finish - deadlines[index])
            remaining_cost, remaining_order = solve(mask | (1 << index))
            total_cost = tardiness + remaining_cost
            candidate_order = (jobs[index].job_id,) + remaining_order
            if best_cost is None or total_cost < best_cost or (
                total_cost == best_cost and candidate_order < best_order
            ):
                best_cost = total_cost
                best_order = candidate_order
        assert best_cost is not None
        return best_cost, best_order

    cost, order = solve(0)
    return order, cost


def _ready_jobs(world: SchedulingWorld, state: SchedulingState) -> list[SchedulingJob]:
    completed = set(state.completed_jobs)
    return [
        job
        for job in world.jobs
        if job.job_id not in completed and set(job.prerequisites).issubset(completed)
    ]


class SchedulingActionVerifier(BaseVerifier):
    name = "scheduling_action_validity"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        passed = not context.transition.invalid_action
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.ACTION,
                passed=passed,
                score=1.0 if passed else 0.0,
                message="Chosen job must be ready and unscheduled.",
            ),
        )


class SchedulingStateVerifier(BaseVerifier):
    name = "scheduling_state_consistency"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        state = context.next_state
        schedule_job_ids = tuple(entry.job_id for entry in state.schedule)
        running_time = sum(entry.finish - entry.start for entry in state.schedule)
        passed = (
            len(schedule_job_ids) == len(set(schedule_job_ids))
            and state.completed_jobs == schedule_job_ids
            and state.current_time == running_time
        )
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.STATE,
                passed=passed,
                score=1.0 if passed else 0.0,
                message="Schedule state must be sequential, duplicate-free, and time-consistent.",
            ),
        )


class SchedulingGoalVerifier(BaseVerifier):
    name = "scheduling_goal_completion"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        if not context.transition.terminated:
            return ()
        passed = len(context.next_state.completed_jobs) == len(context.world.jobs)
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.GOAL,
                passed=passed,
                score=1.0 if passed else 0.0,
                message="A completed episode must schedule every job exactly once.",
            ),
        )


class SchedulingTrajectoryVerifier(BaseVerifier):
    name = "scheduling_trajectory_optimality"

    def evaluate_trajectory(self, context: TrajectoryContext) -> tuple[VerificationResult, ...]:
        schedule = context.final_state.schedule
        finish_by_job = {entry.job_id: entry.finish for entry in schedule}
        jobs_by_id = {job.job_id: job for job in context.world.jobs}
        prerequisites_ok = True
        total_tardiness = 0
        for entry in schedule:
            job = jobs_by_id[entry.job_id]
            total_tardiness += max(0, entry.finish - job.deadline)
            for prerequisite in job.prerequisites:
                prerequisite_finish = finish_by_job.get(prerequisite)
                if prerequisite_finish is None or prerequisite_finish > entry.start:
                    prerequisites_ok = False
                    break
            if not prerequisites_ok:
                break
        optimal = context.success and total_tardiness == context.world.optimal_total_tardiness
        return (
            VerificationResult(
                name="scheduling_precedence_respected",
                scope=VerificationScope.TRAJECTORY,
                passed=prerequisites_ok,
                score=1.0 if prerequisites_ok else 0.0,
                message="The produced schedule must satisfy all precedence constraints.",
            ),
            VerificationResult(
                name=self.name,
                scope=VerificationScope.TRAJECTORY,
                passed=optimal,
                score=1.0 if optimal else 0.0,
                message="Completed schedules can be checked against the exact optimal tardiness.",
                metadata={
                    "observed_total_tardiness": total_tardiness,
                    "optimal_total_tardiness": context.world.optimal_total_tardiness,
                },
            ),
        )


class SchedulingOracle(Oracle):
    def __init__(self, world: SchedulingWorld) -> None:
        self.world = world

    def solve(self) -> OracleSolution:
        actions = tuple(
            {"name": "schedule", "arguments": {"job_id": job_id}}
            for job_id in self.world.optimal_order
        )
        return OracleSolution(
            actions=actions,
            metadata={
                "strategy": "exact_dynamic_programming",
                "job_order": list(self.world.optimal_order),
            },
            objective_value=self.world.optimal_total_tardiness,
        )


class SchedulingFamily(EnvironmentFamily):
    name = "scheduling"
    description = "Single-machine job scheduling with deadlines and precedence constraints."

    def sample_generation_params(self, config: FamilyConfig, rng: random.Random) -> dict[str, Any]:
        ranges = {
            "easy": {"job_range": (4, 5), "duration_range": (1, 3), "prereq_prob": 0.15},
            "medium": {"job_range": (5, 7), "duration_range": (1, 4), "prereq_prob": 0.25},
            "hard": {"job_range": (7, 9), "duration_range": (1, 5), "prereq_prob": 0.35},
        }
        spec = ranges.get(config.difficulty, ranges["medium"])
        return {
            "num_jobs": rng.randint(*spec["job_range"]),
            "duration_range": spec["duration_range"],
            "prerequisite_probability": spec["prereq_prob"],
            "observability": config.observability,
        }

    def sample_world(self, generation_params: dict[str, Any], rng: random.Random) -> SchedulingWorld:
        num_jobs = generation_params["num_jobs"]
        job_ids = [f"job_{index}" for index in range(num_jobs)]
        durations = [rng.randint(*generation_params["duration_range"]) for _ in range(num_jobs)]
        total_duration = sum(durations)
        prerequisites_by_job: dict[str, tuple[str, ...]] = {}
        precedence_edges: list[tuple[str, str]] = []
        for index, job_id in enumerate(job_ids):
            prerequisites: list[str] = []
            for previous_index in range(index):
                if rng.random() < generation_params["prerequisite_probability"]:
                    prerequisite = job_ids[previous_index]
                    prerequisites.append(prerequisite)
                    precedence_edges.append((prerequisite, job_id))
            prerequisites_by_job[job_id] = tuple(sorted(set(prerequisites)))

        jobs = []
        for job_id, duration in zip(job_ids, durations):
            slack = rng.randint(0, max(2, total_duration // 2))
            deadline = max(duration, total_duration // 2 + slack)
            jobs.append(
                SchedulingJob(
                    job_id=job_id,
                    duration=duration,
                    deadline=deadline,
                    prerequisites=prerequisites_by_job[job_id],
                )
            )
        frozen_jobs = tuple(jobs)
        optimal_order, optimal_total_tardiness = _solve_optimal_schedule(frozen_jobs)
        return SchedulingWorld(
            jobs=frozen_jobs,
            precedence_edges=tuple(precedence_edges),
            optimal_order=optimal_order,
            optimal_total_tardiness=optimal_total_tardiness,
        )

    def derive_objective(
        self,
        world: SchedulingWorld,
        generation_params: dict[str, Any],
        rng: random.Random,
    ) -> TaskObjective:
        return TaskObjective(
            name="minimize_total_tardiness",
            description="Produce a valid single-machine schedule that minimizes total tardiness.",
            success_criteria={
                "must_schedule_all_jobs": True,
                "precedence_constraints": list(world.precedence_edges),
                "optimal_total_tardiness": world.optimal_total_tardiness,
            },
            optimization_target="minimize_total_tardiness",
        )

    def initial_state(
        self,
        world: SchedulingWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> SchedulingState:
        return SchedulingState(current_time=0, completed_jobs=(), schedule=(), total_tardiness=0)

    def observe(
        self,
        world: SchedulingWorld,
        state: SchedulingState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        ready_jobs = _ready_jobs(world, state)
        observation = {
            "mode": generation_params["observability"],
            "current_time": state.current_time,
            "completed_jobs": list(state.completed_jobs),
            "ready_jobs": [
                {
                    "job_id": job.job_id,
                    "duration": job.duration,
                    "deadline": job.deadline,
                    "prerequisites": list(job.prerequisites),
                }
                for job in ready_jobs
            ],
            "schedule": [
                {
                    "job_id": entry.job_id,
                    "start": entry.start,
                    "finish": entry.finish,
                    "tardiness": entry.tardiness,
                }
                for entry in state.schedule
            ],
            "total_tardiness": state.total_tardiness,
        }
        if generation_params["observability"] == "full":
            observation["jobs"] = [
                {
                    "job_id": job.job_id,
                    "duration": job.duration,
                    "deadline": job.deadline,
                    "prerequisites": list(job.prerequisites),
                    "scheduled": job.job_id in state.completed_jobs,
                }
                for job in world.jobs
            ]
        return observation

    def valid_actions(
        self,
        world: SchedulingWorld,
        state: SchedulingState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> list[CanonicalAction]:
        return [CanonicalAction(name="schedule", arguments={"job_id": job.job_id}) for job in _ready_jobs(world, state)]

    def transition(
        self,
        world: SchedulingWorld,
        state: SchedulingState,
        action: CanonicalAction,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> TransitionResult:
        ready_jobs = {job.job_id: job for job in _ready_jobs(world, state)}
        job_id = action.arguments.get("job_id")
        if action.name != "schedule" or job_id not in ready_jobs:
            return TransitionResult(
                next_state=state,
                invalid_action=True,
                reward_hints={"schedule_quality": -0.25},
                info={"invalid_action": True, "reason": "job_not_ready_or_unknown"},
            )

        job = ready_jobs[job_id]
        start = state.current_time
        finish = start + job.duration
        tardiness = max(0, finish - job.deadline)
        scheduled_entry = ScheduledJob(job_id=job.job_id, start=start, finish=finish, tardiness=tardiness)
        next_state = SchedulingState(
            current_time=finish,
            completed_jobs=state.completed_jobs + (job.job_id,),
            schedule=state.schedule + (scheduled_entry,),
            total_tardiness=state.total_tardiness + tardiness,
        )
        success = len(next_state.completed_jobs) == len(world.jobs)
        deadline_scale = max(1, max(existing.deadline for existing in world.jobs))
        return TransitionResult(
            next_state=next_state,
            terminated=success,
            success=success,
            reward_hints={
                "schedule_quality": -tardiness / deadline_scale,
                "completion_progress": 1.0 / len(world.jobs),
            },
            info={
                "scheduled_job": job.job_id,
                "job_finish_time": finish,
                "incremental_tardiness": tardiness,
                "total_tardiness": next_state.total_tardiness,
            },
        )

    def build_verifier_suite(
        self,
        world: SchedulingWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> VerifierSuite:
        return VerifierSuite(
            [
                SchedulingActionVerifier(),
                SchedulingStateVerifier(),
                SchedulingGoalVerifier(),
                SchedulingTrajectoryVerifier(),
            ]
        )

    def build_oracle(
        self,
        world: SchedulingWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> SchedulingOracle:
        return SchedulingOracle(world)

    def recommended_max_steps(self, generation_params: dict[str, Any]) -> int:
        return generation_params["num_jobs"] * 2

    def task_metadata(
        self,
        world: SchedulingWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "num_jobs": len(world.jobs),
            "num_constraints": len(world.precedence_edges),
            "optimal_total_tardiness": world.optimal_total_tardiness,
        }
