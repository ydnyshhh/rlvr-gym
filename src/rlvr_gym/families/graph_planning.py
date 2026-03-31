from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Any

from rlvr_gym.core.family import EnvironmentFamily
from rlvr_gym.core.oracle import Oracle, OracleSolution, ProofCertificate
from rlvr_gym.core.types import CanonicalAction, FamilyConfig, TaskObjective, TaskSpace, TransitionResult
from rlvr_gym.core.verifier import (
    BaseVerifier,
    StepContext,
    TrajectoryContext,
    VerificationKind,
    VerificationResult,
    VerificationScope,
    VerifierSuite,
)


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    cost: int


@dataclass(frozen=True)
class GraphPlanningWorld:
    nodes: tuple[str, ...]
    edges: tuple[GraphEdge, ...]
    adjacency: dict[str, tuple[GraphEdge, ...]]
    start_node: str
    goal_node: str
    directed: bool
    shortest_cost: int
    shortest_path: tuple[str, ...]
    distance_to_goal: dict[str, int]


@dataclass(frozen=True)
class GraphPlanningState:
    current_node: str
    path: tuple[str, ...]
    visited_nodes: tuple[str, ...]
    total_cost: int


def _dijkstra(world: GraphPlanningWorld, start: str, goal: str) -> tuple[int, tuple[str, ...]]:
    queue: list[tuple[int, str, tuple[str, ...]]] = [(0, start, (start,))]
    best_costs: dict[str, int] = {start: 0}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == goal:
            return cost, path
        if cost > best_costs.get(node, math.inf):
            continue
        for edge in world.adjacency[node]:
            next_cost = cost + edge.cost
            if next_cost < best_costs.get(edge.target, math.inf):
                best_costs[edge.target] = next_cost
                heapq.heappush(queue, (next_cost, edge.target, path + (edge.target,)))
    raise ValueError("Generated graph has no valid path to the goal.")


def _graph_distance_table(
    nodes: tuple[str, ...],
    adjacency: dict[str, tuple[GraphEdge, ...]],
    goal: str,
) -> dict[str, int]:
    placeholder_world = GraphPlanningWorld(
        nodes=nodes,
        edges=tuple(edge for edges in adjacency.values() for edge in edges),
        adjacency=adjacency,
        start_node=nodes[0],
        goal_node=goal,
        directed=True,
        shortest_cost=0,
        shortest_path=(),
        distance_to_goal={},
    )
    distances: dict[str, int] = {}
    for node in nodes:
        cost, _ = _dijkstra(placeholder_world, node, goal)
        distances[node] = cost
    return distances


class GraphActionVerifier(BaseVerifier):
    name = "graph_action_validity"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        passed = not context.transition.invalid_action
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.ACTION,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="Action must move along an outgoing edge.",
            ),
        )


class GraphStateVerifier(BaseVerifier):
    name = "graph_state_consistency"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        state = context.next_state
        passed = (
            state.current_node in context.world.nodes
            and state.path
            and state.path[-1] == state.current_node
            and state.total_cost >= 0
        )
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.STATE,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="State must reference a valid node and a non-negative path cost.",
            ),
        )


class GraphGoalVerifier(BaseVerifier):
    name = "graph_goal_reached"

    def evaluate_step(self, context: StepContext) -> tuple[VerificationResult, ...]:
        if not context.transition.terminated:
            return ()
        passed = context.next_state.current_node == context.world.goal_node and context.transition.success
        return (
            VerificationResult(
                name=self.name,
                scope=VerificationScope.GOAL,
                passed=passed,
                score=1.0 if passed else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="Termination should coincide with reaching the goal node.",
            ),
        )


class GraphTrajectoryVerifier(BaseVerifier):
    name = "graph_trajectory_optimality"

    def evaluate_trajectory(self, context: TrajectoryContext) -> tuple[VerificationResult, ...]:
        state = context.final_state
        if not state.path:
            return ()
        edges = {
            (edge.source, edge.target): edge.cost
            for adjacency in context.world.adjacency.values()
            for edge in adjacency
        }
        running_cost = 0
        legal = True
        for source, target in zip(state.path[:-1], state.path[1:]):
            if (source, target) not in edges:
                legal = False
                break
            running_cost += edges[(source, target)]
        optimal = context.success and running_cost == context.world.shortest_cost
        return (
            VerificationResult(
                name="graph_trajectory_legal",
                scope=VerificationScope.TRAJECTORY,
                passed=legal,
                score=1.0 if legal else 0.0,
                kind=VerificationKind.FEASIBILITY,
                weight=2.0,
                hard=True,
                message="The path must follow legal graph edges.",
            ),
            VerificationResult(
                name=self.name,
                scope=VerificationScope.TRAJECTORY,
                passed=optimal,
                score=1.0 if optimal else 0.0,
                kind=VerificationKind.QUALITY,
                weight=1.0,
                hard=False,
                message="Successful solutions can be checked against the shortest-path optimum.",
                metadata={"path_cost": running_cost, "optimal_cost": context.world.shortest_cost},
            ),
        )


class GraphPlanningOracle(Oracle):
    def __init__(self, world: GraphPlanningWorld) -> None:
        self.world = world

    def is_feasible(self) -> bool | None:
        return True

    def estimate_difficulty(self) -> float | None:
        return float(len(self.world.nodes) + len(self.world.edges) / max(1, len(self.world.nodes)))

    def solve(self) -> OracleSolution:
        actions = tuple(
            {"name": "move", "arguments": {"target": node}}
            for node in self.world.shortest_path[1:]
        )
        return OracleSolution(
            actions=actions,
            metadata={"strategy": "shortest_path", "path": list(self.world.shortest_path)},
            objective_value=self.world.shortest_cost,
            feasible=True,
            optimal=True,
            difficulty_estimate=self.estimate_difficulty(),
            certificate=ProofCertificate(
                feasible=True,
                optimal=True,
                summary="Exact shortest path computed with Dijkstra search.",
                witness={
                    "path": list(self.world.shortest_path),
                    "optimal_cost": self.world.shortest_cost,
                },
            ),
        )


class GraphPlanningFamily(EnvironmentFamily):
    name = "graph_planning"
    description = "Shortest-path navigation over procedurally generated weighted graphs."

    def sample_generation_params(self, config: FamilyConfig, rng: random.Random) -> dict[str, Any]:
        ranges = {
            "easy": {"node_range": (4, 6), "edge_prob": 0.30, "max_cost": 4, "directed": False},
            "medium": {"node_range": (6, 9), "edge_prob": 0.40, "max_cost": 7, "directed": False},
            "hard": {"node_range": (9, 12), "edge_prob": 0.50, "max_cost": 9, "directed": True},
        }
        spec = ranges.get(config.difficulty, ranges["medium"])
        return {
            "num_nodes": rng.randint(*spec["node_range"]),
            "edge_probability": spec["edge_prob"],
            "max_cost": spec["max_cost"],
            "directed": spec["directed"],
            "observability": config.observability,
        }

    def sample_world(self, generation_params: dict[str, Any], rng: random.Random) -> GraphPlanningWorld:
        nodes = tuple(f"n{i}" for i in range(generation_params["num_nodes"]))
        start_node = nodes[0]
        goal_node = nodes[-1]
        directed = bool(generation_params["directed"])
        adjacency: dict[str, list[GraphEdge]] = {node: [] for node in nodes}

        def add_edge(source: str, target: str, cost: int) -> None:
            if any(existing.target == target for existing in adjacency[source]):
                return
            adjacency[source].append(GraphEdge(source=source, target=target, cost=cost))

        for source, target in zip(nodes[:-1], nodes[1:]):
            cost = rng.randint(1, generation_params["max_cost"])
            add_edge(source, target, cost)
            if not directed:
                add_edge(target, source, cost)

        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i == j:
                    continue
                if not directed and j <= i:
                    continue
                if rng.random() > generation_params["edge_probability"]:
                    continue
                if directed:
                    edge_source, edge_target = (source, target) if rng.random() < 0.5 else (target, source)
                    add_edge(edge_source, edge_target, rng.randint(1, generation_params["max_cost"]))
                else:
                    cost = rng.randint(1, generation_params["max_cost"])
                    add_edge(source, target, cost)
                    add_edge(target, source, cost)

        frozen_adjacency = {node: tuple(sorted(edges, key=lambda edge: edge.target)) for node, edges in adjacency.items()}
        placeholder_world = GraphPlanningWorld(
            nodes=nodes,
            edges=tuple(edge for edges in frozen_adjacency.values() for edge in edges),
            adjacency=frozen_adjacency,
            start_node=start_node,
            goal_node=goal_node,
            directed=directed,
            shortest_cost=0,
            shortest_path=(),
            distance_to_goal={},
        )
        shortest_cost, shortest_path = _dijkstra(placeholder_world, start_node, goal_node)
        distance_to_goal = _graph_distance_table(nodes, frozen_adjacency, goal_node)
        return GraphPlanningWorld(
            nodes=nodes,
            edges=tuple(edge for edges in frozen_adjacency.values() for edge in edges),
            adjacency=frozen_adjacency,
            start_node=start_node,
            goal_node=goal_node,
            directed=directed,
            shortest_cost=shortest_cost,
            shortest_path=shortest_path,
            distance_to_goal=distance_to_goal,
        )

    def derive_objective(
        self,
        world: GraphPlanningWorld,
        generation_params: dict[str, Any],
        rng: random.Random,
    ) -> TaskObjective:
        return TaskObjective(
            name="reach_goal_with_min_cost",
            description=f"Navigate from {world.start_node} to {world.goal_node} while minimizing path cost.",
            success_criteria={
                "start_node": world.start_node,
                "goal_node": world.goal_node,
                "optimal_cost": world.shortest_cost,
            },
            optimization_target="minimize_total_path_cost",
            constraint_spec={"must_follow_edges": True, "must_terminate_at_goal": True},
            quality_metric="total_path_cost",
        )

    def initial_state(
        self,
        world: GraphPlanningWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> GraphPlanningState:
        return GraphPlanningState(
            current_node=world.start_node,
            path=(world.start_node,),
            visited_nodes=(world.start_node,),
            total_cost=0,
        )

    def observe(
        self,
        world: GraphPlanningWorld,
        state: GraphPlanningState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        neighbors = [{"target": edge.target, "cost": edge.cost} for edge in world.adjacency[state.current_node]]
        observation = {
            "mode": generation_params["observability"],
            "current_node": state.current_node,
            "goal_node": world.goal_node,
            "neighbors": neighbors,
            "path": list(state.path),
            "total_cost": state.total_cost,
        }
        if generation_params["observability"] == "full":
            observation["graph"] = {
                "nodes": list(world.nodes),
                "edges": [
                    {"source": edge.source, "target": edge.target, "cost": edge.cost}
                    for edge in world.edges
                ],
            }
        return observation

    def build_task_space(
        self,
        world: GraphPlanningWorld,
        objective: TaskObjective,
        initial_state: GraphPlanningState,
        generation_params: dict[str, Any],
    ) -> TaskSpace:
        return TaskSpace(
            observation_schema={
                "type": "object",
                "fields": {
                    "current_node": "string",
                    "goal_node": "string",
                    "neighbors": "list[{target: string, cost: int}]",
                    "path": "list[string]",
                    "total_cost": "int",
                },
                "observability": generation_params["observability"],
            },
            action_schema={
                "type": "canonical_object",
                "actions": [{"name": "move", "arguments": {"target": "node_id"}}],
            },
            runtime_api="gym_like",
            notes="Gym-like reset()/step() tuples; a Gymnasium adapter is not yet provided.",
        )

    def valid_actions(
        self,
        world: GraphPlanningWorld,
        state: GraphPlanningState,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> list[CanonicalAction]:
        return [CanonicalAction(name="move", arguments={"target": edge.target}) for edge in world.adjacency[state.current_node]]

    def transition(
        self,
        world: GraphPlanningWorld,
        state: GraphPlanningState,
        action: CanonicalAction,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> TransitionResult:
        neighbors = {edge.target: edge for edge in world.adjacency[state.current_node]}
        target = action.arguments.get("target")
        if action.name != "move" or target not in neighbors:
            return TransitionResult(
                next_state=state,
                invalid_action=True,
                reward_hints={"distance_progress": -0.25},
                info={"invalid_action": True, "reason": "unknown_or_illegal_edge"},
            )

        edge = neighbors[target]
        next_state = GraphPlanningState(
            current_node=edge.target,
            path=state.path + (edge.target,),
            visited_nodes=tuple(dict.fromkeys(state.visited_nodes + (edge.target,))),
            total_cost=state.total_cost + edge.cost,
        )
        prev_distance = world.distance_to_goal[state.current_node]
        next_distance = world.distance_to_goal[next_state.current_node]
        success = next_state.current_node == world.goal_node
        return TransitionResult(
            next_state=next_state,
            terminated=success,
            success=success,
            reward_hints={
                "distance_progress": (prev_distance - next_distance) / max(1, world.shortest_cost),
                "cost_penalty": -edge.cost / max(1, world.shortest_cost),
            },
            info={
                "edge_cost": edge.cost,
                "path_cost": next_state.total_cost,
                "distance_to_goal": next_distance,
            },
        )

    def build_verifier_suite(
        self,
        world: GraphPlanningWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> VerifierSuite:
        return VerifierSuite([GraphActionVerifier(), GraphStateVerifier(), GraphGoalVerifier(), GraphTrajectoryVerifier()])

    def build_oracle(
        self,
        world: GraphPlanningWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> GraphPlanningOracle:
        return GraphPlanningOracle(world)

    def recommended_max_steps(self, generation_params: dict[str, Any]) -> int:
        return generation_params["num_nodes"] * 2

    def task_metadata(
        self,
        world: GraphPlanningWorld,
        objective: TaskObjective,
        generation_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "num_nodes": len(world.nodes),
            "num_edges": len(world.edges),
            "optimal_cost": world.shortest_cost,
        }
