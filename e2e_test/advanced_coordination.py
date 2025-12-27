"""Advanced coordination features.

Implements:
- hi_moe-7b5: Soft routing (weighted specialist contributions)
- hi_moe-2tx: State continuity (context vector evolution)
- hi_moe-2ze: DAG support for parallel task execution

These features enable more sophisticated coordination beyond
simple linear specialist routing.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Soft Routing (hi_moe-7b5)
# =============================================================================

@dataclass
class SpecialistWeight:
    """Weight for a specialist's contribution."""
    specialist: str
    weight: float  # 0-1, sum should be ~1
    confidence: float  # How confident in this weight


@dataclass
class SoftRoutingResult:
    """Result of soft routing decision."""
    weights: list[SpecialistWeight]
    primary: str  # Highest weighted specialist
    strategy: str  # "soft", "hard", "ensemble"
    embedding_similarity: dict[str, float]


class SoftRouter:
    """Routes tasks using weighted specialist contributions (hi_moe-7b5).

    Instead of hard routing to a single specialist, computes soft
    weights based on task similarity to specialist domains.

    Usage:
        router = SoftRouter()
        result = router.route("Implement sorting algorithm")
        # result.weights = [("algorithms", 0.6), ("python", 0.3), ("math", 0.1)]
    """

    def __init__(
        self,
        specialists: list[str] | None = None,
        temperature: float = 1.0,
    ):
        """
        Args:
            specialists: Available specialists
            temperature: Softmax temperature (lower = more peaked)
        """
        self.specialists = specialists or ["python", "math", "algorithms", "debugging", "refactoring"]
        self.temperature = temperature

        # Domain embeddings (simplified as keyword vectors)
        self.domain_keywords = {
            "python": ["python", "code", "implement", "function", "class", "test", "write"],
            "math": ["math", "prove", "theorem", "formula", "calculate", "analysis", "complexity"],
            "algorithms": ["algorithm", "sort", "search", "graph", "tree", "dynamic", "optimal"],
            "debugging": ["debug", "fix", "error", "bug", "trace", "issue", "broken"],
            "refactoring": ["refactor", "clean", "improve", "restructure", "pattern", "simplify"],
        }

        # Learned weight adjustments
        self.weight_adjustments: dict[str, float] = defaultdict(lambda: 1.0)

    def route(self, objective: str, context: dict | None = None) -> SoftRoutingResult:
        """Compute soft routing weights for a task."""
        objective_lower = objective.lower()

        # Compute similarity to each specialist domain
        similarities = {}
        for specialist, keywords in self.domain_keywords.items():
            if specialist not in self.specialists:
                continue
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in objective_lower)
            # Normalize by number of keywords
            similarity = matches / len(keywords) if keywords else 0
            # Apply learned adjustments
            similarity *= self.weight_adjustments[specialist]
            similarities[specialist] = similarity

        # Apply softmax to get weights
        weights = self._softmax(similarities)

        # Build result
        specialist_weights = [
            SpecialistWeight(
                specialist=spec,
                weight=weight,
                confidence=similarities[spec],
            )
            for spec, weight in sorted(weights.items(), key=lambda x: -x[1])
        ]

        primary = specialist_weights[0].specialist if specialist_weights else "general"

        # Determine strategy
        if specialist_weights and specialist_weights[0].weight > 0.7:
            strategy = "hard"  # Clear winner
        elif len([w for w in specialist_weights if w.weight > 0.2]) > 1:
            strategy = "ensemble"  # Multiple contributors
        else:
            strategy = "soft"  # Gradual weights

        return SoftRoutingResult(
            weights=specialist_weights,
            primary=primary,
            strategy=strategy,
            embedding_similarity=similarities,
        )

    def _softmax(self, similarities: dict[str, float]) -> dict[str, float]:
        """Apply softmax to similarity scores."""
        if not similarities:
            return {}

        # Compute exp(x/T) for each
        scores = {k: math.exp(v / self.temperature) for k, v in similarities.items()}
        total = sum(scores.values())

        if total == 0:
            # Uniform distribution
            n = len(scores)
            return {k: 1.0 / n for k in scores}

        return {k: v / total for k, v in scores.items()}

    def update_weights(self, specialist: str, success: bool, learning_rate: float = 0.1) -> None:
        """Update weight adjustments based on outcome."""
        current = self.weight_adjustments[specialist]
        if success:
            self.weight_adjustments[specialist] = current + learning_rate * (1.2 - current)
        else:
            self.weight_adjustments[specialist] = current - learning_rate * (current - 0.8)


# =============================================================================
# State Continuity (hi_moe-2tx)
# =============================================================================

@dataclass
class ContextVector:
    """Compressed context that evolves across tasks (hi_moe-2tx).

    Maintains a sense of trajectory and continuity rather than
    resetting state between tasks.
    """
    # Core state vector (compressed representation)
    vector: list[float] = field(default_factory=lambda: [0.0] * 64)

    # Trajectory info
    task_count: int = 0
    success_momentum: float = 0.5  # EMA of recent successes
    specialist_affinity: dict[str, float] = field(default_factory=dict)

    # Recent history (for context)
    recent_objectives: list[str] = field(default_factory=list)
    recent_outcomes: list[bool] = field(default_factory=list)

    # Evolution parameters
    decay_rate: float = 0.95  # How fast old state decays
    max_history: int = 10


class StateEvolver:
    """Evolves context state smoothly across tasks (hi_moe-2tx).

    Maintains continuity by:
    - Updating rather than resetting state
    - Tracking momentum and affinity
    - Preserving recent trajectory

    Usage:
        evolver = StateEvolver()

        # Before task
        context = evolver.get_context()

        # After task
        evolver.evolve(task_objective, outcome, specialist)
    """

    def __init__(self):
        self.context = ContextVector()
        self._hash_cache: dict[str, list[float]] = {}

    def get_context(self) -> ContextVector:
        """Get current context state."""
        return self.context

    def evolve(
        self,
        objective: str,
        success: bool,
        specialist: str | None = None,
        features: dict | None = None,
    ) -> ContextVector:
        """Evolve state based on task outcome.

        Updates state smoothly rather than resetting.
        """
        ctx = self.context

        # Update task count
        ctx.task_count += 1

        # Update success momentum (EMA)
        success_val = 1.0 if success else 0.0
        ctx.success_momentum = 0.9 * ctx.success_momentum + 0.1 * success_val

        # Update specialist affinity
        if specialist:
            current = ctx.specialist_affinity.get(specialist, 0.5)
            ctx.specialist_affinity[specialist] = 0.8 * current + 0.2 * success_val

        # Update vector (simplified: hash objective to vector, blend with current)
        obj_vector = self._objective_to_vector(objective)
        for i in range(len(ctx.vector)):
            # Decay old state, blend in new
            ctx.vector[i] = ctx.decay_rate * ctx.vector[i] + (1 - ctx.decay_rate) * obj_vector[i]

        # Update history
        ctx.recent_objectives.append(objective[:100])
        ctx.recent_outcomes.append(success)
        if len(ctx.recent_objectives) > ctx.max_history:
            ctx.recent_objectives.pop(0)
            ctx.recent_outcomes.pop(0)

        logger.debug(
            f"[StateEvolver] Evolved state: task #{ctx.task_count}, "
            f"momentum={ctx.success_momentum:.2f}"
        )

        return ctx

    def _objective_to_vector(self, objective: str) -> list[float]:
        """Convert objective to vector (simplified hash-based)."""
        if objective in self._hash_cache:
            return self._hash_cache[objective]

        # Hash to seed
        h = hashlib.md5(objective.encode()).hexdigest()
        seed = int(h[:8], 16)

        # Generate pseudo-random vector (thread-safe RNG)
        rng = np.random.default_rng(seed)
        vector = list(rng.standard_normal(64) * 0.1)

        self._hash_cache[objective] = vector
        return vector

    def get_trajectory_summary(self) -> dict:
        """Get summary of current trajectory."""
        ctx = self.context
        return {
            "task_count": ctx.task_count,
            "success_momentum": ctx.success_momentum,
            "recent_success_rate": (
                sum(ctx.recent_outcomes) / len(ctx.recent_outcomes)
                if ctx.recent_outcomes else 0.5
            ),
            "top_specialists": sorted(
                ctx.specialist_affinity.items(),
                key=lambda x: -x[1]
            )[:3],
            "trajectory_direction": "improving" if ctx.success_momentum > 0.6 else "struggling",
        }


# =============================================================================
# DAG Execution (hi_moe-2ze)
# =============================================================================

@dataclass
class DAGNode:
    """A node in the task dependency graph."""
    id: str
    task: str
    specialist: str | None = None
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: str | None = None


@dataclass
class DAGPlan:
    """A dependency graph of tasks for parallel execution."""
    nodes: dict[str, DAGNode] = field(default_factory=dict)
    root_nodes: list[str] = field(default_factory=list)  # Nodes with no dependencies

    def add_node(
        self,
        node_id: str,
        task: str,
        specialist: str | None = None,
        depends_on: list[str] | None = None,
    ) -> DAGNode:
        """Add a node to the DAG."""
        node = DAGNode(
            id=node_id,
            task=task,
            specialist=specialist,
            depends_on=depends_on or [],
        )
        self.nodes[node_id] = node

        # Update root nodes
        if not node.depends_on:
            self.root_nodes.append(node_id)

        return node

    def get_ready_nodes(self) -> list[DAGNode]:
        """Get nodes that are ready to execute (dependencies satisfied)."""
        ready = []
        for node in self.nodes.values():
            if node.status != "pending":
                continue

            # Check all dependencies completed
            deps_satisfied = all(
                self.nodes[dep].status == "completed"
                for dep in node.depends_on
                if dep in self.nodes
            )

            if deps_satisfied:
                ready.append(node)

        return ready

    def is_complete(self) -> bool:
        """Check if all nodes are completed or failed."""
        return all(n.status in ("completed", "failed") for n in self.nodes.values())

    def get_results(self) -> dict[str, Any]:
        """Get results from all completed nodes."""
        return {
            node_id: node.result
            for node_id, node in self.nodes.items()
            if node.status == "completed"
        }


class DAGExecutor:
    """Executes task DAGs with parallel branches (hi_moe-2ze).

    Supports:
    - Parallel execution of independent tasks
    - Dependency resolution
    - Result aggregation

    Usage:
        executor = DAGExecutor(execute_fn)

        plan = DAGPlan()
        plan.add_node("a", "Task A")
        plan.add_node("b", "Task B")
        plan.add_node("c", "Task C", depends_on=["a", "b"])

        results = await executor.execute(plan)
    """

    def __init__(
        self,
        execute_fn: Callable[[str, str | None], Coroutine[Any, Any, tuple[bool, Any]]],
        max_parallel: int = 4,
    ):
        """
        Args:
            execute_fn: Async function (task, specialist) -> (success, result)
            max_parallel: Max parallel executions
        """
        self.execute_fn = execute_fn
        self.max_parallel = max_parallel
        self.semaphore = asyncio.Semaphore(max_parallel)

    async def execute(self, plan: DAGPlan) -> dict[str, Any]:
        """Execute a DAG plan with parallel branches."""
        logger.info(f"[DAGExecutor] Starting execution of {len(plan.nodes)} nodes")

        while not plan.is_complete():
            ready = plan.get_ready_nodes()

            if not ready:
                # Check for deadlock (pending nodes but none ready)
                pending = [n for n in plan.nodes.values() if n.status == "pending"]
                if pending:
                    logger.error(f"[DAGExecutor] Deadlock: {len(pending)} pending, none ready")
                    for node in pending:
                        node.status = "failed"
                        node.error = "Deadlock: dependencies cannot be satisfied"
                break

            # Execute ready nodes in parallel
            tasks = [self._execute_node(node) for node in ready]
            await asyncio.gather(*tasks)

        logger.info(f"[DAGExecutor] Completed: {sum(1 for n in plan.nodes.values() if n.status == 'completed')}/{len(plan.nodes)} succeeded")

        return plan.get_results()

    async def _execute_node(self, node: DAGNode) -> None:
        """Execute a single node."""
        async with self.semaphore:
            node.status = "running"
            logger.info(f"[DAGExecutor] Running node {node.id}: {node.task[:50]}...")

            try:
                success, result = await self.execute_fn(node.task, node.specialist)

                if success:
                    node.status = "completed"
                    node.result = result
                    logger.info(f"[DAGExecutor] Node {node.id} completed")
                else:
                    node.status = "failed"
                    node.error = str(result) if result else "Execution failed"
                    logger.warning(f"[DAGExecutor] Node {node.id} failed: {node.error}")

            except Exception as e:
                node.status = "failed"
                node.error = str(e)
                logger.error(f"[DAGExecutor] Node {node.id} exception: {e}")


# =============================================================================
# Integration Example
# =============================================================================

async def demo():
    """Demo of advanced coordination features."""

    # Soft routing
    print("=" * 60)
    print("SOFT ROUTING DEMO")
    print("=" * 60)

    router = SoftRouter()

    tasks = [
        "Implement a binary search algorithm",
        "Prove the time complexity is O(log n)",
        "Fix the off-by-one error in the code",
    ]

    for task in tasks:
        result = router.route(task)
        print(f"\nTask: {task}")
        print(f"  Strategy: {result.strategy}")
        print(f"  Primary: {result.primary}")
        print(f"  Weights: {[(w.specialist, f'{w.weight:.2f}') for w in result.weights[:3]]}")

    # State continuity
    print("\n" + "=" * 60)
    print("STATE CONTINUITY DEMO")
    print("=" * 60)

    evolver = StateEvolver()

    outcomes = [
        ("Implement sorting function", True, "python"),
        ("Optimize the algorithm", True, "algorithms"),
        ("Fix memory leak", False, "debugging"),
        ("Refactor the code", True, "refactoring"),
    ]

    for objective, success, specialist in outcomes:
        evolver.evolve(objective, success, specialist)

    print(f"\nTrajectory summary: {evolver.get_trajectory_summary()}")

    # DAG execution
    print("\n" + "=" * 60)
    print("DAG EXECUTION DEMO")
    print("=" * 60)

    async def mock_execute(task: str, specialist: str | None) -> tuple[bool, str]:
        await asyncio.sleep(0.1)  # Simulate work
        return True, f"Result for: {task[:30]}"

    executor = DAGExecutor(mock_execute, max_parallel=2)

    plan = DAGPlan()
    plan.add_node("parse", "Parse the input data")
    plan.add_node("validate", "Validate the data format")
    plan.add_node("transform_a", "Transform branch A", depends_on=["parse", "validate"])
    plan.add_node("transform_b", "Transform branch B", depends_on=["parse", "validate"])
    plan.add_node("merge", "Merge results", depends_on=["transform_a", "transform_b"])

    print(f"\nDAG nodes: {list(plan.nodes.keys())}")
    print(f"Root nodes: {plan.root_nodes}")

    results = await executor.execute(plan)
    print(f"\nResults: {list(results.keys())}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())
