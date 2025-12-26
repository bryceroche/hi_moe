"""Runner control loop for Hi-MoE orchestration.

Implements issue hi_moe-xv1: Python control loop that orchestrates all tiers.

Responsibilities:
1. Receive problem
2. Call Architect for goal/delegation
3. Call Dispatcher for step sequence
4. For each step call appropriate specialist
5. Run CodeRunner if code generated
6. Write outcomes to TaskContext
7. Feed results back up the chain
8. Handle retries and re-planning on failure
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .tiers import (
    AbstractArchitect,
    LLMClient,
    MockLLMClient,
    Outcome,
    ProgressMonitor,
    RoutingDispatcher,
    SpecializedFleet,
    Task,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class RunStatus(Enum):
    """Status of a run through the hierarchy."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TaskContext:
    """In-memory state management for a single run.

    Implements hi_moe-joy: Simple dict-based runtime state.
    Upgrade to Redis/SQLite only if scaling problems arise.
    """
    run_id: str
    data: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by key."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set value at key."""
        self.data[key] = value

    def append(self, key: str, value: Any) -> None:
        """Append value to list at key."""
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(value)

    def to_dict(self) -> dict:
        """Export context as dictionary."""
        return {"run_id": self.run_id, **self.data}


@dataclass
class RetryConfig:
    """Configuration for tiered retry logic.

    Implements hi_moe-a4w: Tiered retry with escalation.
    """
    fleet_retries: int = 2      # Fleet: 2-3 retries, same specialist with error context
    dispatcher_retries: int = 1  # Dispatcher: 1-2 retries, try different specialist
    architect_retries: int = 1   # Architect: 1 retry, revise plan with failure summary
    top_level_retries: int = 1   # Top-level: 1 retry, give up and log everything


@dataclass
class RunResult:
    """Result of a complete run through the hierarchy."""
    run_id: str
    status: RunStatus
    problem: dict
    outcome: Outcome | None = None
    code: str | None = None
    validation: dict | None = None
    error: str | None = None
    context: TaskContext | None = None
    elapsed_ms: float = 0
    retries_used: dict = field(default_factory=dict)


class Runner:
    """Orchestrates the full Hi-MoE tier hierarchy.

    The Runner is the control loop that:
    1. Receives problems
    2. Orchestrates tier execution
    3. Manages state via TaskContext
    4. Handles retries and re-planning
    5. Optionally logs trajectories for training data
    """

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        retry_config: RetryConfig | None = None,
        log_dir: Path | str | None = None,
        code_runner: Callable[[str, list], dict] | None = None,
    ):
        """Initialize the Runner.

        Args:
            llm: LLM client for tier inference
            retry_config: Configuration for retry logic
            log_dir: Directory for trajectory logs (None = no logging)
            code_runner: Optional code execution function
        """
        self.llm = llm
        self.retry_config = retry_config or RetryConfig()
        self.log_dir = Path(log_dir) if log_dir else None
        self.code_runner = code_runner

        # Initialize tiers
        self.fleet = SpecializedFleet(llm)
        self.dispatcher = RoutingDispatcher(self.fleet, llm)
        self.architect = AbstractArchitect(self.dispatcher, llm)
        self.monitor = ProgressMonitor(self.architect)

        # Current run state
        self._trajectory: list[dict] = []

    async def run(self, problem: dict) -> RunResult:
        """Execute a problem through the full tier hierarchy.

        Args:
            problem: Problem dict with 'id', 'title', 'statement', etc.

        Returns:
            RunResult with status, outcome, and any validation results
        """
        run_id = f"run-{problem.get('id', 'unknown')}-{uuid.uuid4().hex[:8]}"
        context = TaskContext(run_id=run_id)
        self._trajectory = []

        start_time = time.monotonic()

        # Store problem in context
        context.set("problem", problem)
        context.set("status", RunStatus.RUNNING.value)

        logger.info(f"[Runner] Starting run {run_id}: {problem.get('title', 'Unknown')}")

        try:
            # Execute with top-level retry
            outcome = await self._execute_with_retry(problem, context)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Extract code from outcome
            code = None
            if outcome and outcome.result:
                code = outcome.result.get("code")

            # Run code validation if we have a code runner and code
            validation = None
            if code and self.code_runner and "test_cases" in problem:
                try:
                    validation = self.code_runner(code, problem["test_cases"])
                    context.set("validation", validation)
                except Exception as e:
                    logger.error(f"[Runner] Code validation failed: {e}")
                    validation = {"error": str(e), "passed": False}

            # Determine final status
            if outcome and outcome.status == TaskStatus.COMPLETED:
                if validation:
                    status = RunStatus.COMPLETED if validation.get("passed") else RunStatus.FAILED
                else:
                    status = RunStatus.COMPLETED
            else:
                status = RunStatus.FAILED

            context.set("status", status.value)

            result = RunResult(
                run_id=run_id,
                status=status,
                problem=problem,
                outcome=outcome,
                code=code,
                validation=validation,
                context=context,
                elapsed_ms=elapsed_ms,
                retries_used=context.get("retries_used", {}),
            )

        except Exception as e:
            logger.error(f"[Runner] Run failed with exception: {e}")
            elapsed_ms = (time.monotonic() - start_time) * 1000
            context.set("status", RunStatus.FAILED.value)
            context.set("error", str(e))

            result = RunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                problem=problem,
                error=str(e),
                context=context,
                elapsed_ms=elapsed_ms,
            )

        # Log trajectory if configured
        if self.log_dir:
            self._save_trajectory(run_id, result)

        logger.info(
            f"[Runner] Run {run_id} completed: {result.status.value} "
            f"in {result.elapsed_ms:.0f}ms"
        )

        return result

    async def _execute_with_retry(
        self,
        problem: dict,
        context: TaskContext,
    ) -> Outcome | None:
        """Execute problem with tiered retry logic."""
        retries_used = {"top_level": 0, "architect": 0}
        context.set("retries_used", retries_used)

        last_outcome = None
        last_error = None

        for attempt in range(self.retry_config.top_level_retries + 1):
            if attempt > 0:
                retries_used["top_level"] = attempt
                logger.info(f"[Runner] Top-level retry {attempt}")
                context.set("retry_attempt", attempt)

            try:
                # Create task from problem
                task = self._create_task(problem, context, attempt)

                # Execute through tier stack
                outcome = await self.monitor.execute(task)

                # Log the call
                self._log_tier_call("monitor", task, outcome)

                # Update context with outcome
                context.set("last_outcome", {
                    "status": outcome.status.value,
                    "result": outcome.result,
                    "error": outcome.error,
                })

                if outcome.status == TaskStatus.COMPLETED:
                    return outcome

                last_outcome = outcome
                last_error = outcome.error

                # Check if we should retry
                if outcome.error and "retry" not in outcome.error.lower():
                    # Non-retryable error
                    logger.warning(f"[Runner] Non-retryable error: {outcome.error}")
                    break

            except Exception as e:
                logger.error(f"[Runner] Execution error: {e}")
                last_error = str(e)
                context.append("errors", str(e))

        # All retries exhausted
        if last_outcome:
            return last_outcome

        # Return a failed outcome
        return Outcome(
            task_id=context.run_id,
            status=TaskStatus.FAILED,
            error=f"All retries exhausted. Last error: {last_error}",
        )

    def _create_task(
        self,
        problem: dict,
        context: TaskContext,
        attempt: int,
    ) -> Task:
        """Create a Task from problem definition."""
        task_id = f"{context.run_id}-attempt{attempt}"

        # Build context with retry information
        task_context = {
            "function_name": problem.get("function_name"),
            "function_signature": problem.get("function_signature"),
            "run_id": context.run_id,
            "attempt": attempt,
        }

        # Add previous failure info for retries
        if attempt > 0:
            last_outcome = context.get("last_outcome")
            if last_outcome:
                task_context["previous_failure"] = {
                    "error": last_outcome.get("error"),
                    "status": last_outcome.get("status"),
                }

        constraints = [
            "Write a complete Python function",
        ]
        if problem.get("function_name"):
            constraints.append(f"Function must be named: {problem['function_name']}")
        if problem.get("function_signature"):
            constraints.append(f"Function signature: {problem['function_signature']}")

        return Task(
            task_id=task_id,
            objective=problem.get("statement", problem.get("title", "Solve the problem")),
            context=task_context,
            constraints=constraints,
        )

    def _log_tier_call(self, tier: str, task: Task, outcome: Outcome) -> None:
        """Log a tier call for trajectory tracking."""
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "tier": tier,
            "task_id": task.task_id,
            "objective": task.objective[:100],  # Truncate for logging
            "status": outcome.status.value,
            "error": outcome.error,
            "execution_time_ms": outcome.execution_time_ms,
        }
        self._trajectory.append(entry)

    def _save_trajectory(self, run_id: str, result: RunResult) -> None:
        """Save trajectory to JSONL file."""
        if not self.log_dir:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create trajectory file
        trajectory_file = self.log_dir / f"{run_id}.jsonl"

        # Write trajectory entries
        with open(trajectory_file, "w") as f:
            # Write run metadata
            metadata = {
                "type": "run_metadata",
                "run_id": run_id,
                "problem_id": result.problem.get("id"),
                "problem_title": result.problem.get("title"),
                "status": result.status.value,
                "elapsed_ms": result.elapsed_ms,
                "retries_used": result.retries_used,
            }
            f.write(json.dumps(metadata) + "\n")

            # Write trajectory entries
            for entry in self._trajectory:
                f.write(json.dumps(entry) + "\n")

            # Write final result
            final = {
                "type": "run_result",
                "status": result.status.value,
                "code": result.code[:1000] if result.code else None,  # Truncate
                "validation": result.validation,
                "error": result.error,
            }
            f.write(json.dumps(final) + "\n")

        logger.info(f"[Runner] Saved trajectory to {trajectory_file}")


async def create_runner(
    endpoint: str | None = None,
    mock: bool = False,
    log_dir: str | None = None,
) -> Runner:
    """Factory function to create a Runner with appropriate LLM client.

    Args:
        endpoint: Modal vLLM endpoint URL (ignored if mock=True)
        mock: Use mock LLM for testing
        log_dir: Directory for trajectory logs

    Returns:
        Configured Runner instance
    """
    if mock:
        llm = MockLLMClient()
    else:
        if not endpoint:
            raise ValueError("endpoint required when mock=False")
        llm = LLMClient(endpoint)

    return Runner(
        llm=llm,
        log_dir=log_dir,
    )
