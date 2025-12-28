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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from .tiers import (
    AbstractArchitect,
    ArchitectMemory,
    DispatcherMemory,
    EmbeddingRouter,
    FleetMemory,
    LLMClient,
    MockLLMClient,
    Outcome,
    ProgressMonitor,
    RoutingDispatcher,
    RoutingMode,
    SpecializedFleet,
    Task,
    TaskStatus,
)
from .outcome_schema import FleetResult
from .trajectory_logger import (
    TrajectoryLogger,
    LoggingLLMClient,
    VLLMCallRecord,
    create_logging_client,
)
from .call_db import CallDB
from .insight_extractor import InsightExtractor

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
    5. Optionally logs trajectories for training data (hi_moe-iz9)
    """

    def __init__(
        self,
        llm: LLMClient | MockLLMClient,
        retry_config: RetryConfig | None = None,
        log_dir: Path | str | None = None,
        code_runner: Callable[[str, list], dict] | None = None,
        enable_trajectory_logging: bool = True,
        enable_fast_path: bool = True,
        enable_training_db: bool = True,
        enable_memory_persistence: bool = True,
        enable_embedding_routing: bool = False,
        routing_mode: str | RoutingMode = "winner_take_all",
        enable_adapters: bool = True,  # Disable for A/B testing (hi_moe-e1v)
    ):
        """Initialize the Runner.

        Args:
            llm: LLM client for tier inference
            retry_config: Configuration for retry logic
            log_dir: Directory for trajectory logs (None = default ./runs)
            code_runner: Optional code execution function
            enable_trajectory_logging: Enable detailed vLLM call logging (hi_moe-iz9)
            enable_fast_path: Enable tier-skip for simple problems (hi_moe-00z)
            enable_training_db: Enable SQLite training data logging (hi_moe-828)
            enable_embedding_routing: Enable semantic embedding routing (hi_moe-awf)
            enable_memory_persistence: Enable persistent agent memory (hi_moe-ycg)
            routing_mode: Routing mode for specialist selection (hi_moe-zrn):
                - "winner_take_all": Always pick highest-scoring specialist (default)
                - "probabilistic": Sample proportionally to similarity scores
                - "blended": Return blend weights for LoRA composition
            enable_adapters: Enable LoRA adapters in Fleet (hi_moe-e1v)
        """
        # Parse routing mode if string
        if isinstance(routing_mode, str):
            routing_mode = RoutingMode(routing_mode)
        self.routing_mode = routing_mode
        self.retry_config = retry_config or RetryConfig()
        self.log_dir = Path(log_dir) if log_dir else Path("./runs")
        self.code_runner = code_runner
        self.enable_trajectory_logging = enable_trajectory_logging
        self.enable_fast_path = enable_fast_path
        self.enable_training_db = enable_training_db
        self.enable_memory_persistence = enable_memory_persistence

        # Initialize training data DB (hi_moe-828)
        self.call_db: CallDB | None = None
        self.insight_extractor: InsightExtractor | None = None
        if enable_training_db:
            self.call_db = CallDB(self.log_dir / "hi_moe.db")
            self.insight_extractor = InsightExtractor(self.call_db)
            logger.info(f"[Runner] Training DB enabled at {self.log_dir / 'hi_moe.db'}")

        # Initialize persistent agent memories (hi_moe-ycg)
        memory_dir = self.log_dir / "memory"
        if enable_memory_persistence:
            self.dispatcher_memory = DispatcherMemory.load(str(memory_dir / "dispatcher.json"))
            self.fleet_memory = FleetMemory.load(str(memory_dir / "fleet.json"))
            logger.info(f"[Runner] Memory persistence enabled at {memory_dir}")
        else:
            self.dispatcher_memory = DispatcherMemory()
            self.fleet_memory = FleetMemory()

        # Set up trajectory logging if enabled
        self.trajectory_logger: TrajectoryLogger | None = None
        if enable_trajectory_logging:
            self.trajectory_logger = TrajectoryLogger(self.log_dir)
            # Wrap LLM client to log all calls
            self.llm = LoggingLLMClient(llm, self.trajectory_logger)
            logger.info(f"[Runner] Trajectory logging enabled, writing to {self.log_dir}")
        else:
            self.llm = llm

        # Initialize tiers with (potentially wrapped) LLM and trajectory logger (hi_moe-r8q)
        # Wire CodeRunner to Fleet for self-healing (hi_moe-ld8)
        # Wire persistent memories (hi_moe-ycg)
        self.fleet = SpecializedFleet(
            self.llm,
            trajectory_logger=self.trajectory_logger,
            code_runner=self.code_runner,
            memory=self.fleet_memory,
            enable_adapters=enable_adapters,  # Toggle adapters for A/B testing (hi_moe-e1v)
        )
        # Initialize embedding router if enabled (hi_moe-awf)
        self.embedding_router = None
        if enable_embedding_routing:
            try:
                self.embedding_router = EmbeddingRouter()
                logger.info("[Runner] Embedding routing enabled (hi_moe-awf)")
            except ImportError as e:
                logger.warning(f"[Runner] Embedding routing unavailable: {e}")

        self.dispatcher = RoutingDispatcher(
            self.fleet,
            self.llm,
            trajectory_logger=self.trajectory_logger,
            call_db=self.call_db,  # Wire call_db for routing decision logging (hi_moe-ehx)
            memory=self.dispatcher_memory,
            embedding_router=self.embedding_router,  # Wire embedding router (hi_moe-awf)
            routing_mode=self.routing_mode,  # Wire routing mode (hi_moe-zrn)
        )
        if self.routing_mode != RoutingMode.WINNER_TAKE_ALL:
            logger.info(f"[Runner] Weighted routing enabled: {self.routing_mode.value} (hi_moe-zrn)")
        self.architect = AbstractArchitect(self.dispatcher, self.llm, trajectory_logger=self.trajectory_logger)
        self.monitor = ProgressMonitor(self.architect)

        # Current run state (legacy, for backward compatibility)
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

        # Reset Architect memory for fresh run (hi_moe-gdf)
        # Memory persists across retries within a run but resets between different problems
        self.architect.memory = ArchitectMemory()

        start_time = time.monotonic()

        # Start trajectory logging for this run (hi_moe-iz9)
        if self.trajectory_logger:
            self.trajectory_logger.start_run(run_id, {
                "problem_id": problem.get("id"),
                "problem_title": problem.get("title"),
            })
            # Set context for LLM call logging
            if isinstance(self.llm, LoggingLLMClient):
                self.llm.set_context(task_id=run_id)

        # Log run start to training DB (hi_moe-828)
        if self.call_db:
            self.call_db.start_run(
                run_id=run_id,
                problem_id=problem.get("id", "unknown"),
                metadata={"title": problem.get("title"), "difficulty": problem.get("difficulty")},
            )

        # Store problem in context
        context.set("problem", problem)
        context.set("status", RunStatus.RUNNING.value)

        logger.info(f"[Runner] Starting run {run_id}: {problem.get('title', 'Unknown')}")

        try:
            # Execute with top-level retry
            outcome = await self._execute_with_retry(problem, context)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            # Extract code from outcome (hi_moe-ld8, hi_moe-qwo)
            # Handle both FleetResult objects and legacy dict results
            code = None
            if outcome and outcome.result:
                if isinstance(outcome.result, FleetResult):
                    code = outcome.result.code
                elif isinstance(outcome.result, dict):
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

            # Log validation to training DB (hi_moe-828)
            if self.call_db and validation:
                last_call_id = context.get("last_call_id")
                self.call_db.log_validation(
                    call_id=last_call_id or 0,
                    run_id=run_id,
                    problem_id=problem.get("id", "unknown"),
                    passed=validation.get("passed", False),
                    tests_total=validation.get("total", len(problem.get("test_cases", []))),
                    tests_passed=validation.get("passed_count", 0),
                    extracted_code=code,
                    execution_output=validation.get("output"),
                    error_type=validation.get("error_type"),
                    error_message=validation.get("error"),
                )

            # Determine final status
            if outcome and outcome.status == TaskStatus.COMPLETED:
                if validation:
                    status = RunStatus.COMPLETED if validation.get("passed") else RunStatus.FAILED
                else:
                    status = RunStatus.COMPLETED
            else:
                status = RunStatus.FAILED

            context.set("status", status.value)

            # Extract insights before context is discarded (hi_moe-3k7)
            if self.insight_extractor and code:
                try:
                    tests_passed = 0
                    tests_total = 0
                    if validation:
                        tests_passed = validation.get("passed_count", 0)
                        tests_total = validation.get("total", len(problem.get("test_cases", [])))

                    self.insight_extractor.extract_from_run(
                        run_id=run_id,
                        problem_id=problem.get("id", "unknown"),
                        code=code,
                        passed=(status == RunStatus.COMPLETED),
                        tests_passed=tests_passed,
                        tests_total=tests_total,
                        error=validation.get("error") if validation else None,
                        retry_history=context.get("retry_history"),
                        routing_decision=context.get("routing_decision"),
                    )
                except Exception as e:
                    logger.warning(f"[Runner] Insight extraction failed: {e}")

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

        # End trajectory logging (hi_moe-iz9)
        if self.trajectory_logger:
            self.trajectory_logger.end_run({
                "status": result.status.value,
                "elapsed_ms": result.elapsed_ms,
                "validation": result.validation,
                "error": result.error,
            })

        # End run in training DB (hi_moe-828)
        if self.call_db:
            self.call_db.end_run(
                run_id=run_id,
                success=result.status == RunStatus.COMPLETED,
                total_time_ms=int(result.elapsed_ms),
                result_code=result.code,
                error=result.error,
            )

        # Log legacy trajectory if configured (deprecated, use trajectory_logger)
        if self.log_dir and not self.trajectory_logger:
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
        """Execute problem with tiered retry logic and optional fast path (hi_moe-00z)."""
        retries_used = {"top_level": 0, "architect": 0}
        context.set("retries_used", retries_used)

        # Fast path: try direct Fleet call for simple problems
        if self.enable_fast_path and self._is_simple_problem(problem):
            logger.info("[Runner] Simple problem detected, trying fast path")
            outcome = await self._execute_fast_path(problem, context)

            if outcome.status == TaskStatus.COMPLETED:
                logger.info("[Runner] Fast path succeeded")
                return outcome

            # Fast path failed, fall back to full tier stack
            logger.info("[Runner] Fast path failed, falling back to full tier stack")
            context.set("fast_path_failed", True)

        last_outcome = None
        last_error = None
        last_failed_call_id = None  # Track for retry logging (hi_moe-828)

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

                # Capture last_call_id from LoggingLLMClient (hi_moe-35o)
                if hasattr(self.llm, "last_call_id") and self.llm.last_call_id:
                    context.set("last_call_id", self.llm.last_call_id)

                # Update context with outcome
                context.set("last_outcome", {
                    "status": outcome.status.value,
                    "result": outcome.result,
                    "error": outcome.error,
                })

                current_call_id = context.get("last_call_id")

                if outcome.status == TaskStatus.COMPLETED:
                    # Log successful retry for self-healing training (hi_moe-828)
                    if attempt > 0 and self.call_db and last_failed_call_id:
                        self.call_db.log_retry(
                            run_id=context.run_id,
                            problem_id=problem.get("id", "unknown"),
                            attempt_number=attempt + 1,
                            failed_call_id=last_failed_call_id,
                            retry_call_id=current_call_id,
                            error_context=last_error,
                            fix_strategy="top_level_retry",
                            retry_succeeded=True,
                        )
                    return outcome

                last_outcome = outcome
                last_error = outcome.error
                last_failed_call_id = current_call_id

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

        # Build context with retry information (hi_moe-ld8)
        # Include test_cases for Fleet self-healing validation
        task_context = {
            "function_name": problem.get("function_name"),
            "function_signature": problem.get("function_signature"),
            "run_id": context.run_id,
            "attempt": attempt,
            "test_cases": problem.get("test_cases"),  # For Fleet self-healing
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

    def _is_simple_problem(self, problem: dict) -> bool:
        """Classify problem as simple (fast path) vs complex (full tier stack).

        Simple problems (hi_moe-00z):
        - Have function_name and function_signature defined
        - Statement is under 1000 chars
        - No explicit multi-step indicators

        Returns:
            True if problem should use fast path (direct Fleet call)
        """
        # Must have function metadata
        if not problem.get("function_name") or not problem.get("function_signature"):
            return False

        statement = problem.get("statement", "")

        # Very long statements may need planning
        if len(statement) > 1000:
            return False

        # Multi-step indicators suggest full planning
        multi_step_indicators = [
            "step 1",
            "step 2",
            "first,",
            "second,",
            "then,",
            "finally,",
            "multiple",
            "several",
            "phase",
        ]
        statement_lower = statement.lower()
        for indicator in multi_step_indicators:
            if indicator in statement_lower:
                return False

        return True

    async def _execute_fast_path(
        self,
        problem: dict,
        context: TaskContext,
    ) -> Outcome:
        """Execute problem via direct Fleet call, skipping Architect/Dispatcher.

        This is the fast path for simple coding problems (hi_moe-00z).
        Saves ~3 LLM calls and ~150s latency.
        """
        logger.info(f"[Runner] Fast path: skipping Architect/Dispatcher for simple problem")

        # Create task for Fleet
        task = self._create_task(problem, context, attempt=0)

        # Execute directly via Fleet with python specialist
        try:
            outcome = await self.fleet.execute(task, specialist="python")

            self._log_tier_call("fleet_fast", task, outcome)

            # Capture last_call_id from LoggingLLMClient (hi_moe-35o)
            if hasattr(self.llm, "last_call_id") and self.llm.last_call_id:
                context.set("last_call_id", self.llm.last_call_id)

            context.set("last_outcome", {
                "status": outcome.status.value,
                "result": outcome.result,
                "error": outcome.error,
            })
            context.set("fast_path", True)

            return outcome

        except Exception as e:
            logger.error(f"[Runner] Fast path failed: {e}")
            # Return failed outcome (will trigger fallback to full tier stack)
            return Outcome(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=f"Fast path failed: {e}",
            )

    def _log_tier_call(self, tier: str, task: Task, outcome: Outcome) -> None:
        """Log a tier call for trajectory tracking."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
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
    enable_trajectory_logging: bool = True,
    enable_fast_path: bool = True,
) -> Runner:
    """Factory function to create a Runner with appropriate LLM client.

    Args:
        endpoint: Modal vLLM endpoint URL (ignored if mock=True)
        mock: Use mock LLM for testing
        log_dir: Directory for trajectory logs
        enable_trajectory_logging: Enable detailed vLLM call logging (hi_moe-iz9)
        enable_fast_path: Enable tier-skip for simple problems (hi_moe-00z)

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
        enable_trajectory_logging=enable_trajectory_logging,
        enable_fast_path=enable_fast_path,
    )
