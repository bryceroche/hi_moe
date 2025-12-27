"""Simplified tier implementations for e2e testing."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Union, TYPE_CHECKING

from .dispatcher_schema import (
    DispatcherPlan,
    Step,
    get_dispatcher_plan,
    VALID_SPECIALISTS,
)
from .outcome_schema import (
    FleetResult,
    DispatcherResult,
    StepResult,
    ValidationSummary,
    OutcomeEvaluation,
)
from .learned_router import LearnedRouter, HybridRouter

if TYPE_CHECKING:
    from .trajectory_logger import TrajectoryLogger

logger = logging.getLogger(__name__)


# Retry configuration per tier (hi_moe-a4w)
@dataclass
class RetryConfig:
    """Configuration for tier-level retries."""
    max_retries: int = 2
    include_error_context: bool = True
    escalate_on_failure: bool = True


TIER_RETRY_CONFIG = {
    "fleet": RetryConfig(max_retries=2, include_error_context=True, escalate_on_failure=True),
    "dispatcher": RetryConfig(max_retries=1, include_error_context=True, escalate_on_failure=True),
    "architect": RetryConfig(max_retries=1, include_error_context=True, escalate_on_failure=True),
    "monitor": RetryConfig(max_retries=1, include_error_context=True, escalate_on_failure=False),
}


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Task passed between tiers."""

    task_id: str
    objective: str
    context: dict = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Outcome:
    """Result from task execution."""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    execution_time_ms: float = 0
    metadata: dict = field(default_factory=dict)


# Multi-turn context support (hi_moe-ceg)
@dataclass
class SpecialistStats:
    """Performance statistics for a specialist."""
    successes: int = 0
    failures: int = 0
    total_time_ms: float = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0-1)."""
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5

    @property
    def avg_time_ms(self) -> float:
        """Average execution time."""
        total = self.successes + self.failures
        return self.total_time_ms / total if total > 0 else 0

    def record_success(self, time_ms: float) -> None:
        """Record a successful execution."""
        self.successes += 1
        self.total_time_ms += time_ms

    def record_failure(self, time_ms: float) -> None:
        """Record a failed execution."""
        self.failures += 1
        self.total_time_ms += time_ms


@dataclass
class SolutionRecord:
    """Record of a previous solution."""
    task_id: str
    objective: str
    code: str
    specialist: str
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArchitectMemory:
    """Per-run memory for Architect tier (hi_moe-gdf).

    Tracks failed plans within a single run so Architect can avoid
    repeating the same mistakes. This is agent-private memory, not
    shared with other tiers.
    """
    failed_plans: list[dict] = field(default_factory=list)
    max_memory: int = 5  # Keep last N failures

    def record_failure(self, plan: str, error: str, task_id: str) -> None:
        """Record a failed plan attempt."""
        self.failed_plans.append({
            "task_id": task_id,
            "plan": plan[:500],  # Truncate long plans
            "error": error[:200],  # Truncate long errors
            "timestamp": datetime.now().isoformat(),
        })
        # Trim to max memory
        if len(self.failed_plans) > self.max_memory:
            self.failed_plans = self.failed_plans[-self.max_memory:]

    def get_memory_prompt(self) -> str:
        """Generate context to inject into Architect prompts.

        Returns empty string if no failures recorded.
        """
        if not self.failed_plans:
            return ""

        lines = ["\n\n--- Previous Failed Approaches (avoid these) ---"]
        for i, failure in enumerate(self.failed_plans[-3:], 1):  # Last 3
            lines.append(f"\nAttempt {i}:")
            lines.append(f"  Plan: {failure['plan'][:150]}...")
            lines.append(f"  Failed because: {failure['error']}")
        lines.append("\nPlease try a different approach.")
        return "\n".join(lines)

    def has_failures(self) -> bool:
        """Check if there are recorded failures."""
        return len(self.failed_plans) > 0


@dataclass
class ConversationContext:
    """Multi-turn context that persists across runs (hi_moe-ceg).

    Tracks:
    - Previous solutions for Architect reference
    - Specialist performance for Dispatcher routing
    - Message history for continuity
    """
    session_id: str
    specialist_stats: dict[str, SpecialistStats] = field(default_factory=dict)
    solution_history: list[SolutionRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    max_history: int = 10  # Keep last N solutions

    def record_outcome(
        self,
        task: Task,
        outcome: Outcome,
        specialist: str,
        code: str | None = None,
    ) -> None:
        """Record task outcome for context tracking."""
        # Update specialist stats
        if specialist not in self.specialist_stats:
            self.specialist_stats[specialist] = SpecialistStats()

        stats = self.specialist_stats[specialist]
        if outcome.status == TaskStatus.COMPLETED:
            stats.record_success(outcome.execution_time_ms)
        else:
            stats.record_failure(outcome.execution_time_ms)

        # Store solution if code was generated
        if code:
            record = SolutionRecord(
                task_id=task.task_id,
                objective=task.objective,
                code=code,
                specialist=specialist,
                success=outcome.status == TaskStatus.COMPLETED,
            )
            self.solution_history.append(record)
            # Trim to max history
            if len(self.solution_history) > self.max_history:
                self.solution_history = self.solution_history[-self.max_history:]

    def get_specialist_preference(self) -> list[str]:
        """Get specialists ordered by success rate for routing preference."""
        if not self.specialist_stats:
            return []

        # Sort by success rate, then by total attempts (prefer more experience)
        ranked = sorted(
            self.specialist_stats.items(),
            key=lambda x: (x[1].success_rate, x[1].successes + x[1].failures),
            reverse=True,
        )
        return [name for name, _ in ranked]

    def get_relevant_solutions(self, objective: str, limit: int = 3) -> list[SolutionRecord]:
        """Get previous solutions relevant to the current objective."""
        # Simple keyword matching - could be enhanced with embeddings
        objective_words = set(objective.lower().split())
        scored = []

        for sol in self.solution_history:
            if sol.success:  # Only consider successful solutions
                sol_words = set(sol.objective.lower().split())
                overlap = len(objective_words & sol_words)
                if overlap > 0:
                    scored.append((overlap, sol))

        # Return top matches
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sol for _, sol in scored[:limit]]

    def add_message(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def get_context_summary(self) -> dict:
        """Get summary of context for debugging/logging."""
        return {
            "session_id": self.session_id,
            "total_tasks": sum(
                s.successes + s.failures for s in self.specialist_stats.values()
            ),
            "specialist_stats": {
                name: {"success_rate": f"{s.success_rate:.2%}", "total": s.successes + s.failures}
                for name, s in self.specialist_stats.items()
            },
            "solutions_stored": len(self.solution_history),
            "messages": len(self.messages),
        }


class LLMClient:
    """Client for Modal-hosted vLLM with adapter support."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")
        self._available_adapters: list[str] | None = None

    async def get_available_adapters(self) -> list[str]:
        """Fetch list of available adapters from the server."""
        if self._available_adapters is not None:
            return self._available_adapters

        import httpx
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.endpoint}/v1/models", timeout=10)
                data = resp.json()
                self._available_adapters = [m["id"] for m in data.get("data", [])]
            except Exception:
                self._available_adapters = ["base"]

        return self._available_adapters

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        adapter: str | None = None,
    ) -> str:
        """Generate completion from LLM with optional adapter."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=f"{self.endpoint}/v1",
            api_key="not-needed",
        )

        # Use adapter if specified, otherwise base
        model = adapter if adapter else "base"

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content


class MockLLMClient:
    """Mock LLM for local testing without Modal."""

    async def get_available_adapters(self) -> list[str]:
        """Return mock list of adapters."""
        return ["base", "python-test"]

    async def generate(self, messages: list[dict], adapter: str | None = None, **kwargs) -> str:
        """Return pre-defined responses for testing."""
        # Combine all message content for context
        all_content = " ".join(m.get("content", "").lower() for m in messages)
        last_content = messages[-1]["content"].lower()

        # Check system prompt to determine if this is planning or execution
        system_content = ""
        for m in messages:
            if m.get("role") == "system":
                system_content = m.get("content", "").lower()

        # Dispatcher structured output request (hi_moe-4dy)
        if "task dispatcher" in system_content and "json" in system_content:
            # Return structured plan based on task content
            if "two sum" in all_content or "add up to target" in all_content:
                return '{"steps": [{"description": "Implement two sum using hash map for O(n) lookup", "specialist": "python"}]}'
            if "parentheses" in all_content or "brackets" in all_content:
                return '{"steps": [{"description": "Implement bracket matching using a stack", "specialist": "python"}]}'
            if "math" in all_content or "algorithm" in all_content:
                return '{"steps": [{"description": "Analyze algorithmic approach", "specialist": "math"}, {"description": "Implement solution", "specialist": "python"}]}'
            # Generic single-step plan
            return '{"steps": [{"description": "Implement the solution", "specialist": "python"}]}'

        # Planning request (system prompt mentions planner)
        if "planner" in system_content or "break down" in system_content:
            return """Plan:
1. Use a hash map to store seen numbers and their indices
2. For each number, check if (target - number) exists in the map
3. Return the indices when found"""

        # Execution request (system prompt mentions programming/expert)
        if "programming" in system_content or "expert" in system_content:
            # Check which problem we're solving from the full context
            if "two sum" in all_content or "add up to target" in all_content:
                return '''```python
def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```'''

            if "parentheses" in all_content or "brackets" in all_content:
                return '''```python
def isValid(s: str) -> bool:
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in '({[':
            stack.append(char)
        elif char in ')}]':
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return len(stack) == 0
```'''

        # Generic fallback
        return '''```python
def solution():
    pass
```'''


class ProgressMonitor:
    """Tier 4: Tracks progress and detects issues with top-level retry (hi_moe-a4w)."""

    def __init__(self, architect: "AbstractArchitect"):
        self.architect = architect
        self.task_history: list[Outcome] = []

    async def execute(self, task: Task) -> Outcome:
        """Execute task with progress monitoring and retry logic (hi_moe-a4w)."""
        config = TIER_RETRY_CONFIG["monitor"]
        errors: list[str] = []

        logger.info(f"[ProgressMonitor] Starting task: {task.task_id}")
        start_time = datetime.now()

        for attempt in range(config.max_retries + 1):
            try:
                # Delegate to Architect
                outcome = await self.architect.execute(task)

                # Track outcome
                self.task_history.append(outcome)

                # Calculate execution time
                elapsed = (datetime.now() - start_time).total_seconds() * 1000
                outcome.execution_time_ms = elapsed

                if outcome.status == TaskStatus.COMPLETED:
                    if attempt > 0:
                        logger.info(f"[ProgressMonitor] Succeeded on retry {attempt}")
                    logger.info(
                        f"[ProgressMonitor] Task {task.task_id} completed: {outcome.status.value}"
                    )
                    return outcome

                # Failed but not exception - record error
                errors.append(outcome.error or "Unknown error")
                logger.warning(
                    f"[ProgressMonitor] Attempt {attempt + 1}/{config.max_retries + 1} failed: {outcome.error}"
                )

            except Exception as e:
                errors.append(str(e))
                logger.error(f"[ProgressMonitor] Task {task.task_id} exception: {e}")

        # All retries exhausted - log everything and give up
        elapsed = (datetime.now() - start_time).total_seconds() * 1000
        logger.error(
            f"[ProgressMonitor] Task {task.task_id} failed after {config.max_retries + 1} attempts. "
            f"Total time: {elapsed:.0f}ms. Errors: {errors}"
        )

        final_outcome = Outcome(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"All attempts exhausted. Errors: {errors}",
            execution_time_ms=elapsed,
            metadata={
                "total_attempts": config.max_retries + 1,
                "errors": errors,
            },
        )
        self.task_history.append(final_outcome)
        return final_outcome


class AbstractArchitect:
    """Tier 1: Strategic planning and task decomposition."""

    def __init__(
        self,
        dispatcher: "RoutingDispatcher",
        llm: LLMClient,
        trajectory_logger: "TrajectoryLogger | None" = None,
        memory: ArchitectMemory | None = None,
    ):
        self.dispatcher = dispatcher
        self.llm = llm
        self.trajectory_logger = trajectory_logger
        self.memory = memory or ArchitectMemory()  # Per-run memory (hi_moe-gdf)

    async def execute(self, task: Task) -> Outcome:
        """Create execution plan and delegate to Dispatcher with retry (hi_moe-a4w)."""
        config = TIER_RETRY_CONFIG["architect"]
        errors: list[str] = []

        for attempt in range(config.max_retries + 1):
            # Generate plan (with error context for retries)
            error_context = ""
            if attempt > 0 and errors and config.include_error_context:
                error_context = f"\n\nPrevious attempt failed: {errors[-1]}\nPlease revise the plan to address this issue."

            outcome = await self._execute_once(task, error_context)

            if outcome.status == TaskStatus.COMPLETED:
                if attempt > 0:
                    logger.info(f"[Architect] Succeeded on retry {attempt}")
                return outcome

            # Record error for next retry
            errors.append(outcome.error or "Unknown error")
            logger.warning(f"[Architect] Attempt {attempt + 1}/{config.max_retries + 1} failed: {outcome.error}")

        # All retries exhausted
        logger.error(f"[Architect] All {config.max_retries + 1} attempts failed")
        return Outcome(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Architect failed after {config.max_retries + 1} attempts. Errors: {errors}",
            metadata={"retry_errors": errors},
        )

    async def _execute_once(self, task: Task, error_context: str = "") -> Outcome:
        """Single execution attempt with planning."""
        logger.info(f"[Architect] Planning task: {task.task_id}")

        # Build prompt with memory context (hi_moe-gdf)
        memory_context = self.memory.get_memory_prompt()
        plan_prompt = self._create_plan_prompt(task) + error_context + memory_context

        if memory_context:
            logger.info(f"[Architect] Injecting memory of {len(self.memory.failed_plans)} failed approaches")

        plan = await self.llm.generate(
            [
                {
                    "role": "system",
                    "content": "You are a strategic planner for coding problems. "
                    "CRITICAL: Plan ONLY for the exact problem given. Do not confuse with other problems. "
                    "Focus on the specific function signature and test cases provided. "
                    "Output a brief 2-3 step plan for the algorithm approach.",
                },
                {"role": "user", "content": plan_prompt},
            ]
        )

        # Strip <think>...</think> blocks from plan (QwQ model reasoning)
        plan = self._strip_think_blocks(plan)

        logger.info("[Architect] Plan created, delegating to Dispatcher")

        # Create subtask for dispatcher
        subtask = Task(
            task_id=f"{task.task_id}-impl",
            objective=f"Implement solution for: {task.objective}",
            context={
                "plan": plan,
                "original_task": task.objective,
                **task.context,
            },
            constraints=task.constraints,
        )

        # Log architect decision before execution (hi_moe-r8q)
        if self.trajectory_logger:
            from .trajectory_logger import ArchitectRecord
            architect_record = ArchitectRecord(
                ts=datetime.now(timezone.utc).isoformat(),
                task_id=task.task_id,
                goal=task.objective,
                plan=plan,
                delegation={
                    "task_id": subtask.task_id,
                    "objective": subtask.objective,
                },
                success_criteria=task.constraints if task.constraints else None,
                metadata={"context": task.context},
            )

        # Delegate to Dispatcher
        outcome = await self.dispatcher.execute(subtask)

        # Update and log architect record with outcome (hi_moe-r8q)
        if self.trajectory_logger:
            architect_record.outcome_status = outcome.status.value
            architect_record.outcome_summary = (
                f"Completed with code" if outcome.status == TaskStatus.COMPLETED
                else f"Failed: {outcome.error}"
            )
            self.trajectory_logger.log_architect(architect_record)

        # Record failure to memory for future attempts (hi_moe-gdf)
        if outcome.status != TaskStatus.COMPLETED:
            self.memory.record_failure(
                plan=plan,
                error=outcome.error or "Unknown error",
                task_id=task.task_id,
            )
            logger.info(f"[Architect] Recorded failure to memory (total: {len(self.memory.failed_plans)})")

        return outcome

    def _create_plan_prompt(self, task: Task) -> str:
        context_str = ""
        if task.context:
            context_str = f"\n\nContext:\n{task.context}"

        return f"""Task: {task.objective}
{context_str}

Create a brief execution plan (2-3 steps) to solve this task.
Focus on the algorithm approach and implementation strategy."""

    def _strip_think_blocks(self, text: str) -> str:
        """Strip <think>...</think> blocks from text, keeping content after them."""
        if "<think>" not in text:
            return text

        # Find content after </think> if it exists
        think_end = text.find("</think>")
        if think_end != -1:
            after = text[think_end + 8:].strip()  # len("</think>") = 8
            if after:
                return after

        # Remove the <think> block entirely if unclosed or nothing after
        think_start = text.find("<think>")
        if think_start > 0:
            return text[:think_start].strip()

        # Entire text is in <think> block - extract any useful content
        # Look for structured content (numbered steps, bullet points)
        import re
        plan_match = re.search(r"(?:^|\n)((?:\d+\.|[-*])\s+.+(?:\n(?:\d+\.|[-*])\s+.+)*)",
                               text, re.MULTILINE)
        if plan_match:
            return plan_match.group(1).strip()

        # Return empty if we can't extract anything useful
        logger.warning("[Architect] Could not extract useful plan from reasoning-only output")
        return ""


class RoutingDispatcher:
    """Tier 2: Route tasks to appropriate specialists.

    v0.1: Uses structured output via prompt enforcement (hi_moe-4dy).
    Produces linear sequence of steps, executes sequentially.
    Supports multi-turn context for specialist preference (hi_moe-ceg).
    Supports learned routing with heuristic fallback (hi_moe-bh3).
    """

    def __init__(
        self,
        fleet: "SpecializedFleet",
        llm: LLMClient | None = None,
        trajectory_logger: "TrajectoryLogger | None" = None,
        conversation_context: ConversationContext | None = None,
        learned_router: LearnedRouter | None = None,
    ):
        self.fleet = fleet
        self.llm = llm  # Optional: for structured plan generation
        self.trajectory_logger = trajectory_logger
        self.conversation_context = conversation_context  # Multi-turn context (hi_moe-ceg)
        self.learned_router = learned_router  # Optional: learned routing (hi_moe-bh3)

    async def execute(self, task: Task) -> Outcome:
        """Route task to specialist and execute with retry logic (hi_moe-a4w)."""
        logger.info(f"[Dispatcher] Routing task: {task.task_id}")

        # If LLM available, use structured output for planning
        if self.llm:
            try:
                return await self._execute_with_plan(task)
            except ValueError as e:
                logger.warning(f"[Dispatcher] Structured planning failed: {e}")
                logger.info("[Dispatcher] Falling back to heuristic routing")
                # Fall through to heuristic routing

        # Heuristic-based routing with retry logic
        config = TIER_RETRY_CONFIG["dispatcher"]
        tried_specialists: list[str] = []
        errors: list[str] = []

        for attempt in range(config.max_retries + 1):
            # Select specialist (try different one on retry)
            specialist, routing_strategy, routing_signals = self._select_specialist(
                task, exclude=tried_specialists
            )
            tried_specialists.append(specialist)
            logger.info(
                f"[Dispatcher] Selected specialist: {specialist} "
                f"(strategy: {routing_strategy}, signals: {routing_signals}, attempt {attempt + 1})"
            )

            # Log heuristic routing decision (hi_moe-r8q, hi_moe-gr7)
            if self.trajectory_logger:
                from .trajectory_logger import DispatcherRecord
                dispatcher_record = DispatcherRecord(
                    ts=datetime.now(timezone.utc).isoformat(),
                    task_id=task.task_id,
                    task_objective=task.objective,
                    routing_decision="heuristic",
                    specialist=specialist,
                    rationale=f"Heuristic keyword match selected {specialist} (attempt {attempt + 1})",
                    context_summary=str(task.context)[:200] if task.context else None,
                    # Routing strategy fields (hi_moe-gr7)
                    routing_strategy=routing_strategy,
                    routing_signals=routing_signals,
                )

            outcome = await self.fleet.execute(task, specialist)

            # Update and log dispatcher record (hi_moe-r8q)
            if self.trajectory_logger:
                dispatcher_record.outcome_status = outcome.status.value
                dispatcher_record.outcome_error = outcome.error
                self.trajectory_logger.log_dispatcher(dispatcher_record)

            if outcome.status == TaskStatus.COMPLETED:
                if attempt > 0:
                    logger.info(f"[Dispatcher] Succeeded with {specialist} on retry {attempt}")
                # Record outcome to context (hi_moe-ceg)
                self._record_outcome(task, outcome, specialist)
                return outcome

            # Record error for next retry
            errors.append(f"{specialist}: {outcome.error}")
            logger.warning(f"[Dispatcher] Attempt {attempt + 1} with {specialist} failed: {outcome.error}")

        # All retries exhausted
        logger.error(f"[Dispatcher] All {config.max_retries + 1} attempts failed")
        final_outcome = Outcome(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Failed after trying specialists: {tried_specialists}. Errors: {errors}",
            metadata={"tried_specialists": tried_specialists, "errors": errors},
        )
        # Record final failure to context (hi_moe-ceg)
        if tried_specialists:
            self._record_outcome(task, final_outcome, tried_specialists[-1])
        return final_outcome

    async def _execute_with_plan(self, task: Task) -> Outcome:
        """Execute task using LLM-generated structured plan.

        Implements linear sequence execution (hi_moe-d87).
        """
        # Get structured plan from LLM
        plan = await get_dispatcher_plan(
            self.llm,
            objective=task.objective,
            context=task.context,
            max_retries=1,
        )

        logger.info(f"[Dispatcher] Got plan with {len(plan.steps)} steps")

        # Prepare plan steps for logging
        plan_steps = [{"description": s.description, "specialist": s.specialist} for s in plan.steps]

        # Detect routing strategy from plan structure (hi_moe-gr7)
        # If first step is math/analysis, it's math_first; otherwise python_direct
        first_specialist = plan.steps[0].specialist if plan.steps else "python"
        routing_strategy = "math_first" if first_specialist == "math" else "python_direct"
        routing_signals = [f"plan_first_step:{first_specialist}"]
        for step in plan.steps:
            routing_signals.append(f"step:{step.specialist}")

        logger.info(
            f"[Dispatcher] Plan strategy: {routing_strategy} "
            f"(signals: {routing_signals})"
        )

        # Log structured plan routing decision (hi_moe-r8q, hi_moe-gr7)
        if self.trajectory_logger:
            from .trajectory_logger import DispatcherRecord
            dispatcher_record = DispatcherRecord(
                ts=datetime.now(timezone.utc).isoformat(),
                task_id=task.task_id,
                task_objective=task.objective,
                routing_decision="structured_plan",
                plan_steps=plan_steps,
                rationale=f"LLM generated {len(plan.steps)}-step plan",
                context_summary=str(task.context)[:200] if task.context else None,
                # Routing strategy fields (hi_moe-gr7)
                routing_strategy=routing_strategy,
                routing_signals=routing_signals,
            )

        # Execute steps sequentially
        all_results = []
        for i, step in enumerate(plan.steps):
            logger.info(
                f"[Dispatcher] Step {i + 1}/{len(plan.steps)}: "
                f"{step.description} -> {step.specialist}"
            )

            # Create subtask for this step
            step_task = Task(
                task_id=f"{task.task_id}-step{i + 1}",
                objective=step.description,
                context={
                    **task.context,
                    "step_number": i + 1,
                    "total_steps": len(plan.steps),
                    "previous_results": all_results,
                },
                constraints=task.constraints,
            )

            # Execute with designated specialist
            outcome = await self.fleet.execute(step_task, step.specialist)
            all_results.append({
                "step": i + 1,
                "description": step.description,
                "specialist": step.specialist,
                "status": outcome.status.value,
                "result": outcome.result,
            })

            # Record step outcome for learned router (hi_moe-bh3)
            self._record_outcome(step_task, outcome, step.specialist)

            # Stop on failure (re-plan would happen at higher tier)
            if outcome.status == TaskStatus.FAILED:
                logger.warning(f"[Dispatcher] Step {i + 1} failed, stopping execution")
                final_outcome = Outcome(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Step {i + 1} failed: {outcome.error}",
                    result={"completed_steps": all_results},
                    metadata={"plan": plan_steps},
                )

                # Log dispatcher record with failure (hi_moe-r8q)
                if self.trajectory_logger:
                    dispatcher_record.outcome_status = final_outcome.status.value
                    dispatcher_record.outcome_error = final_outcome.error
                    self.trajectory_logger.log_dispatcher(dispatcher_record)

                return final_outcome

        # All steps completed
        logger.info(f"[Dispatcher] All {len(plan.steps)} steps completed")

        # Return final step's result as the outcome
        final_result = all_results[-1] if all_results else None
        final_outcome = Outcome(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            result=final_result.get("result") if final_result else None,
            metadata={
                "plan": plan_steps,
                "all_steps": all_results,
            },
        )

        # Log dispatcher record with success (hi_moe-r8q)
        if self.trajectory_logger:
            dispatcher_record.outcome_status = final_outcome.status.value
            self.trajectory_logger.log_dispatcher(dispatcher_record)

        return final_outcome

    def _select_specialist(
        self, task: Task, exclude: list[str] | None = None
    ) -> tuple[str, str, list[str]]:
        """Select specialist based on task content.

        Uses learned router when available with sufficient confidence,
        otherwise falls back to heuristic routing (hi_moe-bh3).

        Args:
            task: Task to route
            exclude: List of specialists to exclude (already tried)

        Returns:
            Tuple of (specialist, routing_strategy, routing_signals)
            - routing_strategy: "learned", "math_first", or "python_direct"
            - routing_signals: Keywords/reasons that influenced the decision
        """
        exclude = exclude or []

        # Try learned router first (hi_moe-bh3)
        if self.learned_router:
            specialist, scores, reasoning = self.learned_router.predict(
                task.objective, task.context, exclude
            )
            if specialist is not None:
                routing_signals = [f"learned:{reasoning}"]
                for name, score in scores.items():
                    routing_signals.append(f"score:{name}={score:.2f}")
                logger.info(f"[Dispatcher] Learned routing: {specialist} ({reasoning})")
                return specialist, "learned", routing_signals
            else:
                logger.info(f"[Dispatcher] Learned router deferred: {reasoning}")

        # Fall back to heuristic routing
        objective_lower = task.objective.lower()

        # Track routing signals (hi_moe-gr7)
        routing_signals = []
        candidates = []

        # Math/algorithm keywords suggest analysis-first approach
        math_keywords = ["math", "algorithm", "proof", "optimal", "complexity", "theorem"]
        math_matches = [kw for kw in math_keywords if kw in objective_lower]
        if math_matches:
            candidates.append("math")
            routing_signals.extend([f"math:{kw}" for kw in math_matches])

        # Python/code keywords suggest direct implementation
        python_keywords = ["python", "code", "implement", "function", "write", "create"]
        python_matches = [kw for kw in python_keywords if kw in objective_lower]
        if python_matches:
            candidates.append("python")
            routing_signals.extend([f"python:{kw}" for kw in python_matches])

        # Debugging keywords
        debug_keywords = ["debug", "fix", "error", "bug"]
        debug_matches = [kw for kw in debug_keywords if kw in objective_lower]
        if debug_matches:
            candidates.append("debugging")
            routing_signals.extend([f"debug:{kw}" for kw in debug_matches])

        # Refactoring keywords
        refactor_keywords = ["refactor", "clean", "improve"]
        refactor_matches = [kw for kw in refactor_keywords if kw in objective_lower]
        if refactor_matches:
            candidates.append("refactoring")
            routing_signals.extend([f"refactor:{kw}" for kw in refactor_matches])

        # Determine routing strategy (hi_moe-gr7)
        # math_first: math signals detected, suggests analysis before coding
        # python_direct: no math signals, or python signals came first
        if math_matches and (not python_matches or candidates[0] == "math"):
            routing_strategy = "math_first"
        else:
            routing_strategy = "python_direct"

        # Add all valid specialists as fallbacks
        for specialist in VALID_SPECIALISTS:
            if specialist not in candidates:
                candidates.append(specialist)

        # Apply context-based preference (hi_moe-ceg)
        # Reorder candidates based on historical success rate
        if self.conversation_context:
            preferred = self.conversation_context.get_specialist_preference()
            if preferred:
                # Boost specialists with good history to front of candidates
                # but keep keyword-matched candidates first
                keyword_matched = [c for c in candidates if any(
                    s.startswith(f"{c}:") or s.endswith(f":{c}")
                    for s in routing_signals
                ) or c in [candidates[0]] if candidates]

                # Among remaining, prefer by success rate
                remaining = [c for c in candidates if c not in keyword_matched]
                reordered_remaining = sorted(
                    remaining,
                    key=lambda x: preferred.index(x) if x in preferred else len(preferred),
                )
                candidates = keyword_matched + reordered_remaining
                routing_signals.append(f"context:preferred={preferred[:3]}")

        # Return first non-excluded specialist
        for specialist in candidates:
            if specialist not in exclude:
                return specialist, routing_strategy, routing_signals

        # All excluded, return general (shouldn't happen with proper config)
        return "general", routing_strategy, routing_signals

    def _record_outcome(
        self, task: Task, outcome: Outcome, specialist: str
    ) -> None:
        """Record outcome for future routing (hi_moe-ceg, hi_moe-bh3)."""
        # Extract code from outcome if available (hi_moe-qwo)
        code = None
        if isinstance(outcome.result, FleetResult):
            code = outcome.result.code
        elif outcome.result and isinstance(outcome.result, dict):
            code = outcome.result.get("code")

        # Record to conversation context (hi_moe-ceg)
        if self.conversation_context:
            self.conversation_context.record_outcome(
                task=task,
                outcome=outcome,
                specialist=specialist,
                code=code,
            )
            logger.debug(
                f"[Dispatcher] Recorded outcome for {specialist}: "
                f"{self.conversation_context.get_context_summary()}"
            )

        # Record to learned router for online learning (hi_moe-bh3)
        if self.learned_router:
            success = outcome.status == TaskStatus.COMPLETED
            self.learned_router.record_outcome(
                task_id=task.task_id,
                objective=task.objective,
                context=task.context,
                specialist=specialist,
                success=success,
                execution_time_ms=outcome.execution_time_ms,
                error=outcome.error,
            )


class SpecializedFleet:
    """Tier 3: Execute tasks with specialist capabilities.

    Maps specialist types to LoRA adapters when available.
    Supports self-healing with code validation and retry (hi_moe-f5d).
    """

    # Map specialist types to adapter name patterns
    SPECIALIST_TO_ADAPTER = {
        "python": ["python", "code", "programming"],
        "math": ["math", "reasoning", "gsm"],
        "algorithms": ["algorithm", "algo", "contest", "competitive"],
        "data_structures": ["data", "struct", "ds"],
        "debugging": ["debug", "fix", "bug"],
        "refactoring": ["refactor", "clean", "improve"],
    }

    def __init__(
        self,
        llm: LLMClient,
        trajectory_logger: "TrajectoryLogger | None" = None,
        code_runner: "Callable[[str, list], dict] | None" = None,
    ):
        self.llm = llm
        self.trajectory_logger = trajectory_logger
        self.code_runner = code_runner  # Optional code validation (hi_moe-f5d)
        self._adapter_cache: dict[str, str | None] = {}

    async def _get_adapter_for_specialist(self, specialist: str) -> str | None:
        """Find best matching adapter for a specialist type."""
        if specialist in self._adapter_cache:
            return self._adapter_cache[specialist]

        # Get available adapters from server
        available = await self.llm.get_available_adapters()
        available_lower = {a.lower(): a for a in available if a != "base"}

        # Find matching adapter
        patterns = self.SPECIALIST_TO_ADAPTER.get(specialist, [specialist])
        for pattern in patterns:
            for adapter_lower, adapter_name in available_lower.items():
                if pattern in adapter_lower:
                    self._adapter_cache[specialist] = adapter_name
                    logger.info(f"[Fleet] Mapped specialist '{specialist}' -> adapter '{adapter_name}'")
                    return adapter_name

        # No matching adapter found
        self._adapter_cache[specialist] = None
        logger.info(f"[Fleet] No adapter found for specialist '{specialist}', using base model")
        return None

    async def execute(self, task: Task, specialist: str) -> Outcome:
        """Execute task with specified specialist, with retry logic (hi_moe-a4w)."""
        config = TIER_RETRY_CONFIG["fleet"]
        errors: list[str] = []

        for attempt in range(config.max_retries + 1):
            # Build error context for retries
            error_context = ""
            if attempt > 0 and errors and config.include_error_context:
                error_context = f"\n\nPrevious attempt failed with: {errors[-1]}\nPlease fix the issue and try again."

            outcome = await self._execute_once(task, specialist, error_context)

            if outcome.status == TaskStatus.COMPLETED:
                if attempt > 0:
                    logger.info(f"[Fleet] Succeeded on retry {attempt}")
                return outcome

            # Record error for next retry
            errors.append(outcome.error or "Unknown error")
            logger.warning(f"[Fleet] Attempt {attempt + 1}/{config.max_retries + 1} failed: {outcome.error}")

        # All retries exhausted
        logger.error(f"[Fleet] All {config.max_retries + 1} attempts failed")
        return Outcome(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Failed after {config.max_retries + 1} attempts. Errors: {errors}",
            metadata={"specialist": specialist, "retry_errors": errors},
        )

    async def _execute_once(
        self, task: Task, specialist: str, error_context: str = ""
    ) -> Outcome:
        """Single execution attempt with specified specialist."""
        # Find matching adapter for this specialist
        adapter = await self._get_adapter_for_specialist(specialist)
        adapter_info = f" (adapter: {adapter})" if adapter else " (base model)"
        logger.info(f"[Fleet] Executing with {specialist} specialist{adapter_info}")

        prompt = self._create_execution_prompt(task, specialist) + error_context
        system_prompt = self._get_system_prompt(specialist)
        start_time = time.monotonic()

        try:
            response = await self.llm.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4096,  # Server limit; model must fit reasoning + code
                adapter=adapter,
            )

            execution_time_ms = (time.monotonic() - start_time) * 1000

            # Extract code from response
            code = self._extract_code(response)

            # Fail if no code was extracted
            if not code:
                logger.error("[Fleet] No valid code extracted from response")
                return Outcome(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error="Code extraction failed - no valid Python code in response",
                    metadata={"specialist": specialist, "adapter": adapter, "raw_response": response[:500]},
                )

            # Self-healing: validate code if we have a runner and test cases (hi_moe-f5d)
            validation_result = None
            test_cases = task.context.get("test_cases") if task.context else None

            if code and self.code_runner and test_cases:
                try:
                    validation_result = self.code_runner(code, test_cases)
                    logger.info(
                        f"[Fleet] Validation: {validation_result.get('total_passed', 0)}/"
                        f"{validation_result.get('total_passed', 0) + validation_result.get('total_failed', 0)} tests passed"
                    )

                    if not validation_result.get("passed", False):
                        # Build detailed error feedback for retry
                        error_feedback = self._format_validation_error(validation_result)
                        logger.warning(f"[Fleet] Code validation failed: {error_feedback[:200]}...")

                        # Log failed validation attempt (hi_moe-r8q)
                        if self.trajectory_logger:
                            from .trajectory_logger import FleetRecord
                            fleet_record = FleetRecord(
                                ts=datetime.now(timezone.utc).isoformat(),
                                task_id=task.task_id,
                                task_objective=task.objective,
                                specialist=specialist,
                                prompt_used=system_prompt,
                                output_code=code,
                                execution_time_ms=execution_time_ms,
                                status="error",
                                error=f"Validation failed: {error_feedback[:500]}",
                                validation_result=validation_result,
                                metadata={"context": task.context, "adapter": adapter},
                            )
                            self.trajectory_logger.log_fleet(fleet_record)

                        # Build FleetResult for failed validation (hi_moe-qwo)
                        failed_result = FleetResult(
                            code=code,
                            validation=ValidationSummary.from_validation_dict(validation_result),
                            specialist=specialist,
                            adapter=adapter,
                        )
                        return Outcome(
                            task_id=task.task_id,
                            status=TaskStatus.FAILED,
                            result=failed_result,
                            error=error_feedback,
                            metadata={"specialist": specialist, "adapter": adapter},
                        )

                except Exception as e:
                    logger.error(f"[Fleet] Code validation error: {e}")
                    # Continue without validation on error

            # Build structured FleetResult (hi_moe-qwo)
            fleet_result = FleetResult(
                code=code,
                raw_response=response,
                validation=ValidationSummary.from_validation_dict(validation_result),
                specialist=specialist,
                adapter=adapter,
            )

            outcome = Outcome(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=fleet_result,
                metadata={"specialist": specialist, "adapter": adapter},
            )

            # Log fleet execution (hi_moe-r8q)
            if self.trajectory_logger:
                from .trajectory_logger import FleetRecord
                fleet_record = FleetRecord(
                    ts=datetime.now(timezone.utc).isoformat(),
                    task_id=task.task_id,
                    task_objective=task.objective,
                    specialist=specialist,
                    prompt_used=system_prompt,
                    output_code=code,
                    output_raw=response[:2000] if len(response) > 2000 else response,
                    execution_time_ms=execution_time_ms,
                    status="success",
                    metadata={"context": task.context, "adapter": adapter},
                )
                self.trajectory_logger.log_fleet(fleet_record)

            return outcome

        except Exception as e:
            execution_time_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"[Fleet] Execution failed: {e}")

            # Log fleet execution failure (hi_moe-r8q)
            if self.trajectory_logger:
                from .trajectory_logger import FleetRecord
                fleet_record = FleetRecord(
                    ts=datetime.now(timezone.utc).isoformat(),
                    task_id=task.task_id,
                    task_objective=task.objective,
                    specialist=specialist,
                    prompt_used=system_prompt,
                    execution_time_ms=execution_time_ms,
                    status="error",
                    error=str(e),
                )
                self.trajectory_logger.log_fleet(fleet_record)

            return Outcome(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    def _get_system_prompt(self, specialist: str) -> str:
        # Common suffix for all specialists to enforce output format
        format_instruction = (
            "\n\nIMPORTANT FORMAT RULES:"
            "\n- Do NOT use <think> tags. Skip internal reasoning."
            "\n- Output a brief analysis (2-3 sentences max), then the code."
            "\n- Code MUST be in ```python``` blocks."
            "\n- Never output reasoning without code."
        )

        prompts = {
            "python": "You are a Python programming expert. Write clean, efficient, working code.",
            "math": "You are a mathematical reasoning expert. Solve problems step by step with clear calculations, then provide the final answer.",
            "algorithms": "You are an algorithms expert specializing in competitive programming. "
                "Analyze the problem, identify the optimal approach, then provide the solution.",
            "data_structures": "You are a data structures expert. Choose appropriate data structures and implement efficient operations.",
            "debugging": "You are a debugging expert. Identify bugs systematically, explain the root cause, and provide the corrected code.",
            "refactoring": "You are a code refactoring expert. Improve code quality, readability, and maintainability while preserving functionality.",
        }
        base = prompts.get(specialist, prompts["python"])
        return base + format_instruction

    def _create_execution_prompt(self, task: Task, specialist: str) -> str:
        context = task.context
        plan = context.get("plan", "")
        original = context.get("original_task", task.objective)

        return f"""Problem:
{original}

Plan:
{plan}

Write a Python solution. Brief analysis (2-3 sentences), then code in ```python``` blocks.
Do NOT use <think> tags. Output format example:

The approach uses X algorithm with O(n) complexity.

```python
def solution():
    # implementation
```"""

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response.

        Handles QwQ-style <think>...</think> reasoning blocks by looking for code
        after the thinking section, inside the thinking, or within code blocks.
        """
        # Strategy 1: Look for code after </think> tag
        if "<think>" in response:
            think_end = response.find("</think>")
            if think_end != -1:
                after_think = response[think_end + 8:]  # len("</think>") = 8
                # Try ```python blocks in the post-think section
                match = re.search(r"```python\n(.*?)```", after_think, re.DOTALL)
                if match:
                    return match.group(1).strip()
                # Try generic ``` blocks
                match = re.search(r"```\n(.*?)```", after_think, re.DOTALL)
                if match:
                    return match.group(1).strip()

        # Strategy 2: Try ```python blocks anywhere in response (including inside <think>)
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Strategy 3: Try generic ``` blocks
        match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Strategy 4: Look for function definitions inside <think> blocks
        # Sometimes QwQ writes code inline during reasoning
        if "<think>" in response:
            # Extract content inside <think> tag (may be unclosed if token limit hit)
            think_start = response.find("<think>") + 7
            think_end = response.find("</think>")
            if think_end == -1:
                think_content = response[think_start:]  # Unclosed tag
            else:
                think_content = response[think_start:think_end]

            # Look for function definitions inside thinking
            def_match = re.search(r"(def \w+\([^)]*\):.*?)(?=\n\n[A-Z]|\nNow|\nSo|\nOkay|$)",
                                  think_content, re.DOTALL)
            if def_match:
                code = def_match.group(1).strip()
                # Validate it looks like real code (has return or significant logic)
                if "return" in code or code.count("\n") > 2:
                    logger.info("[Fleet] Extracted code from inside <think> block")
                    return code

        # Strategy 5: Response starts with <think> but no code found - fail cleanly
        if response.strip().startswith("<think>"):
            logger.warning("[Fleet] Response is reasoning-only with no extractable code")
            return ""

        # Strategy 6: Raw code without any blocks (starts with def/class/import)
        stripped = response.strip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            return stripped

        # No valid code found
        logger.warning("[Fleet] Could not extract valid Python code from response")
        return ""

    def _format_validation_error(self, validation: dict) -> str:
        """Format validation results into helpful error feedback for retry (hi_moe-f5d).

        Creates detailed feedback about which tests failed and why, so the
        specialist can fix the code on retry.
        """
        lines = ["Code validation failed:"]

        # Summary
        passed = validation.get("total_passed", 0)
        failed = validation.get("total_failed", 0)
        lines.append(f"  Passed: {passed}/{passed + failed} tests")

        # Individual test failures
        test_results = validation.get("test_results", [])
        for result in test_results:
            status = result.get("status", "unknown")
            if status != "passed":
                test_id = result.get("test_id", "unknown")
                lines.append(f"\n  Test {test_id}: {status.upper()}")

                # Show expected vs actual for wrong answers
                if status == "wrong_answer":
                    expected = result.get("expected_output", "")
                    actual = result.get("actual_output", "")
                    lines.append(f"    Expected: {expected[:100]}")
                    lines.append(f"    Got:      {actual[:100]}")

                # Show error message for runtime errors
                error_msg = result.get("error_message")
                if error_msg:
                    lines.append(f"    Error: {error_msg[:200]}")

        lines.append("\nPlease fix the code and try again.")
        return "\n".join(lines)
