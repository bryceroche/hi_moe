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
from .embedding_router import EmbeddingRouter, HybridEmbeddingRouter

if TYPE_CHECKING:
    from .trajectory_logger import TrajectoryLogger
    from .call_db import CallDB

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


class RoutingMode(Enum):
    """Routing mode for specialist selection (hi_moe-zrn).

    WINNER_TAKE_ALL: Always pick highest-scoring specialist (default)
    PROBABILISTIC: Sample proportionally to similarity scores
    BLENDED: Return blend weights for LoRA composition (future)
    """
    WINNER_TAKE_ALL = "winner_take_all"
    PROBABILISTIC = "probabilistic"
    BLENDED = "blended"


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
class DispatcherMemory:
    """Per-run memory for Dispatcher tier (hi_moe-mz5).

    Tracks routing decisions and specialist outcomes so Dispatcher can
    learn which specialists work for which problem types within a run.

    Supports persistence across sessions (hi_moe-ycg).
    """
    routing_history: list[dict] = field(default_factory=list)
    specialist_outcomes: dict[str, dict] = field(default_factory=dict)  # specialist -> {successes, failures}
    max_memory: int = 10
    persist_path: str | None = None  # Path for persistence (hi_moe-ycg)

    def record_routing(self, task_id: str, specialist: str, problem_type: str) -> None:
        """Record a routing decision."""
        self.routing_history.append({
            "task_id": task_id,
            "specialist": specialist,
            "problem_type": problem_type,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.routing_history) > self.max_memory:
            self.routing_history = self.routing_history[-self.max_memory:]
        self.save()  # Auto-persist (hi_moe-ycg)

    def record_outcome(self, specialist: str, success: bool) -> None:
        """Record specialist execution outcome."""
        if specialist not in self.specialist_outcomes:
            self.specialist_outcomes[specialist] = {"successes": 0, "failures": 0}
        if success:
            self.specialist_outcomes[specialist]["successes"] += 1
        else:
            self.specialist_outcomes[specialist]["failures"] += 1
        self.save()  # Auto-persist (hi_moe-ycg)

    def get_memory_prompt(self) -> str:
        """Generate context for Dispatcher prompts."""
        if not self.specialist_outcomes:
            return ""

        lines = ["\n\n--- Specialist Performance This Run ---"]
        for spec, stats in self.specialist_outcomes.items():
            total = stats["successes"] + stats["failures"]
            rate = stats["successes"] / total if total > 0 else 0
            lines.append(f"  {spec}: {rate:.0%} success ({stats['successes']}/{total})")

        if self.routing_history:
            lines.append("\nRecent routing decisions:")
            for r in self.routing_history[-3:]:
                lines.append(f"  {r['problem_type']} → {r['specialist']}")

        return "\n".join(lines)

    # Persistence methods (hi_moe-ycg)
    def to_dict(self) -> dict:
        """Serialize memory to dict for persistence."""
        return {
            "routing_history": self.routing_history,
            "specialist_outcomes": self.specialist_outcomes,
            "max_memory": self.max_memory,
        }

    @classmethod
    def from_dict(cls, data: dict, persist_path: str | None = None) -> "DispatcherMemory":
        """Load memory from dict."""
        return cls(
            routing_history=data.get("routing_history", []),
            specialist_outcomes=data.get("specialist_outcomes", {}),
            max_memory=data.get("max_memory", 10),
            persist_path=persist_path,
        )

    def save(self) -> None:
        """Save memory to disk."""
        if not self.persist_path:
            return
        import json
        from pathlib import Path
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"[DispatcherMemory] Saved to {self.persist_path}")

    @classmethod
    def load(cls, persist_path: str) -> "DispatcherMemory":
        """Load memory from disk, or create new if not exists."""
        import json
        from pathlib import Path
        path = Path(persist_path)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                logger.info(f"[DispatcherMemory] Loaded from {persist_path}")
                return cls.from_dict(data, persist_path)
            except Exception as e:
                logger.warning(f"[DispatcherMemory] Failed to load {persist_path}: {e}")
        return cls(persist_path=persist_path)


@dataclass
class FleetMemory:
    """Per-specialist memory for Fleet tier (hi_moe-mz5).

    Each specialist maintains its own isolated memory of execution patterns
    and common errors. Memories are keyed by specialist name.

    Supports persistence across sessions (hi_moe-ycg).
    """
    # specialist_name -> list of execution records
    executions: dict[str, list[dict]] = field(default_factory=dict)
    max_per_specialist: int = 5
    persist_path: str | None = None  # Path for persistence (hi_moe-ycg)

    def record_execution(
        self,
        specialist: str,
        task_summary: str,
        success: bool,
        error: str | None = None,
        code_snippet: str | None = None,
    ) -> None:
        """Record a specialist execution."""
        if specialist not in self.executions:
            self.executions[specialist] = []

        self.executions[specialist].append({
            "task": task_summary[:200],
            "success": success,
            "error": error[:150] if error else None,
            "code_pattern": code_snippet[:100] if code_snippet else None,
            "timestamp": datetime.now().isoformat(),
        })

        # Trim to max
        if len(self.executions[specialist]) > self.max_per_specialist:
            self.executions[specialist] = self.executions[specialist][-self.max_per_specialist:]
        self.save()  # Auto-persist (hi_moe-ycg)

    def get_memory_prompt(self, specialist: str) -> str:
        """Generate context for a specific specialist's prompt."""
        if specialist not in self.executions or not self.executions[specialist]:
            return ""

        history = self.executions[specialist]
        failures = [e for e in history if not e["success"]]

        lines = [f"\n\n--- Your ({specialist}) Recent History ---"]

        if failures:
            lines.append("Previous errors to avoid:")
            for f in failures[-2:]:
                lines.append(f"  - {f['error']}")

        successes = [e for e in history if e["success"]]
        if successes:
            lines.append("Successful patterns:")
            for s in successes[-2:]:
                lines.append(f"  - {s['task'][:80]}")

        return "\n".join(lines)

    def get_specialist_stats(self, specialist: str) -> dict:
        """Get success/failure counts for a specialist."""
        if specialist not in self.executions:
            return {"successes": 0, "failures": 0}
        history = self.executions[specialist]
        return {
            "successes": sum(1 for e in history if e["success"]),
            "failures": sum(1 for e in history if not e["success"]),
        }

    # Persistence methods (hi_moe-ycg)
    def to_dict(self) -> dict:
        """Serialize memory to dict for persistence."""
        return {
            "executions": self.executions,
            "max_per_specialist": self.max_per_specialist,
        }

    @classmethod
    def from_dict(cls, data: dict, persist_path: str | None = None) -> "FleetMemory":
        """Load memory from dict."""
        return cls(
            executions=data.get("executions", {}),
            max_per_specialist=data.get("max_per_specialist", 5),
            persist_path=persist_path,
        )

    def save(self) -> None:
        """Save memory to disk."""
        if not self.persist_path:
            return
        import json
        from pathlib import Path
        path = Path(self.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"[FleetMemory] Saved to {self.persist_path}")

    @classmethod
    def load(cls, persist_path: str) -> "FleetMemory":
        """Load memory from disk, or create new if not exists."""
        import json
        from pathlib import Path
        path = Path(persist_path)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                logger.info(f"[FleetMemory] Loaded from {persist_path}")
                return cls.from_dict(data, persist_path)
            except Exception as e:
                logger.warning(f"[FleetMemory] Failed to load {persist_path}: {e}")
        return cls(persist_path=persist_path)


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
        max_tokens: int = 2048,
        adapter: str | None = None,
    ) -> str:
        """Generate completion from LLM with optional adapter.

        Includes mitigations for Modal 303 redirect stalls (hi_moe-56q):
        - Hard 10-minute total timeout via asyncio
        - Reduced max_redirects to prevent infinite polling loops
        - Per-request timeout of 5 minutes
        """
        import asyncio
        from openai import AsyncOpenAI
        import httpx

        # Configure httpx with redirect limits (hi_moe-56q)
        # Modal returns 303 for async polling - limit to prevent infinite loops
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),  # 5 min per-request timeout
            max_redirects=10,  # Reduced from default 20
        )
        client = AsyncOpenAI(
            base_url=f"{self.endpoint}/v1",
            api_key="not-needed",
            http_client=http_client,
        )

        # Use adapter if specified, otherwise base
        model = adapter if adapter else "base"

        # Hard total timeout to prevent 27+ min stalls (hi_moe-56q)
        try:
            async with asyncio.timeout(600):  # 10 min hard limit
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
        except asyncio.TimeoutError:
            logger.error("[LLMClient] Request timed out after 10 minutes (hi_moe-56q)")
            raise TimeoutError("Modal request timed out after 10 minutes - possible 303 redirect stall")


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

        # Optimized prompt (hi_moe-eet): condensed from 256 to ~80 chars
        plan = await self.llm.generate(
            [
                {
                    "role": "system",
                    "content": "Planner. Output 2-3 step algorithm plan for the EXACT problem given.",
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
        embedding_router: EmbeddingRouter | None = None,
        memory: DispatcherMemory | None = None,
        call_db: "CallDB | None" = None,
        routing_mode: RoutingMode = RoutingMode.WINNER_TAKE_ALL,
    ):
        self.fleet = fleet
        self.llm = llm  # Optional: for structured plan generation
        self.trajectory_logger = trajectory_logger
        self.conversation_context = conversation_context  # Multi-turn context (hi_moe-ceg)
        self.learned_router = learned_router  # Optional: learned routing (hi_moe-bh3)
        self.embedding_router = embedding_router  # Optional: embedding routing (hi_moe-awf)
        self.memory = memory or DispatcherMemory()  # Per-run memory (hi_moe-mz5)
        self.call_db = call_db  # Optional: for routing decision logging (hi_moe-ehx)
        self.routing_mode = routing_mode  # Routing mode: winner_take_all, probabilistic, blended (hi_moe-zrn)
        self._specialist_rates_cache: dict | None = None  # Cache for session (hi_moe-fsb)
        self._last_blend_weights: dict[str, float] | None = None  # Cache blend weights for LoRA blending (hi_moe-zrn)

    def _get_adaptive_confidence(self, specialist: str, base_confidence: float) -> float:
        """Calculate adaptive confidence based on historical success rates (hi_moe-fsb).

        Blends base confidence with historical data, weighted by sample size.
        """
        if not self.call_db:
            return base_confidence

        # Cache rates for the session to avoid repeated DB queries
        if self._specialist_rates_cache is None:
            self._specialist_rates_cache = self.call_db.get_all_specialist_rates()

        if specialist not in self._specialist_rates_cache:
            return base_confidence

        history = self._specialist_rates_cache[specialist]
        historical_rate = history["success_rate"]
        sample_size = history["total"]

        # Blend: more weight to history as sample size grows
        # At 10 samples, 50/50 blend. At 50+ samples, 80% history.
        history_weight = min(0.8, sample_size / 20)
        base_weight = 1 - history_weight

        adaptive = (base_confidence * base_weight) + (historical_rate * history_weight)
        logger.debug(
            f"[Dispatcher] Adaptive confidence for {specialist}: "
            f"{base_confidence:.2f} base + {historical_rate:.2f} history "
            f"({sample_size} samples) = {adaptive:.2f}"
        )
        return adaptive

    async def execute(self, task: Task) -> Outcome:
        """Route task to specialist and execute with retry logic (hi_moe-a4w)."""
        logger.info(f"[Dispatcher] Routing task: {task.task_id}")

        # Inject memory context if available (hi_moe-mz5)
        memory_context = self.memory.get_memory_prompt()
        if memory_context:
            logger.info(f"[Dispatcher] Injecting memory context ({len(self.memory.specialist_outcomes)} specialists tracked)")

        # If LLM available, use structured output for planning
        if self.llm:
            try:
                return await self._execute_with_plan(task, memory_context)
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

            # Record routing decision to memory (hi_moe-mz5)
            problem_type = routing_strategy  # Use strategy as problem type proxy
            self.memory.record_routing(task.task_id, specialist, problem_type)
            logger.info(
                f"[Dispatcher] Selected specialist: {specialist} "
                f"(strategy: {routing_strategy}, signals: {routing_signals}, attempt {attempt + 1})"
            )

            # Log routing decision to training DB (hi_moe-ehx)
            run_id = task.context.get("run_id") if task.context else None
            problem_id = task.context.get("problem_id") if task.context else None
            if self.call_db and run_id:
                # Extract keywords from routing signals
                keywords = [s.split(":")[-1] for s in routing_signals if ":" in s]
                alternatives = [s for s in tried_specialists if s != specialist]
                # Calculate adaptive confidence from historical data (hi_moe-fsb)
                base_conf = 0.7 if attempt == 0 else 0.5
                adaptive_conf = self._get_adaptive_confidence(specialist, base_conf)
                self.call_db.log_routing_decision(
                    run_id=run_id,
                    problem_id=problem_id or task.task_id,
                    selected_specialist=specialist,
                    confidence=adaptive_conf,
                    problem_keywords=keywords,
                    alternative_specialists=alternatives,
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
                # Update routing outcome in training DB (hi_moe-ehx)
                if self.call_db and run_id:
                    self.call_db.update_routing_outcome(
                        run_id=run_id,
                        decision_correct=True,
                    )
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
        # Update routing outcome in training DB (hi_moe-ehx)
        if self.call_db and run_id:
            self.call_db.update_routing_outcome(
                run_id=run_id,
                decision_correct=False,
                actual_specialist_needed=None,  # Unknown - all failed
            )
        return final_outcome

    async def _execute_with_plan(self, task: Task, memory_context: str = "") -> Outcome:
        """Execute task using LLM-generated structured plan.

        Implements linear sequence execution (hi_moe-d87).
        """
        # Get structured plan from LLM (with memory context if available)
        # Note: memory_context can be used to influence plan generation in future
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

        # Log routing decision to training DB (hi_moe-ehx)
        run_id = task.context.get("run_id") if task.context else None
        problem_id = task.context.get("problem_id") if task.context else None
        if self.call_db and run_id:
            # First specialist in the plan is the primary routing decision
            primary_specialist = plan.steps[0].specialist if plan.steps else "python"
            all_specialists = list({s.specialist for s in plan.steps})
            # Calculate adaptive confidence from historical data (hi_moe-fsb)
            adaptive_conf = self._get_adaptive_confidence(primary_specialist, 0.8)
            self.call_db.log_routing_decision(
                run_id=run_id,
                problem_id=problem_id or task.task_id,
                selected_specialist=primary_specialist,
                confidence=adaptive_conf,
                problem_keywords=routing_signals,
                alternative_specialists=[s for s in all_specialists if s != primary_specialist],
            )

        # Execute steps sequentially
        all_results = []
        for i, step in enumerate(plan.steps):
            logger.info(
                f"[Dispatcher] Step {i + 1}/{len(plan.steps)}: "
                f"{step.description} -> {step.specialist}"
            )

            # Record routing decision to memory (hi_moe-mz5)
            self.memory.record_routing(
                task_id=f"{task.task_id}-step{i + 1}",
                specialist=step.specialist,
                problem_type=routing_strategy,
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

                # Update routing outcome in training DB (hi_moe-ehx)
                if self.call_db and run_id:
                    self.call_db.update_routing_outcome(
                        run_id=run_id,
                        decision_correct=False,
                        actual_specialist_needed=step.specialist,  # Failed specialist
                    )

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

        # Update routing outcome in training DB (hi_moe-ehx)
        if self.call_db and run_id:
            self.call_db.update_routing_outcome(
                run_id=run_id,
                decision_correct=True,
            )

        return final_outcome

    def _select_specialist(
        self, task: Task, exclude: list[str] | None = None
    ) -> tuple[str, str, list[str]]:
        """Select specialist based on task content.

        Priority order (hi_moe-awf, hi_moe-bh3):
        1. Embedding router (semantic similarity)
        2. Learned router (ML-based)
        3. Heuristic routing (keyword matching)

        Args:
            task: Task to route
            exclude: List of specialists to exclude (already tried)

        Returns:
            Tuple of (specialist, routing_strategy, routing_signals)
            - routing_strategy: "embedding", "learned", "math_first", or "python_direct"
            - routing_signals: Keywords/reasons that influenced the decision
        """
        exclude = exclude or []

        # Try embedding router first (hi_moe-awf)
        if self.embedding_router:
            try:
                specialist, scores, reasoning = self.embedding_router.predict(
                    task.objective, task.context, exclude
                )
                if specialist is not None:
                    routing_signals = [f"embedding:{reasoning}"]
                    for name, score in scores.items():
                        routing_signals.append(f"sim:{name}={score:.2f}")
                    logger.info(f"[Dispatcher] Embedding routing: {specialist} ({reasoning})")
                    return specialist, "embedding", routing_signals
                else:
                    logger.info(f"[Dispatcher] Embedding router deferred: {reasoning}")
            except Exception as e:
                logger.warning(f"[Dispatcher] Embedding router error: {e}")

        # Try learned router second (hi_moe-bh3)
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
        """Record outcome for future routing (hi_moe-ceg, hi_moe-bh3, hi_moe-mz5, hi_moe-awf)."""
        # Extract code from outcome if available (hi_moe-qwo)
        code = None
        if isinstance(outcome.result, FleetResult):
            code = outcome.result.code
        elif outcome.result and isinstance(outcome.result, dict):
            code = outcome.result.get("code")

        # Record to per-run memory (hi_moe-mz5)
        success = outcome.status == TaskStatus.COMPLETED
        self.memory.record_outcome(specialist, success)

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

        # Record to embedding router for prototype learning (hi_moe-awf)
        if self.embedding_router:
            try:
                self.embedding_router.record_outcome(
                    task_id=task.task_id,
                    objective=task.objective,
                    context=task.context,
                    specialist=specialist,
                    success=success,
                )
            except Exception as e:
                logger.warning(f"[Dispatcher] Embedding router record error: {e}")

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
        memory: FleetMemory | None = None,
    ):
        self.llm = llm
        self.trajectory_logger = trajectory_logger
        self.code_runner = code_runner  # Optional code validation (hi_moe-f5d)
        self.memory = memory or FleetMemory()  # Per-specialist memory (hi_moe-mz5)
        self._adapter_cache: dict[str, str | None] = {}

    async def _get_adapter_for_specialist(self, specialist: str) -> str | None:
        """Find best matching adapter for a specialist type."""
        # Adapters re-enabled after training with Qwen3-32B (hi_moe-7l1)
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

            # Record execution to memory (hi_moe-mz5)
            self.memory.record_execution(
                specialist=specialist,
                task_summary=task.objective[:200] if task.objective else "",
                success=outcome.status == TaskStatus.COMPLETED,
                error=outcome.error,
            )

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

        # Inject specialist-specific memory context (hi_moe-mz5)
        memory_context = self.memory.get_memory_prompt(specialist)

        prompt = self._create_execution_prompt(task, specialist) + error_context + memory_context
        system_prompt = self._get_system_prompt(specialist)
        start_time = time.monotonic()

        try:
            response = await self.llm.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2048,  # Reduced for latency; sufficient for code output
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
        # Optimized prompts (hi_moe-eet): ~60% token reduction
        # - /no_think directive handles thinking suppression at API level
        # - Format example in user prompt is sufficient for output structure
        # Original: ~250 chars -> Optimized: ~60 chars per specialist
        prompts = {
            "python": "Python expert. Write working code in ```python``` blocks.",
            "math": "Math expert. Show key steps, then code in ```python``` blocks.",
            "algorithms": "Algorithms expert. Optimal solution in ```python``` blocks.",
            "data_structures": "Data structures expert. Efficient code in ```python``` blocks.",
            "debugging": "Debugging expert. Fix bugs, code in ```python``` blocks.",
            "refactoring": "Refactoring expert. Improved code in ```python``` blocks.",
        }
        return prompts.get(specialist, prompts["python"])

    def _create_execution_prompt(self, task: Task, specialist: str) -> str:
        context = task.context
        plan = context.get("plan", "")
        original = context.get("original_task", task.objective)
        function_name = context.get("function_name", "")
        function_signature = context.get("function_signature", "")

        # Build function requirement section if available (hi_moe-66w)
        func_requirement = ""
        if function_signature:
            func_requirement = f"\n\nRequired function signature: {function_signature}"
        elif function_name:
            func_requirement = f"\n\nFunction must be named: {function_name}"

        # Optimized prompt (hi_moe-eet): removed verbose format example
        # /no_think handles thinking suppression, system prompt specifies code blocks
        return f"""Problem:
{original}

Plan:
{plan}{func_requirement}

Solution (code in ```python``` blocks):"""

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
                    # Additional validation: check it's valid Python syntax (not pseudocode)
                    try:
                        compile(code, "<string>", "exec")
                        logger.info("[Fleet] Extracted code from inside <think> block")
                        return code
                    except SyntaxError:
                        logger.debug("[Fleet] Rejected pseudocode from <think> block (syntax error)")
                        pass  # Continue to other strategies

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
