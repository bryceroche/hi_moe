"""Simplified tier implementations for e2e testing."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union, TYPE_CHECKING

from .dispatcher_schema import (
    DispatcherPlan,
    Step,
    get_dispatcher_plan,
    VALID_SPECIALISTS,
)

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
    ):
        self.dispatcher = dispatcher
        self.llm = llm
        self.trajectory_logger = trajectory_logger

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

        # Generate plan using LLM
        plan_prompt = self._create_plan_prompt(task) + error_context
        plan = await self.llm.generate(
            [
                {
                    "role": "system",
                    "content": "You are a strategic planner. Break down the task into clear steps.",
                },
                {"role": "user", "content": plan_prompt},
            ]
        )

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
                ts=datetime.utcnow().isoformat(),
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

        return outcome

    def _create_plan_prompt(self, task: Task) -> str:
        context_str = ""
        if task.context:
            context_str = f"\n\nContext:\n{task.context}"

        return f"""Task: {task.objective}
{context_str}

Create a brief execution plan (2-3 steps) to solve this task.
Focus on the algorithm approach and implementation strategy."""


class RoutingDispatcher:
    """Tier 2: Route tasks to appropriate specialists.

    v0.1: Uses structured output via prompt enforcement (hi_moe-4dy).
    Produces linear sequence of steps, executes sequentially.
    """

    def __init__(
        self,
        fleet: "SpecializedFleet",
        llm: LLMClient | None = None,
        trajectory_logger: "TrajectoryLogger | None" = None,
    ):
        self.fleet = fleet
        self.llm = llm  # Optional: for structured plan generation
        self.trajectory_logger = trajectory_logger

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
            specialist = self._select_specialist(task, exclude=tried_specialists)
            tried_specialists.append(specialist)
            logger.info(f"[Dispatcher] Selected specialist: {specialist} (attempt {attempt + 1})")

            # Log heuristic routing decision (hi_moe-r8q)
            if self.trajectory_logger:
                from .trajectory_logger import DispatcherRecord
                dispatcher_record = DispatcherRecord(
                    ts=datetime.utcnow().isoformat(),
                    task_id=task.task_id,
                    task_objective=task.objective,
                    routing_decision="heuristic",
                    specialist=specialist,
                    rationale=f"Heuristic keyword match selected {specialist} (attempt {attempt + 1})",
                    context_summary=str(task.context)[:200] if task.context else None,
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
                return outcome

            # Record error for next retry
            errors.append(f"{specialist}: {outcome.error}")
            logger.warning(f"[Dispatcher] Attempt {attempt + 1} with {specialist} failed: {outcome.error}")

        # All retries exhausted
        logger.error(f"[Dispatcher] All {config.max_retries + 1} attempts failed")
        return Outcome(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=f"Failed after trying specialists: {tried_specialists}. Errors: {errors}",
            metadata={"tried_specialists": tried_specialists, "errors": errors},
        )

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

        # Log structured plan routing decision (hi_moe-r8q)
        if self.trajectory_logger:
            from .trajectory_logger import DispatcherRecord
            dispatcher_record = DispatcherRecord(
                ts=datetime.utcnow().isoformat(),
                task_id=task.task_id,
                task_objective=task.objective,
                routing_decision="structured_plan",
                plan_steps=plan_steps,
                rationale=f"LLM generated {len(plan.steps)}-step plan",
                context_summary=str(task.context)[:200] if task.context else None,
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

    def _select_specialist(self, task: Task, exclude: list[str] | None = None) -> str:
        """Select specialist based on task content (heuristic fallback).

        Args:
            task: Task to route
            exclude: List of specialists to exclude (already tried)
        """
        exclude = exclude or []
        objective_lower = task.objective.lower()

        # Priority order of specialists to try
        candidates = []

        if any(kw in objective_lower for kw in ["python", "code", "implement", "function"]):
            candidates.append("python")
        if any(kw in objective_lower for kw in ["math", "algorithm", "proof"]):
            candidates.append("math")
        if any(kw in objective_lower for kw in ["debug", "fix", "error", "bug"]):
            candidates.append("debugging")
        if any(kw in objective_lower for kw in ["refactor", "clean", "improve"]):
            candidates.append("refactoring")

        # Add all valid specialists as fallbacks
        for specialist in VALID_SPECIALISTS:
            if specialist not in candidates:
                candidates.append(specialist)

        # Return first non-excluded specialist
        for specialist in candidates:
            if specialist not in exclude:
                return specialist

        # All excluded, return general (shouldn't happen with proper config)
        return "general"


class SpecializedFleet:
    """Tier 3: Execute tasks with specialist capabilities.

    Maps specialist types to LoRA adapters when available.
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
    ):
        self.llm = llm
        self.trajectory_logger = trajectory_logger
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
                adapter=adapter,
            )

            execution_time_ms = (time.monotonic() - start_time) * 1000

            # Extract code from response
            code = self._extract_code(response)

            outcome = Outcome(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={"code": code, "raw_response": response},
                metadata={"specialist": specialist, "adapter": adapter},
            )

            # Log fleet execution (hi_moe-r8q)
            if self.trajectory_logger:
                from .trajectory_logger import FleetRecord
                fleet_record = FleetRecord(
                    ts=datetime.utcnow().isoformat(),
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
                    ts=datetime.utcnow().isoformat(),
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
        prompts = {
            "python": "You are a Python programming expert. Write clean, efficient, working code. Only output the code, no explanations.",
            "math": "You are a mathematical reasoning expert. Solve problems step by step with clear calculations, then provide the final answer.",
            "algorithms": "You are an algorithms expert specializing in competitive programming. Analyze time/space complexity and implement optimal solutions.",
            "data_structures": "You are a data structures expert. Choose appropriate data structures and implement efficient operations.",
            "debugging": "You are a debugging expert. Identify bugs systematically, explain the root cause, and provide the corrected code.",
            "refactoring": "You are a code refactoring expert. Improve code quality, readability, and maintainability while preserving functionality.",
        }
        return prompts.get(specialist, prompts["python"])

    def _create_execution_prompt(self, task: Task, specialist: str) -> str:
        context = task.context
        plan = context.get("plan", "")
        original = context.get("original_task", task.objective)

        return f"""Problem:
{original}

Plan:
{plan}

Write a Python solution. Output only the code, wrapped in ```python``` blocks."""

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try ```python blocks
        match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic ``` blocks
        match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: assume entire response is code
        return response.strip()
