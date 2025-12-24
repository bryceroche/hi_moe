"""Simplified tier implementations for e2e testing."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


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
    """Client for Modal-hosted vLLM."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/")

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Generate completion from LLM."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=f"{self.endpoint}/v1",
            api_key="not-needed",
        )

        response = await client.chat.completions.create(
            model="base",  # Use base model for now
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content


class MockLLMClient:
    """Mock LLM for local testing without Modal."""

    async def generate(self, messages: list[dict], **kwargs) -> str:
        """Return pre-defined responses for testing."""
        # Combine all message content for context
        all_content = " ".join(m.get("content", "").lower() for m in messages)
        last_content = messages[-1]["content"].lower()

        # Check system prompt to determine if this is planning or execution
        system_content = ""
        for m in messages:
            if m.get("role") == "system":
                system_content = m.get("content", "").lower()

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
    """Tier 4: Tracks progress and detects issues."""

    def __init__(self, architect: "AbstractArchitect"):
        self.architect = architect
        self.task_history: list[Outcome] = []

    async def execute(self, task: Task) -> Outcome:
        """Execute task with progress monitoring."""
        logger.info(f"[ProgressMonitor] Starting task: {task.task_id}")
        start_time = datetime.now()

        try:
            # Delegate to Architect
            outcome = await self.architect.execute(task)

            # Track outcome
            self.task_history.append(outcome)

            # Calculate execution time
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            outcome.execution_time_ms = elapsed

            logger.info(
                f"[ProgressMonitor] Task {task.task_id} completed: {outcome.status.value}"
            )
            return outcome

        except Exception as e:
            logger.error(f"[ProgressMonitor] Task {task.task_id} failed: {e}")
            return Outcome(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )


class AbstractArchitect:
    """Tier 1: Strategic planning and task decomposition."""

    def __init__(self, dispatcher: "RoutingDispatcher", llm: LLMClient):
        self.dispatcher = dispatcher
        self.llm = llm

    async def execute(self, task: Task) -> Outcome:
        """Create execution plan and delegate to Dispatcher."""
        logger.info(f"[Architect] Planning task: {task.task_id}")

        # Generate plan using LLM
        plan_prompt = self._create_plan_prompt(task)
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

        # Delegate to Dispatcher
        return await self.dispatcher.execute(subtask)

    def _create_plan_prompt(self, task: Task) -> str:
        context_str = ""
        if task.context:
            context_str = f"\n\nContext:\n{task.context}"

        return f"""Task: {task.objective}
{context_str}

Create a brief execution plan (2-3 steps) to solve this task.
Focus on the algorithm approach and implementation strategy."""


class RoutingDispatcher:
    """Tier 2: Route tasks to appropriate specialists."""

    def __init__(self, fleet: "SpecializedFleet"):
        self.fleet = fleet

    async def execute(self, task: Task) -> Outcome:
        """Route task to specialist and execute."""
        logger.info(f"[Dispatcher] Routing task: {task.task_id}")

        # For e2e test, route to python specialist
        specialist = self._select_specialist(task)
        logger.info(f"[Dispatcher] Selected specialist: {specialist}")

        # Delegate to Fleet
        return await self.fleet.execute(task, specialist)

    def _select_specialist(self, task: Task) -> str:
        """Select specialist based on task content."""
        objective_lower = task.objective.lower()

        if any(
            kw in objective_lower for kw in ["python", "code", "implement", "function"]
        ):
            return "python"
        if any(kw in objective_lower for kw in ["math", "algorithm", "proof"]):
            return "math"

        return "python"  # Default


class SpecializedFleet:
    """Tier 3: Execute tasks with specialist capabilities."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def execute(self, task: Task, specialist: str) -> Outcome:
        """Execute task with specified specialist."""
        logger.info(f"[Fleet] Executing with {specialist} specialist")

        prompt = self._create_execution_prompt(task, specialist)

        try:
            response = await self.llm.generate(
                [
                    {"role": "system", "content": self._get_system_prompt(specialist)},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            # Extract code from response
            code = self._extract_code(response)

            return Outcome(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={"code": code, "raw_response": response},
                metadata={"specialist": specialist},
            )

        except Exception as e:
            logger.error(f"[Fleet] Execution failed: {e}")
            return Outcome(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    def _get_system_prompt(self, specialist: str) -> str:
        prompts = {
            "python": "You are a Python programming expert. Write clean, efficient, working code. Only output the code, no explanations.",
            "math": "You are a mathematical reasoning expert. Solve problems step by step, then provide working code.",
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
