# End-to-End Test Specification

> Validate the full hi_moe tier stack with a single competitive programming problem.

## Purpose

Before investing in LoRA training (~$60, ~20 GPU hours), validate that:
1. Modal deployment works
2. All 4 tiers communicate correctly
3. CodeRunner executes and validates solutions
4. The base model (no LoRA) can solve simple problems

## Test Problem

Use a simple problem that the base Qwen model should handle:

```python
# test_problems.py
TEST_PROBLEM = {
    "id": "two_sum",
    "title": "Two Sum",
    "difficulty": "easy",
    "statement": """
Given an array of integers nums and an integer target, return indices of the two
numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not
use the same element twice.

You can return the answer in any order.

Example 1:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:
Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:
Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists.
""",
    "test_cases": [
        {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
        {"input": {"nums": [3, 2, 4], "target": 6}, "expected": [1, 2]},
        {"input": {"nums": [3, 3], "target": 6}, "expected": [0, 1]},
        {"input": {"nums": [1, 5, 3, 7, 2], "target": 8}, "expected": [1, 2]},
    ],
    "function_name": "twoSum",
    "function_signature": "def twoSum(nums: list[int], target: int) -> list[int]:",
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        E2E Test Runner                           │
│  ┌───────────┐                                                   │
│  │ Test      │──────────────────────────────────────────────────┐│
│  │ Problem   │                                                  ││
│  └───────────┘                                                  ││
│       │                                                         ││
│       ▼                                                         ││
│  ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌────────┐ ││
│  │ Progress  │───▶│ Architect │───▶│ Dispatcher│───▶│ Fleet  │ ││
│  │ Monitor   │    │           │    │           │    │        │ ││
│  └───────────┘    └───────────┘    └───────────┘    └────────┘ ││
│                                                          │      ││
│                                                          ▼      ││
│                                                    ┌──────────┐ ││
│                                                    │ Solution │ ││
│                                                    │ Code     │ ││
│                                                    └──────────┘ ││
│                                                          │      ││
│       ┌──────────────────────────────────────────────────┘      ││
│       ▼                                                         ││
│  ┌───────────┐    ┌───────────┐                                 ││
│  │ CodeRunner│───▶│ Validate  │─────────────────────────────────┘│
│  │ (Docker)  │    │ Results   │                                  │
│  └───────────┘    └───────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Simplified Tier Implementation

For e2e testing, use simplified tier implementations that can run locally:

```python
# e2e_test/tiers.py
"""Simplified tier implementations for e2e testing."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

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

            logger.info(f"[ProgressMonitor] Task {task.task_id} completed: {outcome.status.value}")
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
        plan = await self.llm.generate([
            {"role": "system", "content": "You are a strategic planner. Break down the task into clear steps."},
            {"role": "user", "content": plan_prompt},
        ])

        logger.info(f"[Architect] Plan created, delegating to Dispatcher")

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

        if any(kw in objective_lower for kw in ["python", "code", "implement", "function"]):
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
            response = await self.llm.generate([
                {"role": "system", "content": self._get_system_prompt(specialist)},
                {"role": "user", "content": prompt},
            ], temperature=0.2)

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
        import re

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
```

## Code Validator

```python
# e2e_test/validator.py
"""Validate generated solutions against test cases."""

import subprocess
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    passed: bool
    total_cases: int
    passed_cases: int
    failed_cases: list[dict]
    error: str | None = None


def validate_solution(
    code: str,
    test_cases: list[dict],
    function_name: str,
    timeout_seconds: int = 5,
) -> ValidationResult:
    """Run solution against test cases."""

    # Create test harness
    harness = create_test_harness(code, test_cases, function_name)

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode != 0:
            return ValidationResult(
                passed=False,
                total_cases=len(test_cases),
                passed_cases=0,
                failed_cases=[],
                error=result.stderr,
            )

        # Parse results
        output = json.loads(result.stdout)
        return ValidationResult(
            passed=output["all_passed"],
            total_cases=output["total"],
            passed_cases=output["passed"],
            failed_cases=output["failures"],
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(
            passed=False,
            total_cases=len(test_cases),
            passed_cases=0,
            failed_cases=[],
            error="Timeout: solution took too long",
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            total_cases=len(test_cases),
            passed_cases=0,
            failed_cases=[],
            error=str(e),
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


def create_test_harness(code: str, test_cases: list[dict], function_name: str) -> str:
    """Create a test harness that runs all test cases."""

    test_cases_json = json.dumps(test_cases)

    return f'''
import json
import sys

# Solution code
{code}

# Test harness
def run_tests():
    test_cases = {test_cases_json}
    results = {{"total": len(test_cases), "passed": 0, "failures": []}}

    for i, tc in enumerate(test_cases):
        try:
            # Call the solution function
            actual = {function_name}(**tc["input"])

            # Check result (handle list order for two_sum type problems)
            expected = tc["expected"]
            if isinstance(expected, list) and isinstance(actual, list):
                passed = sorted(actual) == sorted(expected)
            else:
                passed = actual == expected

            if passed:
                results["passed"] += 1
            else:
                results["failures"].append({{
                    "case": i,
                    "input": tc["input"],
                    "expected": expected,
                    "actual": actual,
                }})
        except Exception as e:
            results["failures"].append({{
                "case": i,
                "input": tc["input"],
                "error": str(e),
            }})

    results["all_passed"] = results["passed"] == results["total"]
    print(json.dumps(results))

if __name__ == "__main__":
    run_tests()
'''
```

## E2E Test Runner

```python
#!/usr/bin/env python3
# e2e_test/run_e2e.py
"""End-to-end test runner for hi_moe."""

import asyncio
import argparse
import logging
import sys
from datetime import datetime

from tiers import (
    Task,
    LLMClient,
    ProgressMonitor,
    AbstractArchitect,
    RoutingDispatcher,
    SpecializedFleet,
)
from validator import validate_solution
from test_problems import TEST_PROBLEM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def run_e2e_test(endpoint: str, problem: dict) -> dict:
    """Run a single problem through the full tier stack."""

    logger.info("=" * 60)
    logger.info(f"E2E Test: {problem['title']}")
    logger.info("=" * 60)

    # Initialize tiers
    llm = LLMClient(endpoint)
    fleet = SpecializedFleet(llm)
    dispatcher = RoutingDispatcher(fleet)
    architect = AbstractArchitect(dispatcher, llm)
    monitor = ProgressMonitor(architect)

    # Create task from problem
    task = Task(
        task_id=f"e2e-{problem['id']}-{datetime.now().strftime('%H%M%S')}",
        objective=problem["statement"],
        context={
            "function_name": problem["function_name"],
            "function_signature": problem["function_signature"],
        },
        constraints=[
            "Write a complete Python function",
            f"Function must be named: {problem['function_name']}",
            f"Function signature: {problem['function_signature']}",
        ],
    )

    # Execute through tiers
    logger.info("Executing through tier stack...")
    start_time = datetime.now()

    outcome = await monitor.execute(task)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Tier execution completed in {elapsed:.2f}s")

    # Check if we got code
    if outcome.status.value != "completed" or not outcome.result:
        logger.error(f"Tier execution failed: {outcome.error}")
        return {
            "success": False,
            "stage": "tier_execution",
            "error": outcome.error,
            "elapsed_seconds": elapsed,
        }

    code = outcome.result.get("code", "")
    logger.info(f"Generated code:\n{code}")

    # Validate solution
    logger.info("Validating solution against test cases...")
    validation = validate_solution(
        code=code,
        test_cases=problem["test_cases"],
        function_name=problem["function_name"],
    )

    # Report results
    result = {
        "success": validation.passed,
        "problem": problem["title"],
        "test_cases": {
            "total": validation.total_cases,
            "passed": validation.passed_cases,
        },
        "elapsed_seconds": elapsed,
        "code": code,
    }

    if validation.passed:
        logger.info(f"✅ All {validation.total_cases} test cases passed!")
    else:
        logger.error(f"❌ Failed: {validation.passed_cases}/{validation.total_cases} passed")
        if validation.error:
            logger.error(f"Error: {validation.error}")
        for failure in validation.failed_cases:
            logger.error(f"  Case {failure['case']}: {failure}")
        result["failures"] = validation.failed_cases
        result["error"] = validation.error

    return result


async def health_check(endpoint: str) -> bool:
    """Check if Modal endpoint is healthy."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{endpoint}/health", timeout=10)
            if resp.status_code == 200:
                logger.info(f"✅ Endpoint healthy: {endpoint}")
                return True
            else:
                logger.error(f"❌ Endpoint returned {resp.status_code}")
                return False
    except Exception as e:
        logger.error(f"❌ Endpoint unreachable: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(description="Run hi_moe e2e tests")
    parser.add_argument(
        "--endpoint",
        default="https://bryce-roche--hi-moe-inference-serve.modal.run",
        help="Modal vLLM endpoint URL",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip endpoint health check",
    )
    args = parser.parse_args()

    # Health check
    if not args.skip_health_check:
        healthy = await health_check(args.endpoint)
        if not healthy:
            logger.error("Endpoint health check failed. Deploy with: modal deploy modal_app/vllm_server.py")
            sys.exit(1)

    # Run e2e test
    result = await run_e2e_test(args.endpoint, TEST_PROBLEM)

    # Summary
    logger.info("=" * 60)
    logger.info("E2E Test Summary")
    logger.info("=" * 60)
    logger.info(f"Problem: {result['problem']}")
    logger.info(f"Success: {result['success']}")
    logger.info(f"Test cases: {result['test_cases']['passed']}/{result['test_cases']['total']}")
    logger.info(f"Time: {result['elapsed_seconds']:.2f}s")

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
```

## Running the Test

### Prerequisites

```bash
# Install dependencies
pip install openai httpx

# Ensure Modal is deployed
modal deploy modal_app/vllm_server.py
```

### Run Test

```bash
# With deployed endpoint
python e2e_test/run_e2e.py --endpoint https://bryce-roche--hi-moe-inference-serve.modal.run

# Skip health check (for local testing)
python e2e_test/run_e2e.py --skip-health-check --endpoint http://localhost:8000
```

### Expected Output

```
2025-01-15 10:30:00 [INFO] ============================================================
2025-01-15 10:30:00 [INFO] E2E Test: Two Sum
2025-01-15 10:30:00 [INFO] ============================================================
2025-01-15 10:30:00 [INFO] ✅ Endpoint healthy: https://bryce-roche--hi-moe-inference-serve.modal.run
2025-01-15 10:30:00 [INFO] Executing through tier stack...
2025-01-15 10:30:00 [INFO] [ProgressMonitor] Starting task: e2e-two_sum-103000
2025-01-15 10:30:00 [INFO] [Architect] Planning task: e2e-two_sum-103000
2025-01-15 10:30:02 [INFO] [Architect] Plan created, delegating to Dispatcher
2025-01-15 10:30:02 [INFO] [Dispatcher] Routing task: e2e-two_sum-103000-impl
2025-01-15 10:30:02 [INFO] [Dispatcher] Selected specialist: python
2025-01-15 10:30:02 [INFO] [Fleet] Executing with python specialist
2025-01-15 10:30:05 [INFO] [ProgressMonitor] Task e2e-two_sum-103000 completed: completed
2025-01-15 10:30:05 [INFO] Tier execution completed in 5.23s
2025-01-15 10:30:05 [INFO] Generated code:
def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

2025-01-15 10:30:05 [INFO] Validating solution against test cases...
2025-01-15 10:30:05 [INFO] ✅ All 4 test cases passed!
2025-01-15 10:30:05 [INFO] ============================================================
2025-01-15 10:30:05 [INFO] E2E Test Summary
2025-01-15 10:30:05 [INFO] ============================================================
2025-01-15 10:30:05 [INFO] Problem: Two Sum
2025-01-15 10:30:05 [INFO] Success: True
2025-01-15 10:30:05 [INFO] Test cases: 4/4
2025-01-15 10:30:05 [INFO] Time: 5.23s
```

## Local Testing (No Modal)

For development without Modal, use a local vLLM server or mock:

```python
# e2e_test/mock_llm.py
"""Mock LLM for local testing."""


class MockLLMClient:
    """Returns pre-defined responses for testing."""

    async def generate(self, messages: list[dict], **kwargs) -> str:
        # Check if this is a planning or execution request
        content = messages[-1]["content"].lower()

        if "plan" in content or "break down" in content:
            return """Plan:
1. Use a hash map to store seen numbers and their indices
2. For each number, check if (target - number) exists in the map
3. Return the indices when found"""

        # Return a working solution
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
```

Run with mock:
```bash
# In run_e2e.py, swap LLMClient for MockLLMClient for local testing
python e2e_test/run_e2e.py --skip-health-check
```

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Health check | Pass | Modal endpoint responding |
| Tier execution | Complete | All 4 tiers execute without error |
| Code generation | Valid Python | Syntactically correct |
| Test cases | 100% pass | All test cases pass |
| Latency | < 30s | End-to-end including LLM calls |

## Next Steps After E2E Passes

1. **Run more problems** - Add 5-10 problems of varying difficulty
2. **Track metrics** - Log success rate, latency, token usage to Beads
3. **Compare base vs LoRA** - Once adapters trained, compare performance
4. **Stress test** - Run competitive programming benchmark (issue `hi_moe-mut`)
