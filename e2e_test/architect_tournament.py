"""Architect prompt tournament system (hi_moe-fh2).

Run competitions between Architect prompt variants to find the best
management style without hand-tuning. Tracks success rate, token
efficiency, and execution time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .tiers import (
    AbstractArchitect,
    ArchitectMemory,
    LLMClient,
    MockLLMClient,
    RoutingDispatcher,
    SpecializedFleet,
    Task,
    TaskStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant to compete in the tournament."""

    name: str
    system_prompt: str
    plan_prompt_template: str
    style: str  # "terse", "detailed", "aggressive", "conservative"

    def create_plan_prompt(self, task: Task) -> str:
        """Generate planning prompt for this variant."""
        context_str = ""
        if task.context:
            context_str = f"\n\nContext:\n{task.context}"

        return self.plan_prompt_template.format(
            objective=task.objective,
            context=context_str,
            constraints=", ".join(task.constraints) if task.constraints else "None",
        )


# Predefined prompt variants for tournament
PROMPT_VARIANTS = {
    "terse": PromptVariant(
        name="terse",
        system_prompt="You are a strategic planner. Be concise.",
        plan_prompt_template="""Task: {objective}{context}

Create 2-3 step plan. Brief.""",
        style="terse",
    ),
    "detailed": PromptVariant(
        name="detailed",
        system_prompt="""You are a strategic software architect and planner.
Your role is to create comprehensive execution plans that anticipate
potential issues and provide clear guidance to downstream specialists.
Consider edge cases, complexity analysis, and implementation trade-offs.""",
        plan_prompt_template="""## Task Analysis

**Objective:** {objective}
{context}

**Constraints:** {constraints}

## Planning Requirements

Please create a detailed execution plan that includes:
1. Algorithm approach and rationale
2. Key data structures to use
3. Edge cases to consider
4. Expected time/space complexity
5. Implementation steps (2-4 steps)

Focus on providing enough context for specialists to implement correctly.""",
        style="detailed",
    ),
    "aggressive": PromptVariant(
        name="aggressive",
        system_prompt="""You are an aggressive optimizer focused on finding
the fastest, most efficient solution. Prioritize performance and minimal code.
Skip obvious steps - assume specialists are experts.""",
        plan_prompt_template="""TASK: {objective}{context}

REQUIREMENTS:
- Find the optimal O(n) or better solution
- Skip brute force approaches
- Minimize steps, maximize efficiency

DELIVER: 1-2 step plan with optimal algorithm only.""",
        style="aggressive",
    ),
    "conservative": PromptVariant(
        name="conservative",
        system_prompt="""You are a careful, methodical planner who prioritizes
correctness over speed. Start with simple approaches, validate assumptions,
and build incrementally. Better to have working code than optimal code.""",
        plan_prompt_template="""Task: {objective}
{context}

Constraints: {constraints}

Please create a safe, incremental plan:
1. Start with the simplest working approach
2. Verify correctness at each step
3. Only optimize after correctness is established

Include validation steps in your plan.""",
        style="conservative",
    ),
}


@dataclass
class VariantResult:
    """Result from running a single problem with a variant."""

    variant_name: str
    problem_id: str
    success: bool
    execution_time_ms: float
    plan_tokens: int  # Estimated from plan length
    error: str | None = None
    plan: str | None = None


@dataclass
class TournamentResult:
    """Aggregate results from a tournament run."""

    variant_name: str
    total_problems: int
    successes: int
    failures: int
    avg_execution_time_ms: float
    avg_plan_tokens: float
    problems: list[VariantResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0-1)."""
        return self.successes / self.total_problems if self.total_problems > 0 else 0

    @property
    def efficiency_score(self) -> float:
        """Combined score: success rate / log(tokens).

        Higher is better - want high success with low tokens.
        """
        import math
        if self.avg_plan_tokens <= 0:
            return 0
        return self.success_rate / math.log(self.avg_plan_tokens + 1)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "variant": self.variant_name,
            "total": self.total_problems,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_time_ms": round(self.avg_execution_time_ms, 1),
            "avg_tokens": round(self.avg_plan_tokens, 1),
            "efficiency": round(self.efficiency_score, 4),
        }


class VariantArchitect(AbstractArchitect):
    """Architect that uses a specific prompt variant."""

    def __init__(
        self,
        variant: PromptVariant,
        dispatcher: RoutingDispatcher,
        llm: LLMClient,
        trajectory_logger=None,
    ):
        super().__init__(dispatcher, llm, trajectory_logger)
        self.variant = variant
        self._last_plan: str | None = None

    async def _execute_once(self, task: Task, error_context: str = "") -> Any:
        """Execute with variant-specific prompts."""
        logger.info(f"[Architect:{self.variant.name}] Planning task: {task.task_id}")

        # Build prompt with variant template
        memory_context = self.memory.get_memory_prompt()
        plan_prompt = self.variant.create_plan_prompt(task) + error_context + memory_context

        plan = await self.llm.generate(
            [
                {"role": "system", "content": self.variant.system_prompt},
                {"role": "user", "content": plan_prompt},
            ]
        )

        self._last_plan = plan
        logger.info(f"[Architect:{self.variant.name}] Plan created ({len(plan)} chars)")

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
        outcome = await self.dispatcher.execute(subtask)

        # Record failure to memory for future attempts
        if outcome.status != TaskStatus.COMPLETED:
            self.memory.record_failure(
                plan=plan,
                error=outcome.error or "Unknown error",
                task_id=task.task_id,
            )

        return outcome

    def get_last_plan(self) -> str | None:
        """Get the last plan generated (for token counting)."""
        return self._last_plan


class ArchitectTournament:
    """Tournament runner for comparing Architect prompt variants (hi_moe-fh2)."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        variants: list[str] | None = None,
        results_dir: Path | None = None,
    ):
        """Initialize tournament.

        Args:
            llm: LLM client (uses MockLLMClient if None)
            variants: List of variant names to compete (uses all if None)
            results_dir: Directory to save results (optional)
        """
        self.llm = llm or MockLLMClient()
        self.variant_names = variants or list(PROMPT_VARIANTS.keys())
        self.results_dir = results_dir
        self.results: dict[str, TournamentResult] = {}

    async def run_tournament(
        self,
        problems: list[dict],
        code_runner: Callable[[str, list], dict] | None = None,
    ) -> dict[str, TournamentResult]:
        """Run tournament with all variants against problem set.

        Args:
            problems: List of problem dicts with 'id', 'statement', 'test_cases'
            code_runner: Optional function to validate code

        Returns:
            Dict mapping variant names to their results
        """
        logger.info(f"Starting tournament with {len(self.variant_names)} variants, {len(problems)} problems")

        for variant_name in self.variant_names:
            variant = PROMPT_VARIANTS[variant_name]
            logger.info(f"\n{'='*50}")
            logger.info(f"Running variant: {variant_name} ({variant.style})")
            logger.info(f"{'='*50}")

            result = await self._run_variant(variant, problems, code_runner)
            self.results[variant_name] = result

            logger.info(f"Variant {variant_name}: {result.success_rate:.1%} success, "
                       f"{result.avg_execution_time_ms:.0f}ms avg, "
                       f"{result.avg_plan_tokens:.0f} tokens avg")

        # Log final comparison
        self._log_comparison()

        # Save results if directory provided
        if self.results_dir:
            self._save_results()

        return self.results

    async def _run_variant(
        self,
        variant: PromptVariant,
        problems: list[dict],
        code_runner: Callable[[str, list], dict] | None,
    ) -> TournamentResult:
        """Run a single variant against all problems."""
        # Create tier stack for this variant
        fleet = SpecializedFleet(self.llm, code_runner=code_runner)
        dispatcher = RoutingDispatcher(fleet, self.llm)
        architect = VariantArchitect(variant, dispatcher, self.llm)

        problem_results = []
        total_time = 0
        total_tokens = 0
        successes = 0

        for problem in problems:
            problem_id = problem.get("id", "unknown")
            logger.info(f"  Problem: {problem_id}")

            # Create task
            task = Task(
                task_id=f"tournament-{variant.name}-{problem_id}",
                objective=problem["statement"],
                context={
                    "test_cases": problem.get("test_cases", []),
                    "function_name": problem.get("function_name"),
                },
            )

            # Execute and time
            start = time.monotonic()
            try:
                outcome = await architect.execute(task)
                elapsed_ms = (time.monotonic() - start) * 1000

                # Get plan token estimate
                plan = architect.get_last_plan() or ""
                plan_tokens = len(plan.split())  # Rough word-based estimate

                success = outcome.status == TaskStatus.COMPLETED
                if success:
                    successes += 1

                result = VariantResult(
                    variant_name=variant.name,
                    problem_id=problem_id,
                    success=success,
                    execution_time_ms=elapsed_ms,
                    plan_tokens=plan_tokens,
                    error=outcome.error if not success else None,
                    plan=plan,
                )

            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                result = VariantResult(
                    variant_name=variant.name,
                    problem_id=problem_id,
                    success=False,
                    execution_time_ms=elapsed_ms,
                    plan_tokens=0,
                    error=str(e),
                )

            problem_results.append(result)
            total_time += result.execution_time_ms
            total_tokens += result.plan_tokens

            status = "✓" if result.success else "✗"
            logger.info(f"    {status} {result.execution_time_ms:.0f}ms, {result.plan_tokens} tokens")

        n = len(problems)
        return TournamentResult(
            variant_name=variant.name,
            total_problems=n,
            successes=successes,
            failures=n - successes,
            avg_execution_time_ms=total_time / n if n > 0 else 0,
            avg_plan_tokens=total_tokens / n if n > 0 else 0,
            problems=problem_results,
        )

    def _log_comparison(self) -> None:
        """Log comparison of all variants."""
        logger.info("\n" + "="*60)
        logger.info("TOURNAMENT RESULTS")
        logger.info("="*60)

        # Sort by efficiency score
        ranked = sorted(
            self.results.items(),
            key=lambda x: x[1].efficiency_score,
            reverse=True,
        )

        for i, (name, result) in enumerate(ranked, 1):
            medal = ["🥇", "🥈", "🥉", "  "][min(i-1, 3)]
            logger.info(
                f"{medal} #{i} {name:12} | "
                f"Success: {result.success_rate:5.1%} | "
                f"Time: {result.avg_execution_time_ms:6.0f}ms | "
                f"Tokens: {result.avg_plan_tokens:5.0f} | "
                f"Efficiency: {result.efficiency_score:.4f}"
            )

        logger.info("="*60)
        winner = ranked[0][0] if ranked else None
        if winner:
            logger.info(f"WINNER: {winner}")

    def _save_results(self) -> None:
        """Save tournament results to file."""
        if not self.results_dir:
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"tournament_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "variants": [r.to_dict() for r in self.results.values()],
            "winner": self.get_winner(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def get_winner(self) -> str | None:
        """Get the winning variant name."""
        if not self.results:
            return None

        return max(
            self.results.items(),
            key=lambda x: x[1].efficiency_score,
        )[0]

    def get_ranking(self) -> list[tuple[str, TournamentResult]]:
        """Get variants ranked by efficiency score."""
        return sorted(
            self.results.items(),
            key=lambda x: x[1].efficiency_score,
            reverse=True,
        )


async def run_quick_tournament(
    problems: list[dict] | None = None,
    mock: bool = True,
) -> dict[str, TournamentResult]:
    """Quick tournament runner for testing.

    Args:
        problems: Problem set (uses defaults if None)
        mock: Use MockLLMClient (default True)

    Returns:
        Tournament results by variant
    """
    # Default test problems
    if problems is None:
        problems = [
            {
                "id": "two_sum",
                "statement": """Given an array of integers nums and an integer target,
                return indices of the two numbers that add up to target.""",
                "test_cases": [
                    {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
                ],
                "function_name": "twoSum",
            },
            {
                "id": "valid_parens",
                "statement": """Given a string s containing just '(', ')', '{', '}', '[' and ']',
                determine if the input string is valid.""",
                "test_cases": [
                    {"input": {"s": "()"}, "expected": True},
                ],
                "function_name": "isValid",
            },
        ]

    llm = MockLLMClient() if mock else None
    tournament = ArchitectTournament(llm=llm)
    return await tournament.run_tournament(problems)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_quick_tournament())
    print(f"\nWinner: {max(results.items(), key=lambda x: x[1].efficiency_score)[0]}")
