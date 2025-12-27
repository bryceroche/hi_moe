"""Competitive programming stress test for hierarchical coordination (hi_moe-mut).

Validates that the tier hierarchy actually coordinates effectively by
running many problems and measuring coordination quality.

Key metrics:
- Success rate by problem difficulty
- Tier utilization patterns
- Time to solution
- Coordination efficiency (was planning worth it?)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .test_problems import (
    TEST_PROBLEM,
    MEDIUM_PROBLEM,
    HARD_PROBLEM_1,
    HARD_PROBLEM_2,
    HARD_PROBLEM_3,
    CODEFORCES_PROBLEMS,
)

logger = logging.getLogger(__name__)


@dataclass
class ProblemResult:
    """Result from solving a single problem."""
    problem_id: str
    difficulty: str
    success: bool
    time_ms: int
    tiers_used: list[str]
    specialist_used: str | None
    plan_generated: bool
    retries: int
    error: str | None = None


@dataclass
class StressTestResult:
    """Aggregate results from stress test."""
    timestamp: str
    total_problems: int
    success_count: int
    success_rate: float
    by_difficulty: dict[str, dict[str, float]]
    avg_time_ms: float
    tier_utilization: dict[str, int]
    specialist_utilization: dict[str, int]
    coordination_stats: dict[str, float]
    problem_results: list[ProblemResult]


# Problem sets by difficulty
PROBLEM_SETS = {
    "easy": [TEST_PROBLEM, MEDIUM_PROBLEM],
    "medium": [HARD_PROBLEM_1, HARD_PROBLEM_2, HARD_PROBLEM_3],
    "hard": CODEFORCES_PROBLEMS,  # Advanced algorithmic problems
}


class StressTest:
    """Runs competitive programming stress test.

    Usage:
        test = StressTest()

        # Run with tier system
        result = await test.run(tier_runner)

        # Compare with baseline
        baseline_result = await test.run(baseline_runner)

        # Analyze
        comparison = test.compare(result, baseline_result)
    """

    def __init__(self, problems: list[dict] | None = None):
        """
        Args:
            problems: Custom problem set (uses default if None)
        """
        self.problems = problems or self._get_default_problems()
        self.results: list[ProblemResult] = []

    def _get_default_problems(self) -> list[dict]:
        """Get default problem set."""
        problems = []
        for difficulty, problem_list in PROBLEM_SETS.items():
            for problem in problem_list:
                problems.append(problem)
        return problems

    async def run(
        self,
        runner,  # Callable that takes problem and returns (success, metadata)
        max_parallel: int = 1,
        timeout_per_problem: float = 120.0,
    ) -> StressTestResult:
        """Run stress test with given runner.

        Args:
            runner: Async function (problem) -> (success, metadata)
            max_parallel: Max concurrent problems
            timeout_per_problem: Timeout in seconds per problem
        """
        logger.info(f"[StressTest] Starting with {len(self.problems)} problems")
        self.results = []

        semaphore = asyncio.Semaphore(max_parallel)

        async def run_one(problem: dict) -> ProblemResult:
            async with semaphore:
                problem_start = time.time()
                try:
                    success, metadata = await asyncio.wait_for(
                        runner(problem),
                        timeout=timeout_per_problem,
                    )
                    time_ms = int((time.time() - problem_start) * 1000)

                    return ProblemResult(
                        problem_id=problem["id"],
                        difficulty=problem.get("difficulty", "unknown"),
                        success=success,
                        time_ms=time_ms,
                        tiers_used=metadata.get("tiers_used", []),
                        specialist_used=metadata.get("specialist"),
                        plan_generated=metadata.get("plan_generated", False),
                        retries=metadata.get("retries", 0),
                        error=metadata.get("error"),
                    )
                except asyncio.TimeoutError:
                    return ProblemResult(
                        problem_id=problem["id"],
                        difficulty=problem.get("difficulty", "unknown"),
                        success=False,
                        time_ms=int(timeout_per_problem * 1000),
                        tiers_used=[],
                        specialist_used=None,
                        plan_generated=False,
                        retries=0,
                        error="Timeout",
                    )
                except Exception as e:
                    return ProblemResult(
                        problem_id=problem["id"],
                        difficulty=problem.get("difficulty", "unknown"),
                        success=False,
                        time_ms=int((time.time() - problem_start) * 1000),
                        tiers_used=[],
                        specialist_used=None,
                        plan_generated=False,
                        retries=0,
                        error=str(e),
                    )

        # Run all problems
        tasks = [run_one(p) for p in self.problems]
        self.results = await asyncio.gather(*tasks)

        # Compute aggregate stats
        return self._compute_result()

    def _compute_result(self) -> StressTestResult:
        """Compute aggregate statistics."""
        if not self.results:
            return StressTestResult(
                timestamp=datetime.now().isoformat(),
                total_problems=0,
                success_count=0,
                success_rate=0.0,
                by_difficulty={},
                avg_time_ms=0.0,
                tier_utilization={},
                specialist_utilization={},
                coordination_stats={},
                problem_results=[],
            )

        # Basic stats
        total = len(self.results)
        successes = sum(1 for r in self.results if r.success)

        # By difficulty
        by_difficulty: dict[str, dict[str, float]] = {}
        for difficulty in set(r.difficulty for r in self.results):
            diff_results = [r for r in self.results if r.difficulty == difficulty]
            diff_successes = sum(1 for r in diff_results if r.success)
            by_difficulty[difficulty] = {
                "total": len(diff_results),
                "success": diff_successes,
                "rate": diff_successes / len(diff_results) if diff_results else 0,
                "avg_time_ms": sum(r.time_ms for r in diff_results) / len(diff_results) if diff_results else 0,
            }

        # Tier utilization
        tier_counts: dict[str, int] = {}
        for r in self.results:
            for tier in r.tiers_used:
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Specialist utilization
        specialist_counts: dict[str, int] = {}
        for r in self.results:
            if r.specialist_used:
                specialist_counts[r.specialist_used] = specialist_counts.get(r.specialist_used, 0) + 1

        # Coordination stats
        planned = [r for r in self.results if r.plan_generated]
        unplanned = [r for r in self.results if not r.plan_generated]

        planned_success = sum(1 for r in planned if r.success) / len(planned) if planned else 0
        unplanned_success = sum(1 for r in unplanned if r.success) / len(unplanned) if unplanned else 0

        coordination_stats = {
            "problems_with_plans": len(planned),
            "planned_success_rate": planned_success,
            "unplanned_success_rate": unplanned_success,
            "planning_benefit": planned_success - unplanned_success,
            "avg_retries": sum(r.retries for r in self.results) / total,
        }

        return StressTestResult(
            timestamp=datetime.now().isoformat(),
            total_problems=total,
            success_count=successes,
            success_rate=successes / total,
            by_difficulty=by_difficulty,
            avg_time_ms=sum(r.time_ms for r in self.results) / total,
            tier_utilization=tier_counts,
            specialist_utilization=specialist_counts,
            coordination_stats=coordination_stats,
            problem_results=self.results,
        )

    def compare(
        self,
        tier_result: StressTestResult,
        baseline_result: StressTestResult,
    ) -> dict:
        """Compare tier result with baseline."""
        return {
            "tier_success_rate": tier_result.success_rate,
            "baseline_success_rate": baseline_result.success_rate,
            "improvement": tier_result.success_rate - baseline_result.success_rate,
            "tier_avg_time_ms": tier_result.avg_time_ms,
            "baseline_avg_time_ms": baseline_result.avg_time_ms,
            "time_overhead": tier_result.avg_time_ms - baseline_result.avg_time_ms,
            "by_difficulty": {
                diff: {
                    "tier": tier_result.by_difficulty.get(diff, {}).get("rate", 0),
                    "baseline": baseline_result.by_difficulty.get(diff, {}).get("rate", 0),
                }
                for diff in set(tier_result.by_difficulty.keys()) | set(baseline_result.by_difficulty.keys())
            },
            "planning_benefit": tier_result.coordination_stats.get("planning_benefit", 0),
        }

    def get_report(self, result: StressTestResult) -> str:
        """Generate human-readable report."""
        lines = [
            "=" * 60,
            "STRESS TEST RESULTS",
            "=" * 60,
            "",
            f"Timestamp: {result.timestamp}",
            f"Total problems: {result.total_problems}",
            f"Success rate: {result.success_rate:.1%} ({result.success_count}/{result.total_problems})",
            f"Average time: {result.avg_time_ms:.0f}ms",
            "",
            "## By Difficulty",
        ]

        for diff, stats in sorted(result.by_difficulty.items()):
            lines.append(
                f"  {diff}: {stats['rate']:.1%} ({int(stats['success'])}/{int(stats['total'])}) "
                f"avg {stats['avg_time_ms']:.0f}ms"
            )

        if result.tier_utilization:
            lines.extend([
                "",
                "## Tier Utilization",
            ])
            for tier, count in sorted(result.tier_utilization.items(), key=lambda x: -x[1]):
                lines.append(f"  {tier}: {count}")

        if result.specialist_utilization:
            lines.extend([
                "",
                "## Specialist Utilization",
            ])
            for spec, count in sorted(result.specialist_utilization.items(), key=lambda x: -x[1]):
                lines.append(f"  {spec}: {count}")

        lines.extend([
            "",
            "## Coordination Stats",
            f"  Problems with plans: {result.coordination_stats.get('problems_with_plans', 0)}",
            f"  Planned success rate: {result.coordination_stats.get('planned_success_rate', 0):.1%}",
            f"  Unplanned success rate: {result.coordination_stats.get('unplanned_success_rate', 0):.1%}",
            f"  Planning benefit: {result.coordination_stats.get('planning_benefit', 0):+.1%}",
            f"  Avg retries: {result.coordination_stats.get('avg_retries', 0):.1f}",
        ])

        # List failed problems
        failed = [r for r in result.problem_results if not r.success]
        if failed:
            lines.extend([
                "",
                "## Failed Problems",
            ])
            for r in failed:
                lines.append(f"  {r.problem_id} ({r.difficulty}): {r.error or 'Unknown error'}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, result: StressTestResult, path: Path) -> None:
        """Save results to file."""
        data = {
            "timestamp": result.timestamp,
            "total_problems": result.total_problems,
            "success_count": result.success_count,
            "success_rate": result.success_rate,
            "by_difficulty": result.by_difficulty,
            "avg_time_ms": result.avg_time_ms,
            "tier_utilization": result.tier_utilization,
            "specialist_utilization": result.specialist_utilization,
            "coordination_stats": result.coordination_stats,
            "problem_results": [
                {
                    "problem_id": r.problem_id,
                    "difficulty": r.difficulty,
                    "success": r.success,
                    "time_ms": r.time_ms,
                    "tiers_used": r.tiers_used,
                    "specialist_used": r.specialist_used,
                    "plan_generated": r.plan_generated,
                    "retries": r.retries,
                    "error": r.error,
                }
                for r in result.problem_results
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[StressTest] Saved results to {path}")


async def demo():
    """Demo stress test with mock runner."""
    import random

    async def mock_runner(problem: dict) -> tuple[bool, dict]:
        """Mock runner that simulates success based on difficulty."""
        await asyncio.sleep(0.1)  # Simulate work

        # Harder problems less likely to succeed
        success_prob = {
            "easy": 0.9,
            "medium": 0.7,
            "hard": 0.4,
        }.get(problem.get("difficulty", "easy"), 0.5)

        success = random.random() < success_prob

        return success, {
            "tiers_used": ["monitor", "architect", "dispatcher", "fleet"] if random.random() > 0.3 else ["fleet"],
            "specialist": random.choice(["python", "algorithms", "math"]),
            "plan_generated": random.random() > 0.3,
            "retries": random.randint(0, 2),
            "error": None if success else "Failed to solve",
        }

    test = StressTest()
    result = await test.run(mock_runner, max_parallel=2)
    print(test.get_report(result))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())
