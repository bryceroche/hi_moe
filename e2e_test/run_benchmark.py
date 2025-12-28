#!/usr/bin/env python3
"""Benchmark runner for hi_moe tier system (hi_moe-5lq).

Runs problems through the tier system and collects comprehensive metrics.
Validates that infrastructure works end-to-end.

Usage:
    python -m e2e_test.run_benchmark                    # Run with mock (quick test)
    python -m e2e_test.run_benchmark --live             # Run with Modal endpoint
    python -m e2e_test.run_benchmark --live --all       # Run all problems
    python -m e2e_test.run_benchmark --report           # Generate report from DB
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .test_problems import ALL_PROBLEMS, TEST_PROBLEM, MEDIUM_PROBLEM
from .runner import Runner, RunStatus, create_runner
from .tiers import LLMClient, MockLLMClient, RoutingMode
from .call_db import CallDB
from .validator import validate_solution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result for a single problem."""
    problem_id: str
    problem_title: str
    difficulty: str
    success: bool
    tests_passed: int
    tests_total: int
    elapsed_ms: float
    tokens_in: int
    tokens_out: int
    error: str | None = None


@dataclass
class BenchmarkSummary:
    """Summary of benchmark run."""
    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    endpoint: str = "mock"
    routing_mode: str = "winner_take_all"

    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.success) / len(self.results)

    def total_tokens(self) -> int:
        return sum(r.tokens_in + r.tokens_out for r in self.results)

    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.elapsed_ms for r in self.results) / len(self.results)

    def by_difficulty(self) -> dict:
        """Group results by difficulty."""
        groups = {}
        for r in self.results:
            if r.difficulty not in groups:
                groups[r.difficulty] = {"passed": 0, "total": 0}
            groups[r.difficulty]["total"] += 1
            if r.success:
                groups[r.difficulty]["passed"] += 1
        return groups

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "endpoint": self.endpoint,
            "routing_mode": self.routing_mode,
            "total_problems": len(self.results),
            "success_rate": self.success_rate(),
            "total_tokens": self.total_tokens(),
            "avg_latency_ms": self.avg_latency_ms(),
            "by_difficulty": self.by_difficulty(),
            "results": [
                {
                    "problem_id": r.problem_id,
                    "title": r.problem_title,
                    "difficulty": r.difficulty,
                    "success": r.success,
                    "tests_passed": r.tests_passed,
                    "tests_total": r.tests_total,
                    "elapsed_ms": r.elapsed_ms,
                    "tokens": r.tokens_in + r.tokens_out,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


async def run_problem(runner: Runner, problem: dict) -> BenchmarkResult:
    """Run a single problem and return result."""
    problem_id = problem.get("id", "unknown")
    title = problem.get("title", problem_id)
    difficulty = problem.get("difficulty", "unknown")

    logger.info(f"Running: {title} ({difficulty})")

    try:
        run_result = await runner.run(problem)

        # Validate if we have test cases
        tests_passed = 0
        tests_total = 0
        validation_error = None

        if run_result.code and "test_cases" in problem and "function_name" in problem:
            val_result = validate_solution(
                code=run_result.code,
                test_cases=problem["test_cases"],
                function_name=problem["function_name"],
            )
            tests_passed = val_result.passed_cases
            tests_total = val_result.total_cases
            if not val_result.passed:
                validation_error = val_result.error or f"Failed {tests_total - tests_passed}/{tests_total} tests"

        success = run_result.status == RunStatus.COMPLETED and tests_passed == tests_total

        # Token tracking is in CallDB, not RunResult - use 0 for now
        tokens_in = getattr(run_result, 'tokens_in', 0) or 0
        tokens_out = getattr(run_result, 'tokens_out', 0) or 0

        return BenchmarkResult(
            problem_id=problem_id,
            problem_title=title,
            difficulty=difficulty,
            success=success,
            tests_passed=tests_passed,
            tests_total=tests_total,
            elapsed_ms=run_result.elapsed_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            error=validation_error or run_result.error,
        )
    except Exception as e:
        logger.error(f"Error running {title}: {e}")
        return BenchmarkResult(
            problem_id=problem_id,
            problem_title=title,
            difficulty=difficulty,
            success=False,
            tests_passed=0,
            tests_total=len(problem.get("test_cases", [])),
            elapsed_ms=0,
            tokens_in=0,
            tokens_out=0,
            error=str(e),
        )


async def run_benchmark(
    problems: list[dict],
    endpoint: str | None = None,
    mock: bool = False,
    routing_mode: RoutingMode = RoutingMode.WINNER_TAKE_ALL,
    log_dir: Path = Path("runs"),
) -> BenchmarkSummary:
    """Run benchmark on a set of problems."""
    summary = BenchmarkSummary(
        endpoint=endpoint or "mock",
        routing_mode=routing_mode.value,
    )

    # Create runner using factory
    runner = await create_runner(
        endpoint=endpoint,
        mock=mock,
        log_dir=str(log_dir),
    )

    print(f"\n{'='*60}")
    print("HI-MOE BENCHMARK")
    print(f"{'='*60}")
    print(f"Problems: {len(problems)}")
    print(f"Endpoint: {summary.endpoint}")
    print(f"Routing:  {summary.routing_mode}")
    print(f"{'='*60}\n")

    for i, problem in enumerate(problems, 1):
        result = await run_problem(runner, problem)
        summary.results.append(result)

        status = "✓" if result.success else "✗"
        tests = f"{result.tests_passed}/{result.tests_total}" if result.tests_total else "-"
        print(f"[{i}/{len(problems)}] {status} {result.problem_title}: {tests} tests, {result.elapsed_ms:.0f}ms")

    return summary


def print_summary(summary: BenchmarkSummary):
    """Print benchmark summary."""
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")

    rate = summary.success_rate() * 100
    print(f"\nOverall: {rate:.1f}% ({sum(1 for r in summary.results if r.success)}/{len(summary.results)} passed)")
    print(f"Tokens:  {summary.total_tokens():,}")
    print(f"Latency: {summary.avg_latency_ms():.0f}ms avg")

    # By difficulty
    by_diff = summary.by_difficulty()
    if by_diff:
        print(f"\n--- By Difficulty ---")
        for diff, stats in sorted(by_diff.items()):
            rate = stats["passed"] / stats["total"] * 100 if stats["total"] else 0
            print(f"  {diff:<10} {rate:.0f}% ({stats['passed']}/{stats['total']})")

    # Failures
    failures = [r for r in summary.results if not r.success]
    if failures:
        print(f"\n--- Failures ({len(failures)}) ---")
        for r in failures[:10]:
            error = (r.error or "unknown")[:50]
            print(f"  {r.problem_title}: {error}")


def generate_report(db_path: Path):
    """Generate report from existing database."""
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    from .token_audit import print_audit
    from .specialist_dashboard import print_dashboard

    print(f"\n{'='*60}")
    print("BENCHMARK REPORT")
    print(f"Database: {db_path}")
    print(f"{'='*60}")

    # Token audit
    print_audit(db_path, show_waste=True, show_cost=True)

    # Specialist dashboard
    print_dashboard(db_path)


def main():
    parser = argparse.ArgumentParser(description="Hi-MoE benchmark runner")
    parser.add_argument("--live", action="store_true", help="Use live Modal endpoint")
    parser.add_argument("--endpoint", type=str, help="Custom endpoint URL")
    parser.add_argument("--all", action="store_true", help="Run all problems")
    parser.add_argument("--easy", action="store_true", help="Run only easy problems")
    parser.add_argument("--n", type=int, default=5, help="Number of problems to run")
    parser.add_argument("--routing", choices=["winner_take_all", "probabilistic", "blended", "ensemble"],
                       default="winner_take_all", help="Routing mode")
    parser.add_argument("--report", action="store_true", help="Generate report from DB")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmarks"), help="Output directory")
    parser.add_argument("--db", type=Path, default=Path("runs/hi_moe.db"), help="Database path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.report:
        generate_report(args.db)
        return

    # Select problems
    if args.all:
        problems = ALL_PROBLEMS
    elif args.easy:
        problems = [p for p in ALL_PROBLEMS if p.get("difficulty") == "easy"][:args.n]
    else:
        problems = ALL_PROBLEMS[:args.n]

    # Determine endpoint
    if args.live:
        endpoint = args.endpoint or "https://bryceroche--hi-moe-inference-vllmserver-serve.modal.run"
        print(f"Using LIVE endpoint: {endpoint}")
    else:
        endpoint = None
        print("Using MockLLMClient")

    # Parse routing mode
    routing_mode = RoutingMode(args.routing)

    # Run benchmark
    summary = asyncio.run(run_benchmark(
        problems=problems,
        endpoint=endpoint,
        mock=not args.live,
        routing_mode=routing_mode,
    ))

    # Print results
    print_summary(summary)

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output / f"benchmark_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
