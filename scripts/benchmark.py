#!/usr/bin/env python3
"""Benchmark suite for measuring hi_moe system quality (hi_moe-hvx).

Usage:
    python scripts/benchmark.py                     # Run full benchmark
    python scripts/benchmark.py --quick             # Quick smoke test (3 problems)
    python scripts/benchmark.py --category easy     # Run specific category
    python scripts/benchmark.py --report            # Generate report from past runs
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class BenchmarkProblem:
    """A problem in the benchmark suite."""
    id: str
    name: str
    category: Literal["easy", "medium", "hard"]
    problem_index: int  # Index in test_problems.py
    expected_pass: bool = True  # Whether we expect this to pass


@dataclass
class BenchmarkResult:
    """Result of running a single benchmark problem."""
    problem_id: str
    category: str
    passed: bool
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    num_calls: int = 0
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    timestamp: str
    results: list[BenchmarkResult] = field(default_factory=list)
    total_time_s: float = 0

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_in + r.tokens_out for r in self.results)

    @property
    def by_category(self) -> dict[str, dict]:
        """Group results by category."""
        cats = {}
        for r in self.results:
            if r.category not in cats:
                cats[r.category] = {"total": 0, "passed": 0, "tokens": 0}
            cats[r.category]["total"] += 1
            if r.passed:
                cats[r.category]["passed"] += 1
            cats[r.category]["tokens"] += r.tokens_in + r.tokens_out
        return cats


# Benchmark problem suite
# Maps to problems in e2e_test/test_problems.py by index
BENCHMARK_PROBLEMS = [
    # Easy - should pass consistently
    BenchmarkProblem("two_sum", "Two Sum", "easy", 0),
    BenchmarkProblem("valid_parens", "Valid Parentheses", "easy", 1),
    BenchmarkProblem("merge_intervals", "Merge Intervals", "easy", 2),

    # Medium - tests core capabilities
    BenchmarkProblem("longest_palindrome", "Longest Palindrome", "medium", 3),
    BenchmarkProblem("course_schedule", "Course Schedule", "medium", 4),
    BenchmarkProblem("lru_cache", "LRU Cache", "medium", 5),

    # Hard - stretch goals
    BenchmarkProblem("median_sorted", "Median of Sorted Arrays", "hard", 6),
    BenchmarkProblem("word_break", "Word Break II", "hard", 7),
]


def run_problem(problem: BenchmarkProblem, log_dir: Path) -> BenchmarkResult:
    """Run a single benchmark problem."""
    print(f"  Running {problem.name} ({problem.category})...")

    start = time.time()
    result = subprocess.run(
        [
            sys.executable, "-m", "e2e_test.run_e2e",
            "--problem", str(problem.problem_index),
            "--log-dir", str(log_dir),
        ],
        capture_output=True,
        text=True,
        timeout=600,  # 10 min timeout per problem
    )
    elapsed = time.time() - start

    # Parse result from output
    passed = "All" in result.stdout and "passed" in result.stdout
    error = None
    if result.returncode != 0:
        error = result.stderr[-500:] if result.stderr else "Unknown error"

    # Parse token usage from trajectory
    tokens_in, tokens_out, num_calls = 0, 0, 0
    for jsonl in log_dir.glob(f"run-{problem.id}*.jsonl"):
        with open(jsonl) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "vllm_call":
                        tokens_in += entry.get("tokens_in", 0)
                        tokens_out += entry.get("tokens_out", 0)
                        num_calls += 1
                except:
                    continue

    status = "✓" if passed else "✗"
    print(f"    {status} {elapsed:.1f}s, {tokens_in + tokens_out} tokens")

    return BenchmarkResult(
        problem_id=problem.id,
        category=problem.category,
        passed=passed,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=int(elapsed * 1000),
        num_calls=num_calls,
        error=error,
    )


def run_benchmark(
    problems: list[BenchmarkProblem],
    log_dir: Path,
) -> BenchmarkReport:
    """Run the full benchmark suite."""
    log_dir.mkdir(parents=True, exist_ok=True)

    report = BenchmarkReport(
        timestamp=datetime.now().isoformat(),
    )

    start = time.time()
    for problem in problems:
        try:
            result = run_problem(problem, log_dir)
            report.results.append(result)
        except subprocess.TimeoutExpired:
            report.results.append(BenchmarkResult(
                problem_id=problem.id,
                category=problem.category,
                passed=False,
                error="Timeout (>10 min)",
            ))
        except Exception as e:
            report.results.append(BenchmarkResult(
                problem_id=problem.id,
                category=problem.category,
                passed=False,
                error=str(e),
            ))

    report.total_time_s = time.time() - start
    return report


def print_report(report: BenchmarkReport) -> None:
    """Print benchmark report."""
    print("\n" + "=" * 60)
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total time: {report.total_time_s:.1f}s")
    print(f"Pass rate: {report.pass_rate * 100:.0f}% ({sum(1 for r in report.results if r.passed)}/{len(report.results)})")
    print(f"Total tokens: {report.total_tokens:,}")

    print("\n" + "-" * 60)
    print("BY CATEGORY")
    print("-" * 60)
    for cat, data in sorted(report.by_category.items()):
        rate = 100 * data["passed"] / data["total"] if data["total"] else 0
        print(f"  {cat:<10}: {data['passed']}/{data['total']} ({rate:.0f}%) - {data['tokens']:,} tokens")

    print("\n" + "-" * 60)
    print("DETAILED RESULTS")
    print("-" * 60)
    print(f"{'Problem':<25} {'Cat':<8} {'Pass':>6} {'Tokens':>10} {'Time':>8}")
    print("-" * 60)
    for r in report.results:
        status = "✓" if r.passed else "✗"
        print(f"{r.problem_id:<25} {r.category:<8} {status:>6} {r.tokens_in + r.tokens_out:>10,} {r.latency_ms/1000:>7.1f}s")


def save_report(report: BenchmarkReport, path: Path) -> None:
    """Save report to JSON."""
    with open(path, "w") as f:
        json.dump({
            "timestamp": report.timestamp,
            "total_time_s": report.total_time_s,
            "pass_rate": report.pass_rate,
            "total_tokens": report.total_tokens,
            "by_category": report.by_category,
            "results": [
                {
                    "problem_id": r.problem_id,
                    "category": r.category,
                    "passed": r.passed,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in report.results
            ],
        }, f, indent=2)
    print(f"\nReport saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark suite for hi_moe")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test (easy problems only)",
    )
    parser.add_argument(
        "--category",
        choices=["easy", "medium", "hard"],
        help="Run specific category only",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/benchmark"),
        help="Directory for benchmark logs",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Load and display existing report",
    )
    args = parser.parse_args()

    if args.report:
        with open(args.report) as f:
            data = json.load(f)
        report = BenchmarkReport(
            timestamp=data["timestamp"],
            total_time_s=data["total_time_s"],
            results=[
                BenchmarkResult(**r) for r in data["results"]
            ],
        )
        print_report(report)
        return 0

    # Select problems
    problems = BENCHMARK_PROBLEMS
    if args.quick:
        problems = [p for p in problems if p.category == "easy"]
    elif args.category:
        problems = [p for p in problems if p.category == args.category]

    print(f"Running {len(problems)} benchmark problems...")
    print("=" * 60)

    report = run_benchmark(problems, args.log_dir)
    print_report(report)

    # Save report
    report_path = args.log_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_report(report, report_path)

    return 0 if report.pass_rate >= 0.5 else 1


if __name__ == "__main__":
    exit(main())
