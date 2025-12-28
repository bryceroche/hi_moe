#!/usr/bin/env python3
"""A/B test: adapters vs base model (hi_moe-ioi).

Tests whether our LoRA adapters improve or hurt performance compared to base model.

Usage:
    python -m e2e_test.ab_test_adapters                    # Run with mock LLM
    python -m e2e_test.ab_test_adapters --live             # Run with real Modal endpoint
    python -m e2e_test.ab_test_adapters --problems 20      # Run N problems
    python -m e2e_test.ab_test_adapters --analyze          # Analyze existing results
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .tiers import (
    Task,
    MockLLMClient,
    LLMClient,
    RoutingDispatcher,
    SpecializedFleet,
    DispatcherMemory,
    FleetMemory,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# Sample problems covering different specialists
SAMPLE_PROBLEMS = [
    # Python/code problems
    {
        "id": "two_sum",
        "objective": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "context": {"examples": "Input: nums = [2,7,11,15], target = 9. Output: [0,1]"},
        "expected_specialist": "python",
    },
    {
        "id": "reverse_string",
        "objective": "Write a function that reverses a string in-place.",
        "context": {"examples": "Input: ['h','e','l','l','o']. Output: ['o','l','l','e','h']"},
        "expected_specialist": "python",
    },
    {
        "id": "valid_parentheses",
        "objective": "Given a string containing just '(', ')', '{', '}', '[' and ']', determine if valid.",
        "context": {"examples": "Input: '()[]{}'. Output: true"},
        "expected_specialist": "python",
    },
    # Math/reasoning problems
    {
        "id": "fibonacci",
        "objective": "Write a function to compute the nth Fibonacci number efficiently.",
        "context": {"examples": "fib(0)=0, fib(1)=1, fib(10)=55"},
        "expected_specialist": "math",
    },
    {
        "id": "prime_check",
        "objective": "Write a function to check if a number is prime.",
        "context": {"examples": "isPrime(7)=true, isPrime(4)=false"},
        "expected_specialist": "math",
    },
    {
        "id": "gcd",
        "objective": "Compute the greatest common divisor of two numbers using Euclidean algorithm.",
        "context": {"examples": "gcd(48, 18) = 6"},
        "expected_specialist": "math",
    },
    # Algorithm problems
    {
        "id": "binary_search",
        "objective": "Implement binary search to find target in sorted array, return index or -1.",
        "context": {"examples": "Input: [1,2,3,4,5], target=3. Output: 2"},
        "expected_specialist": "algorithms",
    },
    {
        "id": "merge_sort",
        "objective": "Implement merge sort algorithm.",
        "context": {"examples": "Input: [5,2,8,1]. Output: [1,2,5,8]"},
        "expected_specialist": "algorithms",
    },
    {
        "id": "max_subarray",
        "objective": "Find the contiguous subarray with the largest sum (Kadane's algorithm).",
        "context": {"examples": "Input: [-2,1,-3,4,-1,2,1,-5,4]. Output: 6"},
        "expected_specialist": "algorithms",
    },
    # Data structure problems
    {
        "id": "linked_list_reverse",
        "objective": "Reverse a singly linked list.",
        "context": {"examples": "Input: 1->2->3->4->5. Output: 5->4->3->2->1"},
        "expected_specialist": "data_structures",
    },
]


@dataclass
class TestResult:
    """Result from one test run."""
    problem_id: str
    adapters_enabled: bool
    success: bool
    specialist_used: str
    expected_specialist: str
    execution_time_ms: float
    error: str | None = None


@dataclass
class ABTestSummary:
    """Summary of A/B test results."""
    with_adapters: list[TestResult] = field(default_factory=list)
    without_adapters: list[TestResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def adapter_success_rate(self) -> float:
        if not self.with_adapters:
            return 0.0
        return sum(1 for r in self.with_adapters if r.success) / len(self.with_adapters)

    def base_success_rate(self) -> float:
        if not self.without_adapters:
            return 0.0
        return sum(1 for r in self.without_adapters if r.success) / len(self.without_adapters)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "with_adapters_count": len(self.with_adapters),
            "without_adapters_count": len(self.without_adapters),
            "adapter_success_rate": self.adapter_success_rate(),
            "base_success_rate": self.base_success_rate(),
            "with_adapters": [
                {
                    "problem_id": r.problem_id,
                    "success": r.success,
                    "specialist": r.specialist_used,
                    "expected": r.expected_specialist,
                    "time_ms": r.execution_time_ms,
                    "error": r.error,
                }
                for r in self.with_adapters
            ],
            "without_adapters": [
                {
                    "problem_id": r.problem_id,
                    "success": r.success,
                    "specialist": r.specialist_used,
                    "expected": r.expected_specialist,
                    "time_ms": r.execution_time_ms,
                    "error": r.error,
                }
                for r in self.without_adapters
            ],
        }


async def run_single_test(
    problem: dict,
    llm: LLMClient | MockLLMClient,
    adapters_enabled: bool,
) -> TestResult:
    """Run a single problem and return result."""
    import time

    # Fresh memories for each test
    dispatcher_memory = DispatcherMemory()
    fleet_memory = FleetMemory()

    # Build tier hierarchy with adapter toggle
    fleet = SpecializedFleet(
        llm,
        memory=fleet_memory,
        enable_adapters=adapters_enabled,
    )
    dispatcher = RoutingDispatcher(fleet, llm, memory=dispatcher_memory)

    # Create task
    task = Task(
        task_id=problem["id"],
        objective=problem["objective"],
        context=problem["context"],
    )

    # Execute
    start = time.perf_counter()
    try:
        outcome = await dispatcher.execute(task)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Get specialist from memory
        specialist = "unknown"
        if dispatcher_memory.routing_history:
            specialist = dispatcher_memory.routing_history[-1].get("specialist", "unknown")

        return TestResult(
            problem_id=problem["id"],
            adapters_enabled=adapters_enabled,
            success=outcome.status == TaskStatus.COMPLETED,
            specialist_used=specialist,
            expected_specialist=problem.get("expected_specialist", "unknown"),
            execution_time_ms=elapsed_ms,
            error=outcome.error if outcome.status != TaskStatus.COMPLETED else None,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return TestResult(
            problem_id=problem["id"],
            adapters_enabled=adapters_enabled,
            success=False,
            specialist_used="error",
            expected_specialist=problem.get("expected_specialist", "unknown"),
            execution_time_ms=elapsed_ms,
            error=str(e),
        )


async def run_ab_test(
    problems: list[dict],
    llm: LLMClient | MockLLMClient,
) -> ABTestSummary:
    """Run A/B test: same problems with adapters vs base model."""
    summary = ABTestSummary()

    print(f"\n{'='*60}")
    print("A/B TEST: ADAPTERS vs BASE MODEL")
    print(f"{'='*60}")
    print(f"Problems to test: {len(problems)}")

    # Shuffle for fairness
    shuffled = problems.copy()
    random.shuffle(shuffled)

    # Run WITH adapters
    print(f"\n--- Phase 1: WITH ADAPTERS ---")
    for i, problem in enumerate(shuffled, 1):
        result = await run_single_test(problem, llm, adapters_enabled=True)
        summary.with_adapters.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{i}/{len(shuffled)}] {problem['id']}: {status} ({result.specialist_used})")

    # Run WITHOUT adapters (base model only)
    print(f"\n--- Phase 2: BASE MODEL (no adapters) ---")
    for i, problem in enumerate(shuffled, 1):
        result = await run_single_test(problem, llm, adapters_enabled=False)
        summary.without_adapters.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{i}/{len(shuffled)}] {problem['id']}: {status} ({result.specialist_used})")

    return summary


def print_summary(summary: ABTestSummary):
    """Print A/B test summary."""
    print(f"\n{'='*60}")
    print("A/B TEST RESULTS: ADAPTERS vs BASE MODEL")
    print(f"{'='*60}")

    adapter_rate = summary.adapter_success_rate() * 100
    base_rate = summary.base_success_rate() * 100
    diff = adapter_rate - base_rate

    print(f"\nWith adapters:    {adapter_rate:.1f}% ({sum(1 for r in summary.with_adapters if r.success)}/{len(summary.with_adapters)})")
    print(f"Base model:       {base_rate:.1f}% ({sum(1 for r in summary.without_adapters if r.success)}/{len(summary.without_adapters)})")

    if diff > 0:
        print(f"\n✓ ADAPTERS ARE BETTER by {diff:.1f}pp")
    elif diff < 0:
        print(f"\n✗ ADAPTERS ARE WORSE by {-diff:.1f}pp")
        print("  Consider: disabling adapters, using HF adapters, or retraining")
    else:
        print(f"\n= NO DIFFERENCE")

    # Per-problem comparison
    print(f"\n--- Per-Problem Comparison ---")
    print(f"{'Problem':<25} {'Adapter':<10} {'Base':<10} {'Winner':<12}")
    print("-" * 60)

    adapter_wins = 0
    base_wins = 0
    ties = 0

    for ar, br in zip(summary.with_adapters, summary.without_adapters):
        a_status = "PASS" if ar.success else "FAIL"
        b_status = "PASS" if br.success else "FAIL"

        if ar.success == br.success:
            winner = "tie"
            ties += 1
        elif ar.success:
            winner = "ADAPTER"
            adapter_wins += 1
        else:
            winner = "BASE"
            base_wins += 1

        print(f"{ar.problem_id:<25} {a_status:<10} {b_status:<10} {winner:<12}")

    print(f"\nWins: Adapter={adapter_wins}, Base={base_wins}, Tie={ties}")

    # By expected specialist
    print(f"\n--- By Expected Specialist ---")
    specialists = set(r.expected_specialist for r in summary.with_adapters)
    for spec in sorted(specialists):
        adapter_results = [r for r in summary.with_adapters if r.expected_specialist == spec]
        base_results = [r for r in summary.without_adapters if r.expected_specialist == spec]

        adapter_pass = sum(1 for r in adapter_results if r.success)
        base_pass = sum(1 for r in base_results if r.success)
        total = len(adapter_results)

        if total > 0:
            a_rate = adapter_pass / total * 100
            b_rate = base_pass / total * 100
            diff = a_rate - b_rate
            indicator = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {spec:<15} Adapter: {a_rate:.0f}%, Base: {b_rate:.0f}% {indicator}")


def analyze_existing_results(results_dir: Path):
    """Analyze existing A/B test results."""
    results_files = list(results_dir.glob("ab_adapter_*.json"))
    if not results_files:
        print("No adapter A/B test results found.")
        return

    print(f"\n{'='*60}")
    print("HISTORICAL ADAPTER A/B TEST ANALYSIS")
    print(f"{'='*60}")
    print(f"Found {len(results_files)} test runs")

    all_adapter = []
    all_base = []

    for f in sorted(results_files):
        with open(f) as fp:
            data = json.load(fp)
        all_adapter.extend(data.get("with_adapters", []))
        all_base.extend(data.get("without_adapters", []))

    if not all_adapter or not all_base:
        print("No results to analyze.")
        return

    adapter_rate = sum(1 for r in all_adapter if r["success"]) / len(all_adapter) * 100
    base_rate = sum(1 for r in all_base if r["success"]) / len(all_base) * 100
    diff = adapter_rate - base_rate

    print(f"\nAggregate results:")
    print(f"  With adapters: {adapter_rate:.1f}% ({sum(1 for r in all_adapter if r['success'])}/{len(all_adapter)})")
    print(f"  Base model:    {base_rate:.1f}% ({sum(1 for r in all_base if r['success'])}/{len(all_base)})")
    print(f"  Difference:    {diff:+.1f}pp {'(adapters better)' if diff > 0 else '(base better)' if diff < 0 else ''}")


def main():
    parser = argparse.ArgumentParser(description="A/B test adapters vs base model")
    parser.add_argument("--live", action="store_true", help="Use live Modal endpoint")
    parser.add_argument("--endpoint", type=str, help="Custom Modal endpoint URL")
    parser.add_argument("--problems", type=int, default=10, help="Number of problems to test")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    parser.add_argument("--output", type=Path, default=Path("runs/ab_tests"), help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.analyze:
        analyze_existing_results(args.output)
        return

    # Initialize LLM
    if args.live:
        endpoint = args.endpoint or "https://bryceroche--hi-moe-inference-vllmserver-serve.modal.run"
        print(f"Using LIVE Modal endpoint: {endpoint}")
        llm = LLMClient(endpoint)
    else:
        print("Using MockLLMClient (offline test)")
        llm = MockLLMClient()

    # Select problems
    problems = SAMPLE_PROBLEMS[:args.problems]

    # Run test
    summary = asyncio.run(run_ab_test(problems, llm))

    # Print results
    print_summary(summary)

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output / f"ab_adapter_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
