#!/usr/bin/env python3
"""A/B test: adaptive confidence vs static confidence (hi_moe-c5m).

Compares routing performance with historical data (adaptive) vs fixed confidence (static).
Runs the same problems twice and measures success rates.

Usage:
    python -m e2e_test.ab_test_confidence                    # Run with mock LLM
    python -m e2e_test.ab_test_confidence --live             # Run with real Modal endpoint
    python -m e2e_test.ab_test_confidence --problems 20      # Run N problems
    python -m e2e_test.ab_test_confidence --analyze          # Analyze existing results
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
from .call_db import CallDB

logger = logging.getLogger(__name__)

# Sample problems for testing (subset from demo)
SAMPLE_PROBLEMS = [
    {
        "id": "two_sum",
        "objective": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "context": {
            "examples": "Input: nums = [2,7,11,15], target = 9. Output: [0,1]",
            "constraints": "Each input has exactly one solution.",
        },
    },
    {
        "id": "valid_parentheses",
        "objective": "Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
        "context": {
            "examples": "Input: '()[]{}'. Output: true.",
            "constraints": "Brackets must be closed in the correct order.",
        },
    },
    {
        "id": "fibonacci",
        "objective": "Write a function to compute the nth Fibonacci number efficiently.",
        "context": {
            "examples": "fib(0)=0, fib(1)=1, fib(10)=55",
            "constraints": "Handle large n efficiently.",
        },
    },
    {
        "id": "binary_search",
        "objective": "Implement binary search to find the index of target in a sorted array, or -1 if not found.",
        "context": {
            "examples": "Input: [1,2,3,4,5], target=3. Output: 2",
            "constraints": "Must be O(log n) time complexity.",
        },
    },
    {
        "id": "reverse_linked_list",
        "objective": "Reverse a singly linked list and return the new head.",
        "context": {
            "examples": "Input: 1->2->3->4->5. Output: 5->4->3->2->1",
            "constraints": "Do it iteratively and recursively.",
        },
    },
    {
        "id": "merge_sorted_arrays",
        "objective": "Merge two sorted arrays into one sorted array.",
        "context": {
            "examples": "Input: [1,3,5], [2,4,6]. Output: [1,2,3,4,5,6]",
            "constraints": "O(n+m) time complexity.",
        },
    },
    {
        "id": "max_subarray",
        "objective": "Find the contiguous subarray with the largest sum.",
        "context": {
            "examples": "Input: [-2,1,-3,4,-1,2,1,-5,4]. Output: 6 (subarray [4,-1,2,1])",
            "constraints": "Must be O(n) time.",
        },
    },
    {
        "id": "palindrome_check",
        "objective": "Check if a string is a palindrome, ignoring non-alphanumeric characters.",
        "context": {
            "examples": "Input: 'A man, a plan, a canal: Panama'. Output: true",
            "constraints": "Case-insensitive comparison.",
        },
    },
]


@dataclass
class ABTestResult:
    """Result from one test run."""
    problem_id: str
    adaptive_enabled: bool
    success: bool
    specialist_chosen: str
    confidence: float
    execution_time_ms: float
    error: str | None = None


@dataclass
class ABTestSummary:
    """Summary of A/B test results."""
    adaptive_results: list[ABTestResult] = field(default_factory=list)
    static_results: list[ABTestResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def adaptive_success_rate(self) -> float:
        if not self.adaptive_results:
            return 0.0
        return sum(1 for r in self.adaptive_results if r.success) / len(self.adaptive_results)

    def static_success_rate(self) -> float:
        if not self.static_results:
            return 0.0
        return sum(1 for r in self.static_results if r.success) / len(self.static_results)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "adaptive_count": len(self.adaptive_results),
            "static_count": len(self.static_results),
            "adaptive_success_rate": self.adaptive_success_rate(),
            "static_success_rate": self.static_success_rate(),
            "adaptive_results": [
                {
                    "problem_id": r.problem_id,
                    "success": r.success,
                    "specialist": r.specialist_chosen,
                    "confidence": r.confidence,
                    "time_ms": r.execution_time_ms,
                    "error": r.error,
                }
                for r in self.adaptive_results
            ],
            "static_results": [
                {
                    "problem_id": r.problem_id,
                    "success": r.success,
                    "specialist": r.specialist_chosen,
                    "confidence": r.confidence,
                    "time_ms": r.execution_time_ms,
                    "error": r.error,
                }
                for r in self.static_results
            ],
        }


async def run_single_test(
    problem: dict,
    llm: LLMClient | MockLLMClient,
    adaptive_enabled: bool,
    call_db: CallDB | None = None,
) -> ABTestResult:
    """Run a single problem and return result."""
    import time

    # Fresh memories for each test
    dispatcher_memory = DispatcherMemory()
    fleet_memory = FleetMemory()

    # Build tier hierarchy
    fleet = SpecializedFleet(llm, memory=fleet_memory)
    dispatcher = RoutingDispatcher(
        fleet,
        llm,
        memory=dispatcher_memory,
        call_db=call_db,
    )
    dispatcher.enable_adaptive_confidence = adaptive_enabled

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

        # Get routing info from memory
        specialist = "unknown"
        if dispatcher_memory.routing_history:
            specialist = dispatcher_memory.routing_history[-1].get("specialist", "unknown")

        return ABTestResult(
            problem_id=problem["id"],
            adaptive_enabled=adaptive_enabled,
            success=outcome.status == TaskStatus.COMPLETED,
            specialist_chosen=specialist,
            confidence=0.7,  # Base confidence (actual logged in call_db)
            execution_time_ms=elapsed_ms,
            error=outcome.error if outcome.status != TaskStatus.COMPLETED else None,
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ABTestResult(
            problem_id=problem["id"],
            adaptive_enabled=adaptive_enabled,
            success=False,
            specialist_chosen="error",
            confidence=0.0,
            execution_time_ms=elapsed_ms,
            error=str(e),
        )


async def run_ab_test(
    problems: list[dict],
    llm: LLMClient | MockLLMClient,
    call_db: CallDB | None = None,
) -> ABTestSummary:
    """Run A/B test: same problems with adaptive vs static confidence."""
    summary = ABTestSummary()

    print(f"\n{'='*60}")
    print("A/B TEST: ADAPTIVE vs STATIC CONFIDENCE")
    print(f"{'='*60}")
    print(f"Problems to test: {len(problems)}")

    # Shuffle problems for fairness
    shuffled = problems.copy()
    random.shuffle(shuffled)

    # Run with adaptive confidence
    print(f"\n--- Phase 1: ADAPTIVE confidence ---")
    for i, problem in enumerate(shuffled, 1):
        result = await run_single_test(problem, llm, adaptive_enabled=True, call_db=call_db)
        summary.adaptive_results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{i}/{len(shuffled)}] {problem['id']}: {status} ({result.specialist_chosen})")

    # Run with static confidence
    print(f"\n--- Phase 2: STATIC confidence ---")
    for i, problem in enumerate(shuffled, 1):
        result = await run_single_test(problem, llm, adaptive_enabled=False, call_db=call_db)
        summary.static_results.append(result)
        status = "PASS" if result.success else "FAIL"
        print(f"  [{i}/{len(shuffled)}] {problem['id']}: {status} ({result.specialist_chosen})")

    return summary


def print_summary(summary: ABTestSummary):
    """Print A/B test summary."""
    print(f"\n{'='*60}")
    print("A/B TEST RESULTS")
    print(f"{'='*60}")

    adaptive_rate = summary.adaptive_success_rate() * 100
    static_rate = summary.static_success_rate() * 100
    diff = adaptive_rate - static_rate

    print(f"\nAdaptive confidence: {adaptive_rate:.1f}% ({sum(1 for r in summary.adaptive_results if r.success)}/{len(summary.adaptive_results)})")
    print(f"Static confidence:   {static_rate:.1f}% ({sum(1 for r in summary.static_results if r.success)}/{len(summary.static_results)})")

    if diff > 0:
        print(f"\n Adaptive is {diff:.1f}pp BETTER")
    elif diff < 0:
        print(f"\n Static is {-diff:.1f}pp BETTER")
    else:
        print(f"\n No difference")

    # Per-problem comparison
    print(f"\n--- Per-Problem Comparison ---")
    print(f"{'Problem':<25} {'Adaptive':<10} {'Static':<10} {'Diff':<10}")
    print("-" * 55)

    for ar, sr in zip(summary.adaptive_results, summary.static_results):
        a_status = "PASS" if ar.success else "FAIL"
        s_status = "PASS" if sr.success else "FAIL"
        if ar.success == sr.success:
            diff_str = "same"
        elif ar.success:
            diff_str = "A better"
        else:
            diff_str = "S better"
        print(f"{ar.problem_id:<25} {a_status:<10} {s_status:<10} {diff_str:<10}")

    # Specialist distribution
    print(f"\n--- Specialist Distribution ---")
    adaptive_specs = {}
    static_specs = {}
    for r in summary.adaptive_results:
        adaptive_specs[r.specialist_chosen] = adaptive_specs.get(r.specialist_chosen, 0) + 1
    for r in summary.static_results:
        static_specs[r.specialist_chosen] = static_specs.get(r.specialist_chosen, 0) + 1

    print(f"Adaptive: {adaptive_specs}")
    print(f"Static:   {static_specs}")


def analyze_existing_results(results_dir: Path):
    """Analyze existing A/B test results."""
    results_files = list(results_dir.glob("ab_test_*.json"))
    if not results_files:
        print("No A/B test results found.")
        return

    print(f"\n{'='*60}")
    print(f"HISTORICAL A/B TEST ANALYSIS")
    print(f"{'='*60}")
    print(f"Found {len(results_files)} test runs")

    all_adaptive = []
    all_static = []

    for f in sorted(results_files):
        with open(f) as fp:
            data = json.load(fp)
        all_adaptive.extend(data.get("adaptive_results", []))
        all_static.extend(data.get("static_results", []))

    if not all_adaptive or not all_static:
        print("No results to analyze.")
        return

    adaptive_rate = sum(1 for r in all_adaptive if r["success"]) / len(all_adaptive) * 100
    static_rate = sum(1 for r in all_static if r["success"]) / len(all_static) * 100
    diff = adaptive_rate - static_rate

    print(f"\nAggregate results:")
    print(f"  Adaptive: {adaptive_rate:.1f}% ({sum(1 for r in all_adaptive if r['success'])}/{len(all_adaptive)})")
    print(f"  Static:   {static_rate:.1f}% ({sum(1 for r in all_static if r['success'])}/{len(all_static)})")
    print(f"  Diff:     {diff:+.1f}pp")

    # Statistical significance (simple)
    if len(all_adaptive) >= 30:
        import math
        n = len(all_adaptive)
        p1 = adaptive_rate / 100
        p2 = static_rate / 100
        pooled = (p1 + p2) / 2
        se = math.sqrt(pooled * (1 - pooled) * 2 / n) if pooled > 0 else 0
        z = abs(p1 - p2) / se if se > 0 else 0
        significant = z > 1.96  # 95% confidence
        print(f"  Z-score:  {z:.2f} ({'significant' if significant else 'not significant'} at 95%)")


def main():
    parser = argparse.ArgumentParser(description="A/B test adaptive vs static confidence")
    parser.add_argument("--live", action="store_true", help="Use live Modal endpoint")
    parser.add_argument("--endpoint", type=str, help="Custom Modal endpoint URL")
    parser.add_argument("--problems", type=int, default=8, help="Number of problems to test")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing results")
    parser.add_argument("--output", type=Path, default=Path("runs/ab_tests"), help="Output directory")
    parser.add_argument("--db", type=Path, default=Path("runs/hi_moe.db"), help="CallDB path")
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

    # Initialize CallDB
    call_db = CallDB(args.db) if args.db.exists() else None

    # Select problems
    problems = SAMPLE_PROBLEMS[:args.problems]

    # Run test
    summary = asyncio.run(run_ab_test(problems, llm, call_db))

    # Print results
    print_summary(summary)

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output / f"ab_test_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
