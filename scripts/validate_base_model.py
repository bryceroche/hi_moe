#!/usr/bin/env python3
"""Validate base model capabilities without LoRA adapters.

This script tests whether the base QwQ-32B model can solve coding problems
through the tier system using only prompt variations (no LoRA adapters).

The goal: If base model + prompts cannot pass problems, LoRAs won't save you.
If it can, architecture is validated without adapter complexity.
"""
from __future__ import annotations

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from e2e_test.tiers import (
    Task,
    LLMClient,
    MockLLMClient,
    ProgressMonitor,
    AbstractArchitect,
    RoutingDispatcher,
    SpecializedFleet,
)


# Test problems of increasing difficulty
TEST_PROBLEMS = [
    {
        "id": "easy-1",
        "difficulty": "easy",
        "problem": "Write a Python function called `add` that takes two numbers and returns their sum.",
        "expected_contains": ["def add", "return"],
    },
    {
        "id": "easy-2",
        "difficulty": "easy",
        "problem": "Write a Python function called `is_even` that returns True if a number is even, False otherwise.",
        "expected_contains": ["def is_even", "return", "%"],
    },
    {
        "id": "medium-1",
        "difficulty": "medium",
        "problem": "Write a Python function called `fibonacci` that returns the nth Fibonacci number using recursion or iteration.",
        "expected_contains": ["def fibonacci"],
    },
    {
        "id": "medium-2",
        "difficulty": "medium",
        "problem": "Write a Python function called `is_palindrome` that checks if a string is a palindrome (ignoring case and spaces).",
        "expected_contains": ["def is_palindrome"],
    },
    {
        "id": "hard-1",
        "difficulty": "hard",
        "problem": "Write a Python function called `merge_sort` that implements the merge sort algorithm to sort a list of integers.",
        "expected_contains": ["def merge_sort", "merge"],
    },
    {
        "id": "hard-2",
        "difficulty": "hard",
        "problem": "Write a Python function called `lcs` that finds the longest common subsequence of two strings and returns its length.",
        "expected_contains": ["def lcs"],
    },
]


async def run_validation(endpoint: str, use_mock: bool = False, verbose: bool = False):
    """Run validation tests against base model."""

    print("=" * 60)
    print("  Base Model Validation (No LoRA Adapters)")
    print("=" * 60)
    print()

    # Setup client
    if use_mock:
        print("Mode: MockLLMClient (local testing)")
        llm = MockLLMClient()
    else:
        print(f"Mode: Live inference at {endpoint}")
        llm = LLMClient(endpoint)

        # Check available adapters
        try:
            adapters = await llm.get_available_adapters()
            print(f"Available adapters: {adapters}")
            if adapters != ["base"]:
                print("WARNING: LoRA adapters detected. This test validates BASE model only.")
        except Exception as e:
            print(f"Could not fetch adapters: {e}")

    print()

    # Build tier stack (no LoRA routing - base model only)
    fleet = SpecializedFleet(llm)
    dispatcher = RoutingDispatcher(fleet, llm)
    architect = AbstractArchitect(dispatcher, llm)
    monitor = ProgressMonitor(architect)

    print("Tier stack: ProgressMonitor → Architect → Dispatcher → Fleet")
    print()

    # Run tests
    results = []

    for problem in TEST_PROBLEMS:
        print(f"[{problem['id']}] {problem['difficulty'].upper()}: {problem['problem'][:50]}...")

        task = Task(
            task_id=problem["id"],
            objective=problem["problem"],
        )

        try:
            outcome = await monitor.execute(task)

            # Check result
            success = outcome.status.value == "completed"
            code = outcome.result.get("code", "") if outcome.result else ""

            # Check if expected patterns are in code
            patterns_found = all(
                pattern.lower() in code.lower()
                for pattern in problem["expected_contains"]
            )

            result = {
                "id": problem["id"],
                "difficulty": problem["difficulty"],
                "status": outcome.status.value,
                "has_code": bool(code),
                "patterns_found": patterns_found,
                "execution_time_ms": outcome.execution_time_ms,
                "specialist": outcome.metadata.get("specialist"),
            }

            if success and patterns_found:
                print(f"  ✓ PASS ({outcome.execution_time_ms:.0f}ms)")
            elif success:
                print(f"  ~ PARTIAL - Code generated but missing expected patterns")
            else:
                print(f"  ✗ FAIL - {outcome.error or 'No code generated'}")

            if verbose and code:
                print(f"  Code preview: {code[:100]}...")

            results.append(result)

        except Exception as e:
            print(f"  ✗ ERROR - {e}")
            results.append({
                "id": problem["id"],
                "difficulty": problem["difficulty"],
                "status": "error",
                "error": str(e),
            })

    # Summary
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for r in results if r.get("patterns_found"))
    partial = sum(1 for r in results if r.get("has_code") and not r.get("patterns_found"))
    failed = total - passed - partial

    by_difficulty = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"passed": 0, "total": 0}
        by_difficulty[d]["total"] += 1
        if r.get("patterns_found"):
            by_difficulty[d]["passed"] += 1

    print(f"Total: {passed}/{total} passed ({100*passed/total:.0f}%)")
    print()
    for d in ["easy", "medium", "hard"]:
        if d in by_difficulty:
            stats = by_difficulty[d]
            print(f"  {d.capitalize()}: {stats['passed']}/{stats['total']}")

    print()

    # Conclusion
    if passed == total:
        print("CONCLUSION: Base model passes all tests. LoRA adapters may not be needed.")
        print("Consider keeping architecture simple with prompt-only specialists.")
    elif passed >= total * 0.7:
        print("CONCLUSION: Base model passes most tests. LoRA adapters could help edge cases.")
        print("Recommended: Validate on harder benchmarks before training adapters.")
    elif passed >= total * 0.3:
        print("CONCLUSION: Base model struggles. LoRA adapters may improve performance.")
        print("Recommended: Proceed with adapter training for weak areas.")
    else:
        print("CONCLUSION: Base model fails most tests. Check prompts and model config.")
        print("LoRA adapters alone may not fix fundamental issues.")

    # Save results
    output_file = Path(__file__).parent / "base_model_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "endpoint": endpoint if not use_mock else "mock",
            "results": results,
            "summary": {
                "total": total,
                "passed": passed,
                "partial": partial,
                "failed": failed,
                "by_difficulty": by_difficulty,
            }
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Validate base model without LoRA adapters")
    parser.add_argument(
        "--endpoint",
        default="https://bryce-roche--hi-moe-inference-vllmserver-serve.modal.run",
        help="Modal vLLM endpoint URL",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for local testing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show code previews",
    )
    args = parser.parse_args()

    passed, total = asyncio.run(run_validation(args.endpoint, args.mock, args.verbose))

    # Exit code based on pass rate
    if passed == total:
        sys.exit(0)
    elif passed >= total * 0.5:
        sys.exit(1)  # Partial success
    else:
        sys.exit(2)  # Failure


if __name__ == "__main__":
    main()
