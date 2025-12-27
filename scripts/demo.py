#!/usr/bin/env python3
"""Demo script showcasing the hi-moe tier system (hi_moe-2cf).

One-command demo that runs a problem through the full tier hierarchy,
displaying routing decisions, specialist execution, and results.

Usage:
    python scripts/demo.py                    # Run with mock LLM
    python scripts/demo.py --live             # Run with real Modal endpoint
    python scripts/demo.py --problem two_sum  # Run specific problem
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from e2e_test.tiers import (
    Task,
    MockLLMClient,
    LLMClient,
    RoutingDispatcher,
    SpecializedFleet,
    DispatcherMemory,
    FleetMemory,
    TaskStatus,
)


# Sample problems for demo
SAMPLE_PROBLEMS = {
    "two_sum": {
        "id": "two_sum",
        "objective": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.",
        "context": {
            "examples": "Input: nums = [2,7,11,15], target = 9. Output: [0,1]",
            "constraints": "Each input has exactly one solution. You may not use the same element twice.",
        },
    },
    "valid_parentheses": {
        "id": "valid_parentheses",
        "objective": "Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
        "context": {
            "examples": "Input: '()[]{}'. Output: true. Input: '(]'. Output: false.",
            "constraints": "An input string is valid if brackets are closed in the correct order.",
        },
    },
    "fibonacci": {
        "id": "fibonacci",
        "objective": "Write a function to compute the nth Fibonacci number efficiently.",
        "context": {
            "examples": "fib(0)=0, fib(1)=1, fib(10)=55",
            "constraints": "Handle large n efficiently (up to 10000).",
        },
    },
}


def print_header(text: str):
    """Print a styled header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---")


async def run_demo(problem_id: str, use_live: bool = False, endpoint: str | None = None):
    """Run a demo of the tier system."""
    print_header("HI-MOE TIER SYSTEM DEMO")

    # Get problem
    if problem_id not in SAMPLE_PROBLEMS:
        print(f"Unknown problem: {problem_id}")
        print(f"Available: {', '.join(SAMPLE_PROBLEMS.keys())}")
        return

    problem = SAMPLE_PROBLEMS[problem_id]

    print(f"\nProblem: {problem_id}")
    print(f"Objective: {problem['objective'][:80]}...")

    # Initialize LLM
    print_section("Initializing")
    if use_live:
        if not endpoint:
            endpoint = "https://bryceroche--hi-moe-inference-vllmserver-serve.modal.run"
        print(f"Using LIVE Modal endpoint: {endpoint}")
        llm = LLMClient(endpoint)
    else:
        print("Using MockLLMClient (offline demo)")
        llm = MockLLMClient()

    # Initialize memories
    dispatcher_memory = DispatcherMemory()
    fleet_memory = FleetMemory()
    print(f"Dispatcher memory: {len(dispatcher_memory.specialist_outcomes)} specialists tracked")
    print(f"Fleet memory: {len(fleet_memory.executions)} specialists tracked")

    # Initialize tiers
    print_section("Building Tier Hierarchy")
    fleet = SpecializedFleet(llm, memory=fleet_memory)
    dispatcher = RoutingDispatcher(fleet, llm, memory=dispatcher_memory)
    print("  Fleet (Tier 3) - Specialist execution")
    print("  Dispatcher (Tier 2) - Routing decisions")
    print("  [Architect and Monitor omitted for demo]")

    # Create task
    task = Task(
        task_id=problem["id"],
        objective=problem["objective"],
        context=problem["context"],
    )

    # Execute
    print_section("Executing Task")
    print(f"Task ID: {task.task_id}")
    print(f"Objective: {task.objective[:60]}...")
    print("\nRouting...")

    outcome = await dispatcher.execute(task)

    # Show results
    print_section("Results")
    status_icon = "✓" if outcome.status == TaskStatus.COMPLETED else "✗"
    print(f"Status: {status_icon} {outcome.status.value}")

    if outcome.error:
        print(f"Error: {outcome.error}")

    if outcome.result:
        print(f"\nResult preview:")
        result_str = str(outcome.result)
        if len(result_str) > 500:
            print(result_str[:500] + "...")
        else:
            print(result_str)

    # Show memory state
    print_section("Memory State After Execution")
    print("\nDispatcher Memory:")
    for spec, stats in dispatcher_memory.specialist_outcomes.items():
        total = stats["successes"] + stats["failures"]
        rate = stats["successes"] / total * 100 if total > 0 else 0
        print(f"  {spec}: {rate:.0f}% success ({stats['successes']}/{total})")

    if dispatcher_memory.routing_history:
        print("\nRouting History:")
        for r in dispatcher_memory.routing_history[-3:]:
            print(f"  {r['problem_type']} → {r['specialist']}")

    print("\nFleet Memory:")
    for spec, executions in fleet_memory.executions.items():
        successes = sum(1 for e in executions if e["success"])
        print(f"  {spec}: {successes}/{len(executions)} successful executions")

    # Summary
    print_header("Demo Complete")
    print(f"""
Summary:
  - Problem: {problem_id}
  - Status: {outcome.status.value}
  - Specialists used: {list(dispatcher_memory.specialist_outcomes.keys())}
  - Memory persisted: Ready for next run
    """)


def main():
    parser = argparse.ArgumentParser(description="Hi-MoE tier system demo")
    parser.add_argument(
        "--problem",
        type=str,
        default="two_sum",
        help=f"Problem to run ({', '.join(SAMPLE_PROBLEMS.keys())})",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live Modal endpoint instead of mock",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Custom Modal endpoint URL",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available problems",
    )
    args = parser.parse_args()

    if args.list:
        print("Available demo problems:")
        for pid, p in SAMPLE_PROBLEMS.items():
            print(f"  {pid}: {p['objective'][:60]}...")
        return

    asyncio.run(run_demo(args.problem, args.live, args.endpoint))


if __name__ == "__main__":
    main()
