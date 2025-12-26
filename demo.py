#!/usr/bin/env python3
"""Interactive demo for hi_moe tier system."""
from __future__ import annotations

import asyncio
import argparse
import sys


async def run_demo(endpoint: str, use_mock: bool = False):
    """Run interactive demo of the tier system."""
    from e2e_test.tiers import (
        Task,
        LLMClient,
        MockLLMClient,
        ProgressMonitor,
        AbstractArchitect,
        RoutingDispatcher,
        SpecializedFleet,
    )

    print("=" * 60)
    print("  hi_moe - Hierarchical Mixture of Experts Demo")
    print("=" * 60)
    print()

    if use_mock:
        print("Using: MockLLMClient (local testing)")
        llm = MockLLMClient()
    else:
        print(f"Using: {endpoint}")
        llm = LLMClient(endpoint)

        # Check available adapters
        try:
            adapters = await llm.get_available_adapters()
            print(f"Available adapters: {adapters}")
        except Exception as e:
            print(f"Warning: Could not fetch adapters: {e}")

    print()

    # Build tier stack
    fleet = SpecializedFleet(llm)
    dispatcher = RoutingDispatcher(fleet, llm)
    architect = AbstractArchitect(dispatcher, llm)
    monitor = ProgressMonitor(architect)

    print("Tier stack initialized:")
    print("  T4: ProgressMonitor")
    print("  T1: AbstractArchitect")
    print("  T2: RoutingDispatcher")
    print("  T3: SpecializedFleet")
    print()
    print("Enter a coding problem (or 'quit' to exit):")
    print("-" * 60)

    task_counter = 0

    while True:
        try:
            problem = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not problem:
            continue

        if problem.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        task_counter += 1
        task = Task(
            task_id=f"demo-{task_counter}",
            objective=problem,
        )

        print(f"\n[Task {task.task_id}] Processing...")

        try:
            outcome = await monitor.execute(task)

            print(f"\n[Status] {outcome.status.value}")
            print(f"[Time] {outcome.execution_time_ms:.0f}ms")

            if outcome.result:
                code = outcome.result.get("code", "")
                if code:
                    print("\n[Code]")
                    print("-" * 40)
                    print(code)
                    print("-" * 40)

                # Show adapter used
                adapter = outcome.metadata.get("adapter")
                specialist = outcome.metadata.get("specialist")
                if specialist:
                    adapter_info = f" (adapter: {adapter})" if adapter else " (base)"
                    print(f"[Specialist] {specialist}{adapter_info}")

            if outcome.error:
                print(f"[Error] {outcome.error}")

        except Exception as e:
            print(f"[Error] {e}")


def main():
    parser = argparse.ArgumentParser(description="hi_moe interactive demo")
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
    args = parser.parse_args()

    asyncio.run(run_demo(args.endpoint, args.mock))


if __name__ == "__main__":
    main()
