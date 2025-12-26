#!/usr/bin/env python3
"""End-to-end test runner for hi_moe."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from typing import Union

from .test_problems import ALL_PROBLEMS, TEST_PROBLEM
from .tiers import (
    AbstractArchitect,
    LLMClient,
    MockLLMClient,
    ProgressMonitor,
    RoutingDispatcher,
    SpecializedFleet,
    Task,
)
from .validator import validate_solution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def run_e2e_test(
    llm: LLMClient | MockLLMClient,
    problem: dict,
) -> dict:
    """Run a single problem through the full tier stack."""

    logger.info("=" * 60)
    logger.info(f"E2E Test: {problem['title']}")
    logger.info("=" * 60)

    # Initialize tiers
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
            "problem": problem["title"],
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
        logger.error(
            f"❌ Failed: {validation.passed_cases}/{validation.total_cases} passed"
        )
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
        "--mock",
        action="store_true",
        help="Use mock LLM (no Modal required)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test problems",
    )
    parser.add_argument(
        "--problem",
        type=int,
        default=0,
        help="Problem index to run (default: 0)",
    )
    args = parser.parse_args()

    # Select LLM client
    if args.mock:
        logger.info("Using mock LLM (no Modal required)")
        llm = MockLLMClient()
    else:
        # Health check
        healthy = await health_check(args.endpoint)
        if not healthy:
            logger.error(
                "Endpoint health check failed. Deploy with: modal deploy modal_app/vllm_server.py"
            )
            logger.error("Or use --mock for local testing")
            sys.exit(1)
        llm = LLMClient(args.endpoint)

    # Select problems
    if args.all:
        problems = ALL_PROBLEMS
    else:
        if args.problem >= len(ALL_PROBLEMS):
            logger.error(f"Problem index {args.problem} out of range (0-{len(ALL_PROBLEMS)-1})")
            sys.exit(1)
        problems = [ALL_PROBLEMS[args.problem]]

    # Run tests
    results = []
    for problem in problems:
        result = await run_e2e_test(llm, problem)
        results.append(result)

    # Summary
    logger.info("=" * 60)
    logger.info("E2E Test Summary")
    logger.info("=" * 60)

    total_passed = sum(1 for r in results if r["success"])
    total_tests = len(results)

    for result in results:
        status = "✅" if result["success"] else "❌"
        tc = result.get("test_cases", {})
        logger.info(
            f"{status} {result['problem']}: {tc.get('passed', 0)}/{tc.get('total', 0)} cases, {result['elapsed_seconds']:.2f}s"
        )

    logger.info("-" * 60)
    logger.info(f"Total: {total_passed}/{total_tests} problems passed")

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    asyncio.run(main())
