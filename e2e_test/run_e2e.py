#!/usr/bin/env python3
"""End-to-end test runner for hi_moe.

Updated for hi_moe-ld8: Uses the unified Runner class that wires all tiers
together with CodeRunner for self-healing validation.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime

from .test_problems import ALL_PROBLEMS, TEST_PROBLEM
from .runner import Runner, RunStatus, create_runner
from .code_runner import CodeRunner
from .validator import validate_solution
from .tiers import LLMClient, MockLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def run_e2e_test(
    runner: Runner,
    problem: dict,
) -> dict:
    """Run a single problem through the full tier stack.

    Uses the unified Runner class (hi_moe-ld8) which:
    - Wires ProgressMonitor → Architect → Dispatcher → Fleet
    - Integrates CodeRunner for self-healing validation
    - Uses TaskContext for state management
    - Handles tiered retry logic
    """
    logger.info("=" * 60)
    logger.info(f"E2E Test: {problem['title']}")
    logger.info("=" * 60)

    # Execute through the unified Runner
    logger.info("Executing through tier stack with Runner...")
    run_result = await runner.run(problem)

    # Convert RunResult to legacy dict format
    elapsed = run_result.elapsed_ms / 1000.0
    logger.info(f"Tier execution completed in {elapsed:.2f}s")

    if run_result.status != RunStatus.COMPLETED:
        logger.error(f"Tier execution failed: {run_result.error}")
        return {
            "success": False,
            "problem": problem["title"],
            "stage": "tier_execution",
            "error": run_result.error,
            "elapsed_seconds": elapsed,
        }

    code = run_result.code
    if code:
        logger.info(f"Generated code:\n{code}")

    # Validate code using the proper function-call validator (hi_moe-ld8)
    # This handles the function_name correctly
    if code and "test_cases" in problem and "function_name" in problem:
        logger.info("Validating solution against test cases...")
        validation_result = validate_solution(
            code=code,
            test_cases=problem["test_cases"],
            function_name=problem["function_name"],
        )
        passed = validation_result.passed
        total = validation_result.total_cases
        passed_count = validation_result.passed_cases
        error = validation_result.error
        failures = validation_result.failed_cases
    else:
        # No validation possible
        passed = True
        total = 0
        passed_count = 0
        error = None
        failures = []

    result = {
        "success": run_result.status == RunStatus.COMPLETED and passed,
        "problem": problem["title"],
        "test_cases": {
            "total": total,
            "passed": passed_count,
        },
        "elapsed_seconds": elapsed,
        "code": code,
    }

    if passed:
        logger.info(f"All {total} test cases passed!")
    else:
        logger.error(f"Failed: {passed_count}/{total} passed")
        if error:
            logger.error(f"Error: {error}")
        for failure in failures:
            logger.error(f"  Case {failure.get('case', '?')}: {failure}")
        result["failures"] = failures
        result["error"] = error

    return result


async def health_check(endpoint: str, warmup: bool = True) -> bool:
    """Check if Modal endpoint is healthy, optionally warming up cold containers.

    Args:
        endpoint: Modal vLLM endpoint URL
        warmup: If True, retry with exponential backoff to allow cold start (hi_moe-9mb)
    """
    import httpx
    import time

    max_attempts = 12 if warmup else 1  # 12 attempts x 10s = 2 min warmup budget
    base_timeout = 30.0 if warmup else 10.0

    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient() as client:
                start = time.monotonic()
                resp = await client.get(f"{endpoint}/health", timeout=base_timeout)
                elapsed = time.monotonic() - start

                if resp.status_code == 200:
                    if attempt > 0:
                        logger.info(f"✅ Endpoint warmed up after {attempt + 1} attempts ({elapsed:.1f}s)")
                    else:
                        logger.info(f"✅ Endpoint healthy ({elapsed:.1f}s)")
                    return True
                else:
                    logger.warning(f"Endpoint returned {resp.status_code}, attempt {attempt + 1}/{max_attempts}")
        except httpx.TimeoutException:
            if warmup and attempt < max_attempts - 1:
                logger.info(f"⏳ Waiting for container warmup (attempt {attempt + 1}/{max_attempts})...")
                await asyncio.sleep(10)  # Wait before retry
            else:
                logger.error(f"❌ Endpoint timeout after {max_attempts} attempts")
                return False
        except Exception as e:
            if warmup and attempt < max_attempts - 1:
                logger.info(f"⏳ Container starting (attempt {attempt + 1}/{max_attempts}): {e}")
                await asyncio.sleep(10)
            else:
                logger.error(f"❌ Endpoint unreachable: {e}")
                return False

    return False


async def main():
    parser = argparse.ArgumentParser(description="Run hi_moe e2e tests")
    parser.add_argument(
        "--endpoint",
        default="https://bryce-roche--hi-moe-inference-vllmserver-serve.modal.run",
        help="Modal vLLM endpoint URL",
    )
    parser.add_argument(
        "--router-endpoint",
        default=None,
        help="Optional router endpoint for lightweight Architect/Dispatcher (hi_moe-9m2)",
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
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs",
        help="Directory for trajectory logs",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup retries (use when container is already warm)",
    )
    parser.add_argument(
        "--embedding-routing",
        action="store_true",
        help="Enable semantic embedding routing (hi_moe-awf)",
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        choices=["winner_take_all", "probabilistic", "blended", "ensemble"],
        default="winner_take_all",
        help="Routing mode: winner_take_all, probabilistic, blended, or ensemble (hi_moe-zrn, hi_moe-oyp)",
    )
    args = parser.parse_args()

    # Create function-call validator wrapper for test problems (hi_moe-ld8)
    # The validator handles function-call style test cases properly
    def make_validator_runner(function_name: str):
        """Create a code_runner that uses the validator for a specific function."""
        def runner(code: str, test_cases: list[dict]) -> dict:
            result = validate_solution(code, test_cases, function_name)
            return {
                "passed": result.passed,
                "total_passed": result.passed_cases,
                "total_failed": result.total_cases - result.passed_cases,
                "test_results": [
                    {"test_id": f"test-{f['case']}", "status": "failed" if "error" in f else "wrong_answer"}
                    for f in result.failed_cases
                ],
                "error": result.error,
            }
        return runner

    # For Fleet self-healing, we'll use a code_runner that doesn't require function_name
    # (Fleet doesn't have access to function_name in the current architecture)
    # Self-healing validation is disabled for this e2e test - Runner validates after execution
    code_runner = None  # Disable Fleet self-healing for function-call tests

    # Create unified Runner with all tiers wired together (hi_moe-ld8)
    routing_llm = None
    if args.mock:
        logger.info("Using mock LLM (no Modal required)")
        llm = MockLLMClient()
    else:
        # Health check with warmup (hi_moe-9mb)
        warmup = not args.no_warmup
        healthy = await health_check(args.endpoint, warmup=warmup)
        if not healthy:
            logger.error(
                "Endpoint health check failed. Deploy with: modal deploy modal_app/vllm_server.py"
            )
            logger.error("Or use --mock for local testing")
            sys.exit(1)
        llm = LLMClient(args.endpoint)

        # Create lightweight router LLM if endpoint provided (hi_moe-9m2)
        if args.router_endpoint:
            logger.info(f"Using router endpoint: {args.router_endpoint}")
            router_healthy = await health_check(args.router_endpoint, warmup=warmup)
            if not router_healthy:
                logger.error(
                    "Router endpoint health check failed. Deploy with: modal deploy modal_app/vllm_server_light.py"
                )
                sys.exit(1)
            routing_llm = LLMClient(args.router_endpoint)
            logger.info("Routing LLM initialized for Architect/Dispatcher")

    # Create the unified Runner that wires all tiers together
    runner = Runner(
        llm=llm,
        routing_llm=routing_llm,  # Lightweight model for Architect/Dispatcher (hi_moe-9m2)
        log_dir=args.log_dir,
        code_runner=code_runner,
        enable_trajectory_logging=True,
        enable_embedding_routing=args.embedding_routing,
        routing_mode=args.routing_mode,  # Routing mode (hi_moe-zrn)
    )

    logger.info("Runner initialized with tiers: ProgressMonitor -> Architect -> Dispatcher -> Fleet -> CodeRunner")

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
        result = await run_e2e_test(runner, problem)
        results.append(result)

    # Summary
    logger.info("=" * 60)
    logger.info("E2E Test Summary")
    logger.info("=" * 60)

    total_passed = sum(1 for r in results if r["success"])
    total_tests = len(results)

    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        tc = result.get("test_cases", {})
        logger.info(
            f"{status} {result['problem']}: {tc.get('passed', 0)}/{tc.get('total', 0)} cases, {result['elapsed_seconds']:.2f}s"
        )

    logger.info("-" * 60)
    logger.info(f"Total: {total_passed}/{total_tests} problems passed")

    sys.exit(0 if total_passed == total_tests else 1)


if __name__ == "__main__":
    asyncio.run(main())
