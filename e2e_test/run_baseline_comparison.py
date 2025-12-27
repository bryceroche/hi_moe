#!/usr/bin/env python3
"""Baseline comparison: single model vs hi-moe hierarchy (hi_moe-0qv).

Compares:
1. Baseline: Single direct LLM call with base model (no adapters, no hierarchy)
2. Hi-MoE: Full tier stack (Monitor → Architect → Dispatcher → Fleet)

Measures: pass rate, latency, LLM call count.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .test_problems import ALL_PROBLEMS
from .runner import Runner, RunStatus
from .tiers import LLMClient, MockLLMClient
from .validator import validate_solution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of a single problem run."""
    problem_id: str
    problem_title: str
    approach: str  # "baseline" or "hierarchy"
    passed: bool
    tests_passed: int
    tests_total: int
    elapsed_ms: float
    llm_calls: int
    code: str | None = None
    error: str | None = None


@dataclass
class ComparisonSummary:
    """Summary of all comparison runs."""
    results: list[ComparisonResult] = field(default_factory=list)

    def add(self, result: ComparisonResult) -> None:
        self.results.append(result)

    def baseline_results(self) -> list[ComparisonResult]:
        return [r for r in self.results if r.approach == "baseline"]

    def hierarchy_results(self) -> list[ComparisonResult]:
        return [r for r in self.results if r.approach == "hierarchy"]

    def print_summary(self) -> None:
        """Print comparison summary."""
        baseline = self.baseline_results()
        hierarchy = self.hierarchy_results()

        print("\n" + "=" * 70)
        print("BASELINE vs HIERARCHY COMPARISON (hi_moe-0qv)")
        print("=" * 70)

        # Pass rates
        baseline_passed = sum(1 for r in baseline if r.passed)
        hierarchy_passed = sum(1 for r in hierarchy if r.passed)

        print(f"\n{'Metric':<25} {'Baseline':>15} {'Hi-MoE':>15} {'Delta':>15}")
        print("-" * 70)

        # Pass rate
        bp = baseline_passed / len(baseline) * 100 if baseline else 0
        hp = hierarchy_passed / len(hierarchy) * 100 if hierarchy else 0
        print(f"{'Pass Rate':<25} {bp:>14.1f}% {hp:>14.1f}% {hp - bp:>+14.1f}%")

        # Avg latency
        bl = sum(r.elapsed_ms for r in baseline) / len(baseline) if baseline else 0
        hl = sum(r.elapsed_ms for r in hierarchy) / len(hierarchy) if hierarchy else 0
        print(f"{'Avg Latency (ms)':<25} {bl:>15.0f} {hl:>15.0f} {hl - bl:>+15.0f}")

        # Avg LLM calls
        bc = sum(r.llm_calls for r in baseline) / len(baseline) if baseline else 0
        hc = sum(r.llm_calls for r in hierarchy) / len(hierarchy) if hierarchy else 0
        print(f"{'Avg LLM Calls':<25} {bc:>15.1f} {hc:>15.1f} {hc - bc:>+15.1f}")

        print("-" * 70)

        # Individual results
        print("\nPer-Problem Results:")
        print(f"{'Problem':<30} {'Baseline':>10} {'Hi-MoE':>10} {'Time (B)':>12} {'Time (H)':>12}")
        print("-" * 70)

        problems = set(r.problem_id for r in self.results)
        for pid in sorted(problems):
            br = next((r for r in baseline if r.problem_id == pid), None)
            hr = next((r for r in hierarchy if r.problem_id == pid), None)

            b_status = "PASS" if br and br.passed else "FAIL"
            h_status = "PASS" if hr and hr.passed else "FAIL"
            b_time = f"{br.elapsed_ms:.0f}ms" if br else "-"
            h_time = f"{hr.elapsed_ms:.0f}ms" if hr else "-"

            title = (br or hr).problem_title[:28] if (br or hr) else pid[:28]
            print(f"{title:<30} {b_status:>10} {h_status:>10} {b_time:>12} {h_time:>12}")

        print("=" * 70)


def extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try ```python blocks
    match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic ``` blocks
    match = re.search(r"```\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: assume entire response is code (strip <think> tags first)
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return cleaned.strip()


async def run_baseline(
    llm: LLMClient,
    problem: dict,
) -> ComparisonResult:
    """Run a problem through the baseline single-model approach.

    Single LLM call, base model, no adapters, no hierarchy.
    """
    start_time = time.monotonic()

    # Build a simple direct prompt
    prompt = f"""Write a Python function to solve this problem.

Problem: {problem['title']}

{problem['statement']}

Requirements:
- Function name: {problem['function_name']}
- Signature: {problem['function_signature']}

Return ONLY the Python code in a ```python block. No explanation."""

    messages = [{"role": "user", "content": prompt}]

    try:
        # Single direct call with base model (no adapter)
        response = await llm.generate(messages, temperature=0.3, max_tokens=2048, adapter=None)
        llm_calls = 1

        # Extract code
        code = extract_code(response)

        # Validate
        validation = validate_solution(
            code=code,
            test_cases=problem["test_cases"],
            function_name=problem["function_name"],
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return ComparisonResult(
            problem_id=problem["id"],
            problem_title=problem["title"],
            approach="baseline",
            passed=validation.passed,
            tests_passed=validation.passed_cases,
            tests_total=validation.total_cases,
            elapsed_ms=elapsed_ms,
            llm_calls=llm_calls,
            code=code,
            error=validation.error,
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(f"Baseline error: {e}")
        return ComparisonResult(
            problem_id=problem["id"],
            problem_title=problem["title"],
            approach="baseline",
            passed=False,
            tests_passed=0,
            tests_total=len(problem.get("test_cases", [])),
            elapsed_ms=elapsed_ms,
            llm_calls=1,
            error=str(e),
        )


async def run_hierarchy(
    runner: Runner,
    problem: dict,
) -> ComparisonResult:
    """Run a problem through the hi-moe hierarchy.

    Full tier stack: Monitor → Architect → Dispatcher → Fleet.
    """
    start_time = time.monotonic()

    try:
        result = await runner.run(problem)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Count LLM calls from trajectory logger
        llm_calls = 0
        if runner.trajectory_logger:
            llm_calls = runner.trajectory_logger._call_count

        # Validate code
        passed = False
        tests_passed = 0
        tests_total = len(problem.get("test_cases", []))
        error = result.error

        if result.code:
            validation = validate_solution(
                code=result.code,
                test_cases=problem["test_cases"],
                function_name=problem["function_name"],
            )
            passed = validation.passed
            tests_passed = validation.passed_cases
            tests_total = validation.total_cases
            error = validation.error

        return ComparisonResult(
            problem_id=problem["id"],
            problem_title=problem["title"],
            approach="hierarchy",
            passed=passed,
            tests_passed=tests_passed,
            tests_total=tests_total,
            elapsed_ms=elapsed_ms,
            llm_calls=llm_calls,
            code=result.code,
            error=error,
        )

    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error(f"Hierarchy error: {e}")
        return ComparisonResult(
            problem_id=problem["id"],
            problem_title=problem["title"],
            approach="hierarchy",
            passed=False,
            tests_passed=0,
            tests_total=len(problem.get("test_cases", [])),
            elapsed_ms=elapsed_ms,
            llm_calls=0,
            error=str(e),
        )


async def health_check(endpoint: str) -> bool:
    """Check if Modal endpoint is healthy."""
    import httpx

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{endpoint}/health", timeout=10)
            if resp.status_code == 200:
                logger.info(f"Endpoint healthy: {endpoint}")
                return True
            else:
                logger.error(f"Endpoint returned {resp.status_code}")
                return False
    except Exception as e:
        logger.error(f"Endpoint unreachable: {e}")
        return False


async def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline (single model) vs hi-moe hierarchy"
    )
    parser.add_argument(
        "--endpoint",
        default="https://bryce-roche--hi-moe-inference-vllmserver-serve.modal.run",
        help="Modal vLLM endpoint URL",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (no Modal required)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    args = parser.parse_args()

    # Setup LLM
    if args.mock:
        logger.info("Using mock LLM")
        llm = MockLLMClient()
    else:
        healthy = await health_check(args.endpoint)
        if not healthy:
            logger.error("Endpoint health check failed")
            sys.exit(1)
        llm = LLMClient(args.endpoint)

    # Create hierarchy runner
    runner = Runner(
        llm=llm,
        log_dir="./runs/comparison",
        enable_trajectory_logging=True,
    )

    summary = ComparisonSummary()

    # Run each problem through both approaches
    for problem in ALL_PROBLEMS:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Problem: {problem['title']}")
        logger.info("=" * 60)

        # Run baseline
        logger.info("Running BASELINE (single model, no hierarchy)...")
        baseline_result = await run_baseline(llm, problem)
        summary.add(baseline_result)
        logger.info(
            f"Baseline: {'PASS' if baseline_result.passed else 'FAIL'} "
            f"({baseline_result.tests_passed}/{baseline_result.tests_total}) "
            f"in {baseline_result.elapsed_ms:.0f}ms"
        )

        # Run hierarchy
        logger.info("Running HIERARCHY (full tier stack)...")
        hierarchy_result = await run_hierarchy(runner, problem)
        summary.add(hierarchy_result)
        logger.info(
            f"Hierarchy: {'PASS' if hierarchy_result.passed else 'FAIL'} "
            f"({hierarchy_result.tests_passed}/{hierarchy_result.tests_total}) "
            f"in {hierarchy_result.elapsed_ms:.0f}ms, "
            f"{hierarchy_result.llm_calls} LLM calls"
        )

    # Print summary
    summary.print_summary()

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "results": [
                        {
                            "problem_id": r.problem_id,
                            "problem_title": r.problem_title,
                            "approach": r.approach,
                            "passed": r.passed,
                            "tests_passed": r.tests_passed,
                            "tests_total": r.tests_total,
                            "elapsed_ms": r.elapsed_ms,
                            "llm_calls": r.llm_calls,
                            "error": r.error,
                        }
                        for r in summary.results
                    ]
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
