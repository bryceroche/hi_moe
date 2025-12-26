#!/usr/bin/env python3
"""Validate base model performance without LoRA adapters.

Implements issue hi_moe-rj4: Test if the hierarchy works with just base QwQ-32B + prompts.
Specialists are prompt variations, not LoRA adapters.

The test: if base model + prompts cannot pass competitive programming problems,
adding LoRAs probably won't save you. If it can, architecture is validated
without adapter complexity.

Usage:
    # Mock mode (local testing)
    python -m e2e_test.validate_base_model --mock

    # Live mode (requires Modal endpoint)
    python -m e2e_test.validate_base_model --endpoint https://your-endpoint.modal.run

    # Run specific problems
    python -m e2e_test.validate_base_model --mock --problems 0 1 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .runner import Runner, RunResult, RunStatus, create_runner
from .test_problems import ALL_PROBLEMS, EASY_PROBLEMS, MEDIUM_PROBLEMS
from .tiers import LLMClient, MockLLMClient
from .trajectory_logger import (
    load_trajectory,
    compute_trajectory_stats,
    compute_tier_stats,
)
from .validator import validate_solution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Report from base model validation run."""

    timestamp: str
    mode: str  # "mock" or "live"
    problems_tested: int
    problems_passed: int
    problems_failed: int
    pass_rate: float
    results: list[dict] = field(default_factory=list)
    trajectory_stats: dict = field(default_factory=dict)
    tier_stats: dict = field(default_factory=dict)
    total_elapsed_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "mode": self.mode,
            "problems_tested": self.problems_tested,
            "problems_passed": self.problems_passed,
            "problems_failed": self.problems_failed,
            "pass_rate": self.pass_rate,
            "results": self.results,
            "trajectory_stats": self.trajectory_stats,
            "tier_stats": self.tier_stats,
            "total_elapsed_ms": self.total_elapsed_ms,
        }


async def validate_problem(
    runner: Runner,
    problem: dict,
) -> dict:
    """Run a single problem through the hierarchy and validate.

    Args:
        runner: Runner instance
        problem: Problem definition

    Returns:
        Result dict with validation info
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {problem['title']} ({problem['difficulty']})")
    logger.info(f"{'='*60}")

    # Run through hierarchy
    result = await runner.run(problem)

    # Validate if we got code
    validation_passed = False
    validation_details = {}

    if result.status == RunStatus.COMPLETED and result.code:
        logger.info(f"Generated code:\n{result.code[:500]}...")

        # Validate solution
        validation = validate_solution(
            code=result.code,
            test_cases=problem["test_cases"],
            function_name=problem["function_name"],
        )

        validation_passed = validation.passed
        validation_details = {
            "total_cases": validation.total_cases,
            "passed_cases": validation.passed_cases,
            "failed_cases": validation.failed_cases,
            "error": validation.error,
        }

        if validation.passed:
            logger.info(f"✅ PASSED: {validation.passed_cases}/{validation.total_cases} test cases")
        else:
            logger.error(f"❌ FAILED: {validation.passed_cases}/{validation.total_cases} test cases")
            if validation.error:
                logger.error(f"   Error: {validation.error}")
    else:
        logger.error(f"❌ FAILED: No code generated")
        if result.error:
            logger.error(f"   Error: {result.error}")

    return {
        "problem_id": problem["id"],
        "problem_title": problem["title"],
        "difficulty": problem["difficulty"],
        "run_id": result.run_id,
        "run_status": result.status.value,
        "validation_passed": validation_passed,
        "validation": validation_details,
        "code": result.code,
        "elapsed_ms": result.elapsed_ms,
    }


async def run_validation(
    problems: list[dict],
    endpoint: str | None = None,
    mock: bool = False,
    log_dir: str = "./runs/validation",
) -> ValidationReport:
    """Run validation on a set of problems.

    Args:
        problems: List of problem definitions
        endpoint: Modal endpoint URL (if not mock)
        mock: Use mock LLM
        log_dir: Directory for trajectory logs

    Returns:
        ValidationReport with all results
    """
    start_time = datetime.now()

    # Create runner
    runner = await create_runner(
        endpoint=endpoint,
        mock=mock,
        log_dir=log_dir,
        enable_trajectory_logging=True,
    )

    results = []
    passed = 0
    failed = 0

    for problem in problems:
        result = await validate_problem(runner, problem)
        results.append(result)

        if result["validation_passed"]:
            passed += 1
        else:
            failed += 1

    total_elapsed = (datetime.now() - start_time).total_seconds() * 1000

    # Compute aggregate stats from all trajectories
    all_records = []
    log_path = Path(log_dir)
    if log_path.exists():
        for trajectory_file in log_path.glob("*.jsonl"):
            try:
                records = load_trajectory(trajectory_file)
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"Failed to load trajectory {trajectory_file}: {e}")

    trajectory_stats = compute_trajectory_stats(all_records)
    tier_stats = compute_tier_stats(all_records)

    return ValidationReport(
        timestamp=start_time.isoformat(),
        mode="mock" if mock else "live",
        problems_tested=len(problems),
        problems_passed=passed,
        problems_failed=failed,
        pass_rate=passed / len(problems) if problems else 0,
        results=results,
        trajectory_stats=trajectory_stats,
        tier_stats=tier_stats,
        total_elapsed_ms=total_elapsed,
    )


def print_report(report: ValidationReport) -> None:
    """Print validation report to console."""
    print("\n" + "=" * 70)
    print("BASE MODEL VALIDATION REPORT (hi_moe-rj4)")
    print("=" * 70)

    print(f"\nMode: {report.mode}")
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Time: {report.total_elapsed_ms / 1000:.2f}s")

    print(f"\n{'─' * 50}")
    print(f"RESULTS: {report.problems_passed}/{report.problems_tested} problems passed ({report.pass_rate:.1%})")
    print(f"{'─' * 50}")

    for result in report.results:
        status = "✅" if result["validation_passed"] else "❌"
        val = result.get("validation", {})
        cases = f"{val.get('passed_cases', 0)}/{val.get('total_cases', 0)}"
        print(f"  {status} {result['problem_title']}: {cases} ({result['elapsed_ms']:.0f}ms)")

    if report.trajectory_stats.get("total_calls", 0) > 0:
        print(f"\n{'─' * 50}")
        print("TRAJECTORY STATS")
        print(f"{'─' * 50}")
        stats = report.trajectory_stats
        print(f"  Total LLM calls: {stats['total_calls']}")
        print(f"  Successful: {stats['successful_calls']}")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Avg latency: {stats['avg_latency_ms']:.0f}ms")

    if report.tier_stats.get("fleet_executions", 0) > 0:
        print(f"\n{'─' * 50}")
        print("TIER STATS")
        print(f"{'─' * 50}")
        tier = report.tier_stats
        print(f"  Architect decisions: {tier['architect_decisions']} ({tier['architect_success_rate']:.1%} success)")
        print(f"  Dispatcher routings: {tier['dispatcher_routings']} ({tier['dispatcher_success_rate']:.1%} success)")
        print(f"  Fleet executions: {tier['fleet_executions']} ({tier['fleet_success_rate']:.1%} success)")
        if tier.get("specialist_usage"):
            print(f"  Specialist usage: {tier['specialist_usage']}")

    print(f"\n{'=' * 70}")

    # Conclusion
    if report.pass_rate >= 0.8:
        print("✅ BASE MODEL VALIDATED: Architecture works without LoRAs!")
        print("   Proceed with base model + prompts for v0.1")
    elif report.pass_rate >= 0.5:
        print("⚠️  PARTIAL SUCCESS: Base model passes some problems")
        print("   Consider which specific capabilities need LoRAs")
    else:
        print("❌ BASE MODEL INSUFFICIENT: Consider adding LoRA adapters")
        print("   Review failed cases to identify capability gaps")

    print("=" * 70 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Validate base model performance (hi_moe-rj4)"
    )
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
        "--problems",
        nargs="+",
        type=int,
        help="Problem indices to test (default: all)",
    )
    parser.add_argument(
        "--easy-only",
        action="store_true",
        help="Only test easy problems",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs/validation",
        help="Directory for trajectory logs",
    )
    args = parser.parse_args()

    # Select problems
    if args.problems:
        problems = [ALL_PROBLEMS[i] for i in args.problems if i < len(ALL_PROBLEMS)]
    elif args.easy_only:
        problems = EASY_PROBLEMS
    else:
        problems = ALL_PROBLEMS

    if not problems:
        logger.error("No problems selected")
        sys.exit(1)

    logger.info(f"Selected {len(problems)} problems for validation")

    # Run validation
    report = await run_validation(
        problems=problems,
        endpoint=args.endpoint if not args.mock else None,
        mock=args.mock,
        log_dir=args.log_dir,
    )

    # Print report
    print_report(report)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        logger.info(f"Report saved to {output_path}")

    # Exit with appropriate code
    sys.exit(0 if report.pass_rate >= 0.8 else 1)


if __name__ == "__main__":
    asyncio.run(main())
