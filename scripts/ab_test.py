#!/usr/bin/env python3
"""A/B testing framework for hi_moe prompt experiments (hi_moe-3g6).

Usage:
    python scripts/ab_test.py --variant-a default --variant-b optimized
    python scripts/ab_test.py --variant-a default --variant-b optimized --problems 0,1,2
    python scripts/ab_test.py --report results/ab_test_*.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class VariantResult:
    """Result from running a variant on a problem."""
    variant: str
    problem_idx: int
    passed: bool
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    num_calls: int = 0
    error: str | None = None


@dataclass
class ABTestResult:
    """Complete A/B test result."""
    timestamp: str
    variant_a: str
    variant_b: str
    problems: list[int]
    results_a: list[VariantResult] = field(default_factory=list)
    results_b: list[VariantResult] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        def variant_stats(results: list[VariantResult]) -> dict:
            passed = sum(1 for r in results if r.passed)
            total_tokens = sum(r.tokens_in + r.tokens_out for r in results)
            total_latency = sum(r.latency_ms for r in results)
            total_calls = sum(r.num_calls for r in results)
            return {
                "pass_rate": passed / len(results) if results else 0,
                "passed": passed,
                "total": len(results),
                "total_tokens": total_tokens,
                "avg_tokens": total_tokens // len(results) if results else 0,
                "total_latency_ms": total_latency,
                "avg_latency_ms": total_latency // len(results) if results else 0,
                "total_calls": total_calls,
                "avg_calls": total_calls / len(results) if results else 0,
            }

        stats_a = variant_stats(self.results_a)
        stats_b = variant_stats(self.results_b)

        # Calculate improvements (positive = B is better)
        token_reduction = 0
        latency_reduction = 0
        if stats_a["total_tokens"] > 0:
            token_reduction = (stats_a["total_tokens"] - stats_b["total_tokens"]) / stats_a["total_tokens"]
        if stats_a["total_latency_ms"] > 0:
            latency_reduction = (stats_a["total_latency_ms"] - stats_b["total_latency_ms"]) / stats_a["total_latency_ms"]

        return {
            "variant_a": {"name": self.variant_a, **stats_a},
            "variant_b": {"name": self.variant_b, **stats_b},
            "improvements": {
                "token_reduction_pct": token_reduction * 100,
                "latency_reduction_pct": latency_reduction * 100,
                "pass_rate_diff": stats_b["pass_rate"] - stats_a["pass_rate"],
            },
            "winner": self._determine_winner(stats_a, stats_b, token_reduction),
        }

    def _determine_winner(self, stats_a: dict, stats_b: dict, token_reduction: float) -> str:
        """Determine which variant wins based on metrics."""
        # Prioritize: 1) pass rate, 2) token efficiency
        if stats_b["pass_rate"] > stats_a["pass_rate"]:
            return self.variant_b
        elif stats_a["pass_rate"] > stats_b["pass_rate"]:
            return self.variant_a
        elif token_reduction > 0.1:  # >10% token reduction
            return self.variant_b
        elif token_reduction < -0.1:
            return self.variant_a
        return "tie"


# Prompt variants - add new variants here
PROMPT_VARIANTS = {
    "default": {
        "description": "Original prompts (baseline)",
        "env": {},  # No overrides, use defaults
    },
    "optimized": {
        "description": "Token-optimized prompts (hi_moe-eet)",
        "env": {"HI_MOE_PROMPT_VARIANT": "optimized"},
    },
    "minimal": {
        "description": "Minimal prompts (extreme compression)",
        "env": {"HI_MOE_PROMPT_VARIANT": "minimal"},
    },
    "verbose": {
        "description": "Verbose prompts (maximum context)",
        "env": {"HI_MOE_PROMPT_VARIANT": "verbose"},
    },
}


def run_variant(
    variant: str,
    problem_idx: int,
    log_dir: Path,
) -> VariantResult:
    """Run a single problem with a specific variant."""
    variant_config = PROMPT_VARIANTS.get(variant, PROMPT_VARIANTS["default"])

    # Set up environment with variant overrides
    env = os.environ.copy()
    env.update(variant_config["env"])

    start = time.time()
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "e2e_test.run_e2e",
                "--problem", str(problem_idx),
                "--log-dir", str(log_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            env=env,
        )
        elapsed = time.time() - start

        passed = "All" in result.stdout and "passed" in result.stdout
        error = None
        if result.returncode != 0:
            error = result.stderr[-500:] if result.stderr else "Unknown error"

        # Parse tokens from trajectory
        tokens_in, tokens_out, num_calls = parse_trajectory_tokens(log_dir, problem_idx)

        return VariantResult(
            variant=variant,
            problem_idx=problem_idx,
            passed=passed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int(elapsed * 1000),
            num_calls=num_calls,
            error=error,
        )

    except subprocess.TimeoutExpired:
        return VariantResult(
            variant=variant,
            problem_idx=problem_idx,
            passed=False,
            latency_ms=300000,
            error="Timeout (>5 min)",
        )
    except Exception as e:
        return VariantResult(
            variant=variant,
            problem_idx=problem_idx,
            passed=False,
            error=str(e),
        )


def parse_trajectory_tokens(log_dir: Path, problem_idx: int) -> tuple[int, int, int]:
    """Parse token usage from trajectory files."""
    tokens_in, tokens_out, num_calls = 0, 0, 0

    # Find latest trajectory file for this problem
    for jsonl in sorted(log_dir.glob(f"run-*.jsonl"), reverse=True):
        try:
            with open(jsonl) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "vllm_call":
                            tokens_in += entry.get("tokens_in", 0)
                            tokens_out += entry.get("tokens_out", 0)
                            num_calls += 1
                    except json.JSONDecodeError:
                        continue
            break  # Only process first (latest) file
        except Exception:
            continue

    return tokens_in, tokens_out, num_calls


def run_ab_test(
    variant_a: str,
    variant_b: str,
    problems: list[int],
    base_log_dir: Path,
) -> ABTestResult:
    """Run A/B test comparing two variants."""
    result = ABTestResult(
        timestamp=datetime.now().isoformat(),
        variant_a=variant_a,
        variant_b=variant_b,
        problems=problems,
    )

    for problem_idx in problems:
        print(f"\n--- Problem {problem_idx} ---")

        # Run variant A
        log_dir_a = base_log_dir / f"variant_{variant_a}"
        log_dir_a.mkdir(parents=True, exist_ok=True)
        print(f"  Running variant A ({variant_a})...")
        result_a = run_variant(variant_a, problem_idx, log_dir_a)
        status_a = "PASS" if result_a.passed else "FAIL"
        print(f"    {status_a}: {result_a.tokens_in + result_a.tokens_out} tokens, {result_a.latency_ms}ms")
        result.results_a.append(result_a)

        # Run variant B
        log_dir_b = base_log_dir / f"variant_{variant_b}"
        log_dir_b.mkdir(parents=True, exist_ok=True)
        print(f"  Running variant B ({variant_b})...")
        result_b = run_variant(variant_b, problem_idx, log_dir_b)
        status_b = "PASS" if result_b.passed else "FAIL"
        print(f"    {status_b}: {result_b.tokens_in + result_b.tokens_out} tokens, {result_b.latency_ms}ms")
        result.results_b.append(result_b)

    return result


def print_report(result: ABTestResult) -> None:
    """Print A/B test report."""
    summary = result.summary()

    print("\n" + "=" * 70)
    print("A/B TEST REPORT")
    print("=" * 70)
    print(f"Timestamp: {result.timestamp}")
    print(f"Problems tested: {result.problems}")

    print("\n" + "-" * 70)
    print("VARIANT COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<25} {'A (' + result.variant_a + ')':<20} {'B (' + result.variant_b + ')':<20}")
    print("-" * 70)

    va = summary["variant_a"]
    vb = summary["variant_b"]

    print(f"{'Pass Rate':<25} {va['pass_rate']*100:>6.0f}% ({va['passed']}/{va['total']}){'':<6} {vb['pass_rate']*100:>6.0f}% ({vb['passed']}/{vb['total']})")
    print(f"{'Total Tokens':<25} {va['total_tokens']:>12,}{'':<6} {vb['total_tokens']:>12,}")
    print(f"{'Avg Tokens/Problem':<25} {va['avg_tokens']:>12,}{'':<6} {vb['avg_tokens']:>12,}")
    print(f"{'Total Latency (ms)':<25} {va['total_latency_ms']:>12,}{'':<6} {vb['total_latency_ms']:>12,}")
    print(f"{'Avg Latency/Problem (ms)':<25} {va['avg_latency_ms']:>12,}{'':<6} {vb['avg_latency_ms']:>12,}")
    print(f"{'LLM Calls':<25} {va['total_calls']:>12}{'':<6} {vb['total_calls']:>12}")

    print("\n" + "-" * 70)
    print("IMPROVEMENTS (B vs A)")
    print("-" * 70)
    imp = summary["improvements"]

    token_arrow = "v" if imp["token_reduction_pct"] > 0 else "^"
    latency_arrow = "v" if imp["latency_reduction_pct"] > 0 else "^"

    print(f"Token Reduction:   {imp['token_reduction_pct']:+.1f}% {token_arrow}")
    print(f"Latency Reduction: {imp['latency_reduction_pct']:+.1f}% {latency_arrow}")
    print(f"Pass Rate Diff:    {imp['pass_rate_diff']*100:+.1f}%")

    print("\n" + "-" * 70)
    winner = summary["winner"]
    if winner == "tie":
        print("RESULT: TIE - No significant difference")
    else:
        print(f"WINNER: {winner}")
    print("-" * 70)


def save_report(result: ABTestResult, path: Path) -> None:
    """Save A/B test report to JSON."""
    summary = result.summary()

    with open(path, "w") as f:
        json.dump({
            "timestamp": result.timestamp,
            "variant_a": result.variant_a,
            "variant_b": result.variant_b,
            "problems": result.problems,
            "summary": summary,
            "results_a": [
                {
                    "problem_idx": r.problem_idx,
                    "passed": r.passed,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "latency_ms": r.latency_ms,
                    "num_calls": r.num_calls,
                    "error": r.error,
                }
                for r in result.results_a
            ],
            "results_b": [
                {
                    "problem_idx": r.problem_idx,
                    "passed": r.passed,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    "latency_ms": r.latency_ms,
                    "num_calls": r.num_calls,
                    "error": r.error,
                }
                for r in result.results_b
            ],
        }, f, indent=2)
    print(f"\nReport saved to {path}")


def load_and_display_report(path: Path) -> None:
    """Load and display an existing report."""
    with open(path) as f:
        data = json.load(f)

    result = ABTestResult(
        timestamp=data["timestamp"],
        variant_a=data["variant_a"],
        variant_b=data["variant_b"],
        problems=data["problems"],
        results_a=[VariantResult(variant=data["variant_a"], **r) for r in data["results_a"]],
        results_b=[VariantResult(variant=data["variant_b"], **r) for r in data["results_b"]],
    )
    print_report(result)


def main():
    parser = argparse.ArgumentParser(description="A/B testing framework for hi_moe prompts")
    parser.add_argument(
        "--variant-a",
        default="default",
        choices=list(PROMPT_VARIANTS.keys()),
        help="First variant (baseline)",
    )
    parser.add_argument(
        "--variant-b",
        default="optimized",
        choices=list(PROMPT_VARIANTS.keys()),
        help="Second variant (experiment)",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default="0,1,2",
        help="Comma-separated list of problem indices to test",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("runs/ab_test"),
        help="Directory for test logs",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Load and display existing report",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available prompt variants",
    )
    args = parser.parse_args()

    if args.list_variants:
        print("Available prompt variants:")
        print("-" * 50)
        for name, config in PROMPT_VARIANTS.items():
            print(f"  {name:<15} {config['description']}")
        return 0

    if args.report:
        load_and_display_report(args.report)
        return 0

    problems = [int(p.strip()) for p in args.problems.split(",")]

    print(f"A/B Test: {args.variant_a} vs {args.variant_b}")
    print(f"Testing {len(problems)} problems: {problems}")
    print("=" * 70)

    result = run_ab_test(
        variant_a=args.variant_a,
        variant_b=args.variant_b,
        problems=problems,
        base_log_dir=args.log_dir,
    )

    print_report(result)

    # Save report
    report_path = args.log_dir / f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    args.log_dir.mkdir(parents=True, exist_ok=True)
    save_report(result, report_path)

    # Return 0 if variant B wins or ties, 1 if variant A is better
    summary = result.summary()
    return 0 if summary["winner"] in [args.variant_b, "tie"] else 1


if __name__ == "__main__":
    exit(main())
