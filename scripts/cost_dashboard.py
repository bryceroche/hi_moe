#!/usr/bin/env python3
"""Cost tracking dashboard for hi_moe runs (hi_moe-iv9).

Usage:
    python scripts/cost_dashboard.py                    # Show all runs
    python scripts/cost_dashboard.py --runs-dir runs/   # Specific directory
    python scripts/cost_dashboard.py --by-tier          # Group by tier
    python scripts/cost_dashboard.py --by-problem       # Group by problem
    python scripts/cost_dashboard.py --top 10           # Top N expensive runs
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# Token pricing (approximate, adjust as needed)
# These are example rates - update based on actual provider pricing
TOKEN_RATES = {
    "input": 0.0001 / 1000,   # $0.0001 per 1K input tokens
    "output": 0.0003 / 1000,  # $0.0003 per 1K output tokens
}


@dataclass
class RunStats:
    """Statistics for a single run."""
    run_id: str
    problem_id: str
    timestamp: str
    tokens_in: int = 0
    tokens_out: int = 0
    num_calls: int = 0
    latency_ms: int = 0
    status: str = "unknown"
    tiers: dict = None

    def __post_init__(self):
        if self.tiers is None:
            self.tiers = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0, "calls": 0})

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def estimated_cost(self) -> float:
        return (self.tokens_in * TOKEN_RATES["input"] +
                self.tokens_out * TOKEN_RATES["output"])


def parse_trajectory(jsonl_path: Path) -> RunStats | None:
    """Parse a trajectory file and extract stats."""
    stats = None
    tiers = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0, "calls": 0})

    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")

                if entry_type == "run_start":
                    stats = RunStats(
                        run_id=entry.get("run_id", ""),
                        problem_id=entry.get("problem_id", ""),
                        timestamp=entry.get("ts", ""),
                    )

                elif entry_type == "vllm_call" and stats:
                    tokens_in = entry.get("tokens_in", 0)
                    tokens_out = entry.get("tokens_out", 0)
                    latency = entry.get("latency_ms", 0)

                    stats.tokens_in += tokens_in
                    stats.tokens_out += tokens_out
                    stats.num_calls += 1
                    stats.latency_ms += latency

                    # Classify tier from system prompt
                    tier = classify_tier(entry.get("input", []))
                    tiers[tier]["tokens_in"] += tokens_in
                    tiers[tier]["tokens_out"] += tokens_out
                    tiers[tier]["calls"] += 1

                elif entry_type == "run_end" and stats:
                    stats.status = entry.get("status", "unknown")

        if stats:
            stats.tiers = dict(tiers)
        return stats

    except Exception as e:
        print(f"Error parsing {jsonl_path}: {e}")
        return None


def classify_tier(messages: list) -> str:
    """Classify which tier a call belongs to based on system prompt."""
    if not isinstance(messages, list):
        return "unknown"

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "").lower()
            if "dispatcher" in content or "json" in content:
                return "dispatcher"
            elif "planner" in content:
                return "architect"
            elif "expert" in content:
                return "fleet"
    return "unknown"


def collect_stats(runs_dir: Path) -> list[RunStats]:
    """Collect stats from all trajectory files."""
    stats = []
    for jsonl_file in runs_dir.rglob("run-*.jsonl"):
        run_stats = parse_trajectory(jsonl_file)
        if run_stats:
            stats.append(run_stats)
    return stats


def format_cost(cost: float) -> str:
    """Format cost for display."""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def print_summary(stats: list[RunStats]) -> None:
    """Print overall summary."""
    total_in = sum(s.tokens_in for s in stats)
    total_out = sum(s.tokens_out for s in stats)
    total_cost = sum(s.estimated_cost for s in stats)
    total_calls = sum(s.num_calls for s in stats)
    successful = sum(1 for s in stats if s.status == "completed")

    print("=" * 60)
    print("COST DASHBOARD SUMMARY")
    print("=" * 60)
    print(f"Total runs:        {len(stats)}")
    print(f"Successful:        {successful} ({100*successful/len(stats):.0f}%)" if stats else "")
    print(f"Total LLM calls:   {total_calls}")
    print(f"Total tokens:      {total_in + total_out:,}")
    print(f"  Input:           {total_in:,}")
    print(f"  Output:          {total_out:,}")
    print(f"Estimated cost:    {format_cost(total_cost)}")
    if stats:
        print(f"Avg cost/run:      {format_cost(total_cost/len(stats))}")
        print(f"Avg tokens/call:   {(total_in + total_out) // total_calls if total_calls else 0}")


def print_by_tier(stats: list[RunStats]) -> None:
    """Print breakdown by tier."""
    tier_totals = defaultdict(lambda: {"tokens_in": 0, "tokens_out": 0, "calls": 0})

    for s in stats:
        for tier, data in s.tiers.items():
            tier_totals[tier]["tokens_in"] += data["tokens_in"]
            tier_totals[tier]["tokens_out"] += data["tokens_out"]
            tier_totals[tier]["calls"] += data["calls"]

    print("\n" + "=" * 60)
    print("BREAKDOWN BY TIER")
    print("=" * 60)
    print(f"{'Tier':<15} {'Calls':>8} {'Tokens In':>12} {'Tokens Out':>12} {'Cost':>10}")
    print("-" * 60)

    for tier in ["architect", "dispatcher", "fleet", "unknown"]:
        if tier in tier_totals:
            data = tier_totals[tier]
            cost = (data["tokens_in"] * TOKEN_RATES["input"] +
                   data["tokens_out"] * TOKEN_RATES["output"])
            print(f"{tier:<15} {data['calls']:>8} {data['tokens_in']:>12,} {data['tokens_out']:>12,} {format_cost(cost):>10}")


def print_by_problem(stats: list[RunStats]) -> None:
    """Print breakdown by problem."""
    problem_stats = defaultdict(lambda: {"runs": 0, "tokens": 0, "cost": 0, "success": 0})

    for s in stats:
        problem_stats[s.problem_id]["runs"] += 1
        problem_stats[s.problem_id]["tokens"] += s.total_tokens
        problem_stats[s.problem_id]["cost"] += s.estimated_cost
        if s.status == "completed":
            problem_stats[s.problem_id]["success"] += 1

    print("\n" + "=" * 60)
    print("BREAKDOWN BY PROBLEM")
    print("=" * 60)
    print(f"{'Problem':<25} {'Runs':>6} {'Pass%':>8} {'Tokens':>12} {'Cost':>10}")
    print("-" * 60)

    for problem, data in sorted(problem_stats.items(), key=lambda x: -x[1]["cost"]):
        pass_rate = 100 * data["success"] / data["runs"] if data["runs"] else 0
        print(f"{problem:<25} {data['runs']:>6} {pass_rate:>7.0f}% {data['tokens']:>12,} {format_cost(data['cost']):>10}")


def print_top_expensive(stats: list[RunStats], n: int = 10) -> None:
    """Print top N most expensive runs."""
    sorted_stats = sorted(stats, key=lambda s: -s.estimated_cost)[:n]

    print("\n" + "=" * 60)
    print(f"TOP {n} MOST EXPENSIVE RUNS")
    print("=" * 60)
    print(f"{'Problem':<25} {'Calls':>6} {'Tokens':>10} {'Cost':>10} {'Status':<10}")
    print("-" * 60)

    for s in sorted_stats:
        status_icon = "✓" if s.status == "completed" else "✗"
        print(f"{s.problem_id:<25} {s.num_calls:>6} {s.total_tokens:>10,} {format_cost(s.estimated_cost):>10} {status_icon} {s.status:<8}")


def main():
    parser = argparse.ArgumentParser(description="Cost tracking dashboard for hi_moe")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing trajectory files (default: runs/)",
    )
    parser.add_argument(
        "--by-tier",
        action="store_true",
        help="Show breakdown by tier",
    )
    parser.add_argument(
        "--by-problem",
        action="store_true",
        help="Show breakdown by problem",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show top N most expensive runs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all breakdowns",
    )
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"Runs directory not found: {args.runs_dir}")
        return 1

    stats = collect_stats(args.runs_dir)

    if not stats:
        print("No trajectory files found.")
        return 1

    print_summary(stats)

    if args.all or args.by_tier:
        print_by_tier(stats)

    if args.all or args.by_problem:
        print_by_problem(stats)

    if args.all or args.top:
        print_top_expensive(stats, args.top or 10)

    return 0


if __name__ == "__main__":
    exit(main())
