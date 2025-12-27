#!/usr/bin/env python3
"""Aggregate stress test results across multiple runs/instances (hi_moe-cge).

Usage:
    python -m e2e_test.aggregate_results [--dir runs/stress_test]
    python -m e2e_test.aggregate_results --all  # Include all runs/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RunSummary:
    """Summary of a single run."""
    run_id: str
    problem_id: str
    status: str  # completed, failed, timeout
    elapsed_ms: int
    total_calls: int
    tokens_in: int = 0
    tokens_out: int = 0
    tiers_used: set[str] = field(default_factory=set)
    specialist: str | None = None
    error: str | None = None
    timestamp: str = ""

    @property
    def success(self) -> bool:
        return self.status == "completed"


def parse_run_file(path: Path) -> RunSummary | None:
    """Parse a JSONL trajectory log into a summary."""
    run_id = None
    problem_id = None
    status = "unknown"
    elapsed_ms = 0
    total_calls = 0
    tokens_in = 0
    tokens_out = 0
    tiers_used: set[str] = set()
    specialist = None
    error = None
    timestamp = ""

    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type", "")

                if entry_type == "run_start":
                    run_id = entry.get("run_id")
                    problem_id = entry.get("problem_id")
                    timestamp = entry.get("ts", "")

                elif entry_type == "vllm_call":
                    total_calls += 1
                    tokens_in += entry.get("tokens_in", 0)
                    tokens_out += entry.get("tokens_out", 0)
                    tier = entry.get("tier")
                    if tier and tier != "unknown":
                        tiers_used.add(tier)

                elif entry_type == "fleet_execution":
                    spec = entry.get("specialist")
                    if spec:
                        specialist = spec
                    exec_status = entry.get("status")
                    if exec_status == "success":
                        status = "completed"

                elif entry_type == "run_end":
                    status = entry.get("status", status)
                    elapsed_ms = entry.get("elapsed_ms", 0)
                    error = entry.get("error")
                    if entry.get("total_calls"):
                        total_calls = entry["total_calls"]

        if not run_id:
            run_id = path.stem
        if not problem_id:
            # Extract from filename: run-problem_id-hash.jsonl
            parts = path.stem.split("-")
            if len(parts) >= 2:
                problem_id = "-".join(parts[1:-1])  # Handle multi-word problem ids

        return RunSummary(
            run_id=run_id,
            problem_id=problem_id or "unknown",
            status=status,
            elapsed_ms=int(elapsed_ms),
            total_calls=total_calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tiers_used=tiers_used,
            specialist=specialist,
            error=error,
            timestamp=timestamp,
        )
    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None


@dataclass
class AggregateStats:
    """Aggregated statistics across runs."""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0

    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_calls: int = 0
    total_time_ms: int = 0

    by_problem: dict[str, dict] = field(default_factory=dict)
    by_specialist: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_tier: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    runs: list[RunSummary] = field(default_factory=list)


def aggregate(runs_dir: Path, include_subdirs: bool = False) -> AggregateStats:
    """Aggregate all runs in a directory."""
    stats = AggregateStats()

    # Find all JSONL files
    pattern = "**/*.jsonl" if include_subdirs else "*.jsonl"
    files = list(runs_dir.glob(pattern))

    for f in sorted(files):
        summary = parse_run_file(f)
        if not summary:
            continue

        stats.runs.append(summary)
        stats.total_runs += 1

        if summary.success:
            stats.successful_runs += 1
        else:
            stats.failed_runs += 1

        stats.total_tokens_in += summary.tokens_in
        stats.total_tokens_out += summary.tokens_out
        stats.total_calls += summary.total_calls
        stats.total_time_ms += summary.elapsed_ms

        # By problem
        if summary.problem_id not in stats.by_problem:
            stats.by_problem[summary.problem_id] = {
                "runs": 0, "successes": 0, "total_time_ms": 0
            }
        stats.by_problem[summary.problem_id]["runs"] += 1
        if summary.success:
            stats.by_problem[summary.problem_id]["successes"] += 1
        stats.by_problem[summary.problem_id]["total_time_ms"] += summary.elapsed_ms

        # By specialist
        if summary.specialist:
            stats.by_specialist[summary.specialist] += 1

        # By tier
        for tier in summary.tiers_used:
            stats.by_tier[tier] += 1

    return stats


def format_report(stats: AggregateStats) -> str:
    """Generate human-readable report."""
    lines = [
        "=" * 70,
        "AGGREGATED STRESS TEST RESULTS",
        "=" * 70,
        "",
        f"Total runs:     {stats.total_runs}",
        f"Successful:     {stats.successful_runs} ({stats.successful_runs/max(1,stats.total_runs):.1%})",
        f"Failed:         {stats.failed_runs}",
        "",
        f"Total LLM calls: {stats.total_calls}",
        f"Total tokens:    {stats.total_tokens_in + stats.total_tokens_out:,} "
        f"(in: {stats.total_tokens_in:,}, out: {stats.total_tokens_out:,})",
        f"Total time:      {stats.total_time_ms/1000:.1f}s",
        "",
    ]

    # By problem
    if stats.by_problem:
        lines.append("## By Problem")
        for prob, data in sorted(stats.by_problem.items()):
            rate = data["successes"] / max(1, data["runs"])
            avg_time = data["total_time_ms"] / max(1, data["runs"])
            lines.append(
                f"  {prob:30s} {data['successes']}/{data['runs']} ({rate:.0%}) "
                f"avg {avg_time/1000:.1f}s"
            )
        lines.append("")

    # By specialist
    if stats.by_specialist:
        lines.append("## By Specialist")
        for spec, count in sorted(stats.by_specialist.items(), key=lambda x: -x[1]):
            lines.append(f"  {spec:20s} {count}")
        lines.append("")

    # By tier
    if stats.by_tier:
        lines.append("## By Tier")
        for tier, count in sorted(stats.by_tier.items(), key=lambda x: -x[1]):
            lines.append(f"  {tier:20s} {count}")
        lines.append("")

    # Recent runs
    lines.append("## Recent Runs (last 10)")
    for run in sorted(stats.runs, key=lambda r: r.timestamp, reverse=True)[:10]:
        status_icon = "✓" if run.success else "✗"
        ts = run.timestamp.split("T")[1][:8] if "T" in run.timestamp else ""
        lines.append(
            f"  {status_icon} {run.problem_id:25s} {run.elapsed_ms/1000:5.1f}s "
            f"{run.tokens_out:5d} tok  {ts}"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate stress test results")
    parser.add_argument(
        "--dir", "-d",
        type=Path,
        default=Path("runs/stress_test"),
        help="Directory containing JSONL trajectory logs"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Include all runs/ subdirectories"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON instead of text"
    )
    args = parser.parse_args()

    if args.all:
        runs_dir = Path("runs")
        include_subdirs = True
    else:
        runs_dir = args.dir
        include_subdirs = False

    if not runs_dir.exists():
        print(f"Directory not found: {runs_dir}")
        return 1

    stats = aggregate(runs_dir, include_subdirs)

    if args.json:
        output = {
            "total_runs": stats.total_runs,
            "successful_runs": stats.successful_runs,
            "failed_runs": stats.failed_runs,
            "success_rate": stats.successful_runs / max(1, stats.total_runs),
            "total_tokens": stats.total_tokens_in + stats.total_tokens_out,
            "total_time_ms": stats.total_time_ms,
            "by_problem": stats.by_problem,
            "by_specialist": dict(stats.by_specialist),
            "by_tier": dict(stats.by_tier),
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(stats))

    return 0


if __name__ == "__main__":
    exit(main())
