#!/usr/bin/env python3
"""Token efficiency audit tool (hi_moe-0a8).

Profile token usage across tiers to identify waste and optimize costs.

Usage:
    python -m e2e_test.token_audit                  # Full audit
    python -m e2e_test.token_audit --by-tier        # Breakdown by tier
    python -m e2e_test.token_audit --by-problem     # Breakdown by problem
    python -m e2e_test.token_audit --waste          # Identify waste patterns
    python -m e2e_test.token_audit --cost           # Cost analysis
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path


DEFAULT_DB = Path("runs/hi_moe.db")

# Cost per 1K tokens (Qwen3-32B estimate, adjust as needed)
COST_PER_1K_INPUT = 0.001   # $0.001 per 1K input tokens
COST_PER_1K_OUTPUT = 0.002  # $0.002 per 1K output tokens


def get_token_stats(db_path: Path) -> dict:
    """Get overall token statistics."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    row = conn.execute("""
        SELECT
            COUNT(*) as total_calls,
            SUM(tokens_in) as total_in,
            SUM(tokens_out) as total_out,
            AVG(tokens_in) as avg_in,
            AVG(tokens_out) as avg_out,
            MAX(tokens_in) as max_in,
            MAX(tokens_out) as max_out,
            AVG(latency_ms) as avg_latency
        FROM calls
    """).fetchone()
    conn.close()

    return dict(row) if row else {}


def get_stats_by_tier(db_path: Path) -> list[dict]:
    """Get token stats broken down by tier."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            tier,
            COUNT(*) as calls,
            SUM(tokens_in) as total_in,
            SUM(tokens_out) as total_out,
            AVG(tokens_in) as avg_in,
            AVG(tokens_out) as avg_out,
            AVG(latency_ms) as avg_latency,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
        FROM calls
        GROUP BY tier
        ORDER BY total_in + total_out DESC
    """).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_stats_by_specialist(db_path: Path) -> list[dict]:
    """Get token stats broken down by specialist."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            specialist,
            COUNT(*) as calls,
            SUM(tokens_in) as total_in,
            SUM(tokens_out) as total_out,
            AVG(tokens_in) as avg_in,
            AVG(tokens_out) as avg_out,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
        FROM calls
        WHERE specialist IS NOT NULL
        GROUP BY specialist
        ORDER BY total_in + total_out DESC
    """).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_stats_by_problem(db_path: Path) -> list[dict]:
    """Get token stats broken down by problem."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            problem_id,
            COUNT(*) as calls,
            SUM(tokens_in) as total_in,
            SUM(tokens_out) as total_out,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
        FROM calls
        GROUP BY problem_id
        ORDER BY total_in + total_out DESC
        LIMIT 20
    """).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_retry_waste(db_path: Path) -> dict:
    """Analyze token waste from retries."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get runs with multiple calls (indicates retries)
    rows = conn.execute("""
        SELECT
            run_id,
            problem_id,
            COUNT(*) as call_count,
            SUM(tokens_in) as total_in,
            SUM(tokens_out) as total_out,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
        FROM calls
        GROUP BY run_id
        HAVING call_count > 1
        ORDER BY total_in + total_out DESC
    """).fetchall()
    conn.close()

    total_retry_tokens = sum(r["total_in"] + r["total_out"] for r in rows)
    total_retry_calls = sum(r["call_count"] - 1 for r in rows)  # -1 for initial attempt

    return {
        "runs_with_retries": len(rows),
        "total_retry_calls": total_retry_calls,
        "total_retry_tokens": total_retry_tokens,
        "examples": [dict(r) for r in rows[:5]],
    }


def get_large_prompts(db_path: Path, threshold: int = 2000) -> list[dict]:
    """Find calls with unusually large input prompts."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            id,
            run_id,
            problem_id,
            tier,
            specialist,
            tokens_in,
            tokens_out,
            success
        FROM calls
        WHERE tokens_in > ?
        ORDER BY tokens_in DESC
        LIMIT 20
    """, (threshold,)).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_failed_call_waste(db_path: Path) -> dict:
    """Analyze tokens spent on failed calls."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    row = conn.execute("""
        SELECT
            COUNT(*) as failed_calls,
            SUM(tokens_in) as wasted_in,
            SUM(tokens_out) as wasted_out
        FROM calls
        WHERE success = 0
    """).fetchone()

    total_row = conn.execute("""
        SELECT SUM(tokens_in + tokens_out) as total FROM calls
    """).fetchone()
    conn.close()

    total = total_row["total"] or 1
    wasted = (row["wasted_in"] or 0) + (row["wasted_out"] or 0)

    return {
        "failed_calls": row["failed_calls"] or 0,
        "wasted_tokens": wasted,
        "waste_percentage": wasted / total * 100 if total > 0 else 0,
    }


def calculate_cost(tokens_in: int, tokens_out: int) -> float:
    """Calculate cost in dollars."""
    return (tokens_in / 1000 * COST_PER_1K_INPUT) + (tokens_out / 1000 * COST_PER_1K_OUTPUT)


def print_audit(db_path: Path, show_waste: bool = False, show_cost: bool = False):
    """Print token efficiency audit."""
    print("=" * 70)
    print("TOKEN EFFICIENCY AUDIT")
    print(f"Database: {db_path}")
    print("=" * 70)

    stats = get_token_stats(db_path)
    if not stats or not stats.get("total_calls"):
        print("\nNo call data found in database.")
        return

    total_in = stats["total_in"] or 0
    total_out = stats["total_out"] or 0
    total_tokens = total_in + total_out

    print(f"\n--- Overall Statistics ---")
    print(f"Total calls:        {stats['total_calls']:>10,}")
    print(f"Total tokens:       {total_tokens:>10,}")
    print(f"  Input tokens:     {total_in:>10,}")
    print(f"  Output tokens:    {total_out:>10,}")
    print(f"Avg tokens/call:    {(stats['avg_in'] or 0) + (stats['avg_out'] or 0):>10,.0f}")
    print(f"Max input:          {stats['max_in'] or 0:>10,}")
    print(f"Max output:         {stats['max_out'] or 0:>10,}")
    print(f"Avg latency:        {stats['avg_latency'] or 0:>10,.0f} ms")

    # By tier
    tier_stats = get_stats_by_tier(db_path)
    if tier_stats:
        print(f"\n--- By Tier ---")
        print(f"{'Tier':<15} {'Calls':>8} {'Tokens':>12} {'Avg':>8} {'Success':>8}")
        print("-" * 55)
        for t in tier_stats:
            tier = t["tier"] or "unknown"
            tokens = (t["total_in"] or 0) + (t["total_out"] or 0)
            avg = (t["avg_in"] or 0) + (t["avg_out"] or 0)
            success = t["successes"] or 0
            calls = t["calls"] or 0
            rate = success / calls * 100 if calls > 0 else 0
            print(f"{tier:<15} {calls:>8} {tokens:>12,} {avg:>8,.0f} {rate:>7.0f}%")

    # By specialist
    spec_stats = get_stats_by_specialist(db_path)
    if spec_stats:
        print(f"\n--- By Specialist ---")
        print(f"{'Specialist':<15} {'Calls':>8} {'Tokens':>12} {'Avg':>8} {'Success':>8}")
        print("-" * 55)
        for s in spec_stats:
            spec = s["specialist"] or "unknown"
            tokens = (s["total_in"] or 0) + (s["total_out"] or 0)
            avg = (s["avg_in"] or 0) + (s["avg_out"] or 0)
            success = s["successes"] or 0
            calls = s["calls"] or 0
            rate = success / calls * 100 if calls > 0 else 0
            print(f"{spec:<15} {calls:>8} {tokens:>12,} {avg:>8,.0f} {rate:>7.0f}%")

    # Waste analysis
    if show_waste:
        print(f"\n--- Waste Analysis ---")

        # Failed calls
        failed = get_failed_call_waste(db_path)
        print(f"\nFailed call waste:")
        print(f"  Failed calls:     {failed['failed_calls']:>10,}")
        print(f"  Wasted tokens:    {failed['wasted_tokens']:>10,}")
        print(f"  Waste %:          {failed['waste_percentage']:>10.1f}%")

        # Retry waste
        retry = get_retry_waste(db_path)
        print(f"\nRetry waste:")
        print(f"  Runs w/ retries:  {retry['runs_with_retries']:>10,}")
        print(f"  Extra calls:      {retry['total_retry_calls']:>10,}")
        print(f"  Retry tokens:     {retry['total_retry_tokens']:>10,}")

        # Large prompts
        large = get_large_prompts(db_path)
        if large:
            print(f"\nLarge prompts (>2000 tokens):")
            for p in large[:5]:
                print(f"  {p['problem_id'][:25]:<25} {p['tokens_in']:>6} in, tier={p['tier']}")

    # Cost analysis
    if show_cost:
        print(f"\n--- Cost Analysis ---")
        total_cost = calculate_cost(total_in, total_out)
        print(f"Total cost:         ${total_cost:>10.4f}")
        print(f"Cost per call:      ${total_cost / stats['total_calls']:>10.6f}")

        # Cost by tier
        if tier_stats:
            print(f"\nCost by tier:")
            for t in tier_stats:
                tier = t["tier"] or "unknown"
                cost = calculate_cost(t["total_in"] or 0, t["total_out"] or 0)
                print(f"  {tier:<15} ${cost:.4f}")

        # Top expensive problems
        prob_stats = get_stats_by_problem(db_path)
        if prob_stats:
            print(f"\nMost expensive problems:")
            for p in prob_stats[:5]:
                cost = calculate_cost(p["total_in"] or 0, p["total_out"] or 0)
                success = "✓" if p["successes"] else "✗"
                print(f"  {success} {p['problem_id'][:30]:<30} ${cost:.4f} ({p['calls']} calls)")

    # Recommendations
    print(f"\n--- Recommendations ---")
    recommendations = []

    failed = get_failed_call_waste(db_path)
    if failed["waste_percentage"] > 20:
        recommendations.append(
            f"High failure waste ({failed['waste_percentage']:.0f}%). "
            "Consider improving prompts or adding validation."
        )

    retry = get_retry_waste(db_path)
    if retry["runs_with_retries"] > 0:
        avg_retries = retry["total_retry_calls"] / retry["runs_with_retries"]
        if avg_retries > 2:
            recommendations.append(
                f"High retry rate ({avg_retries:.1f} avg). "
                "Review error patterns and specialist selection."
            )

    large = get_large_prompts(db_path, 3000)
    if large:
        recommendations.append(
            f"{len(large)} calls with >3000 input tokens. "
            "Consider prompt compression or context trimming."
        )

    if stats["avg_out"] and stats["avg_out"] > 1500:
        recommendations.append(
            f"High avg output ({stats['avg_out']:.0f} tokens). "
            "Consider reducing max_tokens or adding stop sequences."
        )

    if not recommendations:
        recommendations.append("No major inefficiencies detected. Good token hygiene!")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Token efficiency audit")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Database path")
    parser.add_argument("--waste", action="store_true", help="Show waste analysis")
    parser.add_argument("--cost", action="store_true", help="Show cost analysis")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--full", action="store_true", help="Show all analyses")
    args = parser.parse_args()

    if args.full:
        args.waste = True
        args.cost = True

    if args.json:
        import json
        data = {
            "stats": get_token_stats(args.db),
            "by_tier": get_stats_by_tier(args.db),
            "by_specialist": get_stats_by_specialist(args.db),
            "by_problem": get_stats_by_problem(args.db),
            "retry_waste": get_retry_waste(args.db),
            "failed_waste": get_failed_call_waste(args.db),
            "large_prompts": get_large_prompts(args.db),
        }
        print(json.dumps(data, indent=2, default=str))
    else:
        print_audit(args.db, args.waste, args.cost)


if __name__ == "__main__":
    main()
