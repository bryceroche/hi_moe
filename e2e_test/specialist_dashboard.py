#!/usr/bin/env python3
"""Specialist performance dashboard (hi_moe-e1b).

Visualizes which specialists succeed for which problem types using
the routing_decisions table from call_db.

Usage:
    python -m e2e_test.specialist_dashboard
    python -m e2e_test.specialist_dashboard --db runs/hi_moe.db
    python -m e2e_test.specialist_dashboard --specialist python
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


DEFAULT_DB = Path("runs/hi_moe.db")


def get_specialist_stats(db_path: Path) -> dict:
    """Get success/failure stats per specialist."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Query routing decisions
    rows = conn.execute("""
        SELECT
            selected_specialist,
            decision_correct,
            COUNT(*) as count
        FROM routing_decisions
        WHERE decision_correct IS NOT NULL
        GROUP BY selected_specialist, decision_correct
    """).fetchall()
    conn.close()

    stats = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})
    for row in rows:
        spec = row["selected_specialist"]
        if row["decision_correct"]:
            stats[spec]["correct"] += row["count"]
        else:
            stats[spec]["incorrect"] += row["count"]
        stats[spec]["total"] += row["count"]

    return dict(stats)


def get_problem_breakdown(db_path: Path) -> dict:
    """Get success rates by problem type/keywords."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            problem_id,
            selected_specialist,
            decision_correct,
            problem_keywords
        FROM routing_decisions
        WHERE decision_correct IS NOT NULL
    """).fetchall()
    conn.close()

    # Group by problem
    problems = defaultdict(lambda: {"attempts": 0, "successes": 0, "specialists": set()})
    for row in rows:
        pid = row["problem_id"]
        problems[pid]["attempts"] += 1
        if row["decision_correct"]:
            problems[pid]["successes"] += 1
        problems[pid]["specialists"].add(row["selected_specialist"])

    # Convert sets to lists for display
    for pid in problems:
        problems[pid]["specialists"] = list(problems[pid]["specialists"])

    return dict(problems)


def get_routing_patterns(db_path: Path) -> list[dict]:
    """Get recent routing patterns."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            problem_id,
            selected_specialist,
            confidence,
            decision_correct,
            created_at
        FROM routing_decisions
        ORDER BY created_at DESC
        LIMIT 20
    """).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_failure_analysis(db_path: Path) -> dict:
    """Analyze common failure patterns."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get failures with the specialist that was tried
    rows = conn.execute("""
        SELECT
            selected_specialist,
            problem_id,
            actual_specialist_needed
        FROM routing_decisions
        WHERE decision_correct = 0
    """).fetchall()
    conn.close()

    failures = defaultdict(list)
    for row in rows:
        failures[row["selected_specialist"]].append({
            "problem": row["problem_id"],
            "needed": row["actual_specialist_needed"],
        })

    return dict(failures)


def print_dashboard(db_path: Path, specialist_filter: str | None = None):
    """Print the specialist performance dashboard."""
    print("=" * 60)
    print("SPECIALIST PERFORMANCE DASHBOARD")
    print(f"Database: {db_path}")
    print("=" * 60)

    # Specialist success rates
    stats = get_specialist_stats(db_path)
    if not stats:
        print("\nNo routing data found. Run some problems first!")
        return

    print("\n--- Specialist Success Rates ---")
    print(f"{'Specialist':<15} {'Success':<10} {'Total':<8} {'Rate':<10}")
    print("-" * 45)

    for spec in sorted(stats.keys()):
        if specialist_filter and spec != specialist_filter:
            continue
        s = stats[spec]
        rate = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        bar = "█" * int(rate / 10) + "░" * (10 - int(rate / 10))
        print(f"{spec:<15} {s['correct']:<10} {s['total']:<8} {rate:>5.1f}% {bar}")

    # Problem breakdown
    problems = get_problem_breakdown(db_path)
    if problems:
        print("\n--- Problem Success Rates ---")
        print(f"{'Problem':<25} {'Success':<10} {'Total':<8} {'Specialists':<20}")
        print("-" * 65)

        for pid in sorted(problems.keys()):
            p = problems[pid]
            rate = p["successes"] / p["attempts"] * 100 if p["attempts"] > 0 else 0
            specs = ", ".join(p["specialists"][:3])
            if len(p["specialists"]) > 3:
                specs += "..."
            status = "✓" if rate == 100 else "✗" if rate == 0 else "~"
            print(f"{status} {pid[:23]:<23} {p['successes']:<10} {p['attempts']:<8} {specs:<20}")

    # Recent routing
    patterns = get_routing_patterns(db_path)
    if patterns:
        print("\n--- Recent Routing Decisions ---")
        for p in patterns[:10]:
            status = "✓" if p["decision_correct"] else "✗" if p["decision_correct"] is not None else "?"
            conf = p["confidence"] or 0
            print(f"  {status} {p['problem_id'][:20]:<20} → {p['selected_specialist']:<12} (conf: {conf:.0%})")

    # Failure analysis
    failures = get_failure_analysis(db_path)
    if failures:
        print("\n--- Failure Patterns ---")
        for spec, fails in failures.items():
            if specialist_filter and spec != specialist_filter:
                continue
            print(f"\n  {spec} failures ({len(fails)}):")
            for f in fails[:3]:
                needed = f["needed"] or "unknown"
                print(f"    - {f['problem'][:30]} (needed: {needed})")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Specialist performance dashboard")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Database path")
    parser.add_argument("--specialist", type=str, help="Filter to specific specialist")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.json:
        data = {
            "stats": get_specialist_stats(args.db),
            "problems": get_problem_breakdown(args.db),
            "patterns": get_routing_patterns(args.db),
            "failures": get_failure_analysis(args.db),
        }
        print(json.dumps(data, indent=2, default=str))
    else:
        print_dashboard(args.db, args.specialist)


if __name__ == "__main__":
    main()
