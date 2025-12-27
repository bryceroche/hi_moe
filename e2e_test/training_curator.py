#!/usr/bin/env python3
"""Training data curation CLI (hi_moe-dh1).

Tool to review, filter, and export training data for fine-tuning.

Usage:
    python -m e2e_test.training_curator list                    # List all examples
    python -m e2e_test.training_curator list --passed           # Only passing examples
    python -m e2e_test.training_curator list --specialist python # Filter by specialist
    python -m e2e_test.training_curator stats                   # Show statistics
    python -m e2e_test.training_curator export sft output.jsonl # Export SFT data
    python -m e2e_test.training_curator export dpo output.jsonl # Export DPO pairs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .call_db import CallDB


DEFAULT_DB = Path("runs/hi_moe.db")


def cmd_list(db: CallDB, args):
    """List training examples."""
    with db._connect() as conn:
        query = """
            SELECT
                v.id,
                v.problem_id,
                c.specialist,
                v.passed,
                v.tests_passed,
                v.tests_total,
                v.error_type,
                v.created_at
            FROM validations v
            LEFT JOIN calls c ON c.id = v.call_id
            WHERE 1=1
        """
        params = []

        if args.passed:
            query += " AND v.passed = 1"
        if args.failed:
            query += " AND v.passed = 0"
        if args.specialist:
            query += " AND c.specialist = ?"
            params.append(args.specialist)
        if args.problem:
            query += " AND v.problem_id LIKE ?"
            params.append(f"%{args.problem}%")

        query += " ORDER BY v.created_at DESC"
        if args.limit:
            query += f" LIMIT {args.limit}"

        rows = conn.execute(query, params).fetchall()

    if not rows:
        print("No examples found matching criteria.")
        return

    print(f"\n{'ID':<6} {'Problem':<25} {'Specialist':<12} {'Status':<8} {'Tests':<10} {'Error':<15}")
    print("-" * 80)

    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        tests = f"{row['tests_passed'] or 0}/{row['tests_total'] or 0}"
        error = (row["error_type"] or "")[:15]
        problem = (row["problem_id"] or "")[:24]
        spec = (row["specialist"] or "unknown")[:11]
        print(f"{row['id']:<6} {problem:<25} {spec:<12} {status:<8} {tests:<10} {error:<15}")

    print(f"\nTotal: {len(rows)} examples")


def cmd_stats(db: CallDB, args):
    """Show training data statistics."""
    stats = db.get_training_stats()

    print("\n" + "=" * 50)
    print("TRAINING DATA STATISTICS")
    print("=" * 50)
    print(f"\nSFT Examples (passing):     {stats['sft_examples']:>6}")
    print(f"DPO Pairs:                  {stats['dpo_pairs']:>6}")
    print(f"Self-heal Sequences:        {stats['selfheal_sequences']:>6}")
    print(f"Router Decisions:           {stats['router_decisions']:>6}")

    # Breakdown by specialist
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT
                c.specialist,
                COUNT(*) as total,
                SUM(CASE WHEN v.passed THEN 1 ELSE 0 END) as passed
            FROM validations v
            LEFT JOIN calls c ON c.id = v.call_id
            GROUP BY c.specialist
        """).fetchall()

    if rows:
        print("\n--- By Specialist ---")
        print(f"{'Specialist':<15} {'Passed':<10} {'Total':<10} {'Rate':<10}")
        print("-" * 45)
        for row in rows:
            spec = row["specialist"] or "unknown"
            total = row["total"] or 0
            passed = row["passed"] or 0
            rate = passed / total * 100 if total > 0 else 0
            print(f"{spec:<15} {passed:<10} {total:<10} {rate:>5.1f}%")

    # Error type breakdown
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT error_type, COUNT(*) as count
            FROM validations
            WHERE passed = 0 AND error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()

    if rows:
        print("\n--- Common Errors ---")
        for row in rows:
            print(f"  {row['error_type']}: {row['count']}")

    print()


def cmd_show(db: CallDB, args):
    """Show details of a specific example."""
    with db._connect() as conn:
        row = conn.execute("""
            SELECT
                v.*,
                c.input_preview,
                c.output_preview,
                c.specialist,
                c.tier
            FROM validations v
            LEFT JOIN calls c ON c.id = v.call_id
            WHERE v.id = ?
        """, (args.id,)).fetchone()

    if not row:
        print(f"Example {args.id} not found.")
        return

    print("\n" + "=" * 60)
    print(f"EXAMPLE #{row['id']}")
    print("=" * 60)
    print(f"Problem:    {row['problem_id']}")
    print(f"Specialist: {row['specialist']}")
    print(f"Status:     {'PASS' if row['passed'] else 'FAIL'}")
    print(f"Tests:      {row['tests_passed']}/{row['tests_total']}")
    if row['error_type']:
        print(f"Error:      {row['error_type']}: {row['error_message']}")
    print(f"Created:    {row['created_at']}")

    if row['extracted_code']:
        print("\n--- Extracted Code ---")
        print(row['extracted_code'][:1000])
        if len(row['extracted_code']) > 1000:
            print("... (truncated)")

    if row['execution_output']:
        print("\n--- Execution Output ---")
        print(row['execution_output'][:500])

    print()


def cmd_export(db: CallDB, args):
    """Export training data."""
    output_path = Path(args.output)

    if args.format == "sft":
        count = db.export_sft_data(output_path, min_tests_passed=args.min_tests or 1)
        print(f"Exported {count} SFT examples to {output_path}")

    elif args.format == "dpo":
        count = db.export_dpo_data(output_path)
        print(f"Exported {count} DPO pairs to {output_path}")

    elif args.format == "router":
        # Export router training data
        with db._connect() as conn:
            rows = conn.execute("SELECT * FROM training_router").fetchall()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(dict(row), default=str) + "\n")
        print(f"Exported {len(rows)} router decisions to {output_path}")

    elif args.format == "selfheal":
        # Export self-healing sequences
        with db._connect() as conn:
            rows = conn.execute("SELECT * FROM training_selfheal").fetchall()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(dict(row), default=str) + "\n")
        print(f"Exported {len(rows)} self-heal sequences to {output_path}")


def cmd_clean(db: CallDB, args):
    """Clean low-quality examples."""
    with db._connect() as conn:
        # Count before
        before = conn.execute("SELECT COUNT(*) FROM validations").fetchone()[0]

        if args.dry_run:
            # Just show what would be deleted
            if args.no_code:
                count = conn.execute("""
                    SELECT COUNT(*) FROM validations
                    WHERE extracted_code IS NULL OR extracted_code = ''
                """).fetchone()[0]
                print(f"Would delete {count} examples with no code")

            if args.zero_tests:
                count = conn.execute("""
                    SELECT COUNT(*) FROM validations
                    WHERE tests_total = 0 OR tests_total IS NULL
                """).fetchone()[0]
                print(f"Would delete {count} examples with no tests")
        else:
            deleted = 0
            if args.no_code:
                conn.execute("""
                    DELETE FROM validations
                    WHERE extracted_code IS NULL OR extracted_code = ''
                """)
                deleted += conn.total_changes

            if args.zero_tests:
                conn.execute("""
                    DELETE FROM validations
                    WHERE tests_total = 0 OR tests_total IS NULL
                """)
                deleted += conn.total_changes

            after = conn.execute("SELECT COUNT(*) FROM validations").fetchone()[0]
            print(f"Deleted {before - after} examples. Remaining: {after}")


def main():
    parser = argparse.ArgumentParser(description="Training data curation CLI")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Database path")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # List command
    list_p = subparsers.add_parser("list", help="List training examples")
    list_p.add_argument("--passed", action="store_true", help="Only passing examples")
    list_p.add_argument("--failed", action="store_true", help="Only failing examples")
    list_p.add_argument("--specialist", type=str, help="Filter by specialist")
    list_p.add_argument("--problem", type=str, help="Filter by problem (substring)")
    list_p.add_argument("--limit", type=int, default=50, help="Max results")

    # Stats command
    subparsers.add_parser("stats", help="Show training data statistics")

    # Show command
    show_p = subparsers.add_parser("show", help="Show example details")
    show_p.add_argument("id", type=int, help="Example ID")

    # Export command
    export_p = subparsers.add_parser("export", help="Export training data")
    export_p.add_argument("format", choices=["sft", "dpo", "router", "selfheal"])
    export_p.add_argument("output", help="Output file path")
    export_p.add_argument("--min-tests", type=int, help="Min tests passed (for SFT)")

    # Clean command
    clean_p = subparsers.add_parser("clean", help="Clean low-quality examples")
    clean_p.add_argument("--no-code", action="store_true", help="Remove examples with no code")
    clean_p.add_argument("--zero-tests", action="store_true", help="Remove examples with no tests")
    clean_p.add_argument("--dry-run", action="store_true", help="Show what would be deleted")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    db = CallDB(args.db)

    if args.command == "list":
        cmd_list(db, args)
    elif args.command == "stats":
        cmd_stats(db, args)
    elif args.command == "show":
        cmd_show(db, args)
    elif args.command == "export":
        cmd_export(db, args)
    elif args.command == "clean":
        cmd_clean(db, args)


if __name__ == "__main__":
    main()
