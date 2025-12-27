#!/usr/bin/env python3
"""Error pattern analyzer (hi_moe-x2q).

Analyzes common failure modes from validation data to:
1. Cluster errors by type and message patterns
2. Identify specialist-specific failure patterns
3. Suggest prompt improvements based on common errors

Usage:
    python -m e2e_test.error_analyzer                    # Full analysis
    python -m e2e_test.error_analyzer --specialist python # Filter by specialist
    python -m e2e_test.error_analyzer --top 10           # Top N patterns
    python -m e2e_test.error_analyzer --suggest          # Generate prompt suggestions
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


DEFAULT_DB = Path("runs/hi_moe.db")


def normalize_error(error_msg: str | None) -> str:
    """Normalize error message for clustering."""
    if not error_msg:
        return "unknown"

    # Remove line numbers and file paths
    msg = re.sub(r'line \d+', 'line N', error_msg)
    msg = re.sub(r'File "[^"]*"', 'File "..."', msg)

    # Remove specific variable names
    msg = re.sub(r"'[a-z_][a-z_0-9]*'", "'var'", msg, flags=re.IGNORECASE)

    # Truncate long messages
    if len(msg) > 100:
        msg = msg[:100] + "..."

    return msg.strip()


def get_error_clusters(db_path: Path, specialist: str | None = None) -> dict:
    """Cluster errors by type and normalized message."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            v.error_type,
            v.error_message,
            v.problem_id,
            c.specialist,
            v.extracted_code
        FROM validations v
        LEFT JOIN calls c ON c.id = v.call_id
        WHERE v.passed = 0 AND v.error_type IS NOT NULL
    """
    params = []
    if specialist:
        query += " AND c.specialist = ?"
        params.append(specialist)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    # Cluster by error_type -> normalized_message -> examples
    clusters = defaultdict(lambda: defaultdict(list))
    for row in rows:
        error_type = row["error_type"] or "Unknown"
        normalized = normalize_error(row["error_message"])
        clusters[error_type][normalized].append({
            "problem": row["problem_id"],
            "specialist": row["specialist"],
            "message": row["error_message"],
            "code": row["extracted_code"],
        })

    return dict(clusters)


def get_specialist_error_rates(db_path: Path) -> dict:
    """Get error rates by specialist."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            c.specialist,
            v.error_type,
            COUNT(*) as count
        FROM validations v
        LEFT JOIN calls c ON c.id = v.call_id
        WHERE v.passed = 0 AND v.error_type IS NOT NULL
        GROUP BY c.specialist, v.error_type
        ORDER BY count DESC
    """).fetchall()
    conn.close()

    rates = defaultdict(lambda: defaultdict(int))
    for row in rows:
        rates[row["specialist"]][row["error_type"]] = row["count"]

    return dict(rates)


def generate_prompt_suggestions(clusters: dict) -> list[dict]:
    """Generate prompt improvement suggestions based on error patterns."""
    suggestions = []

    for error_type, patterns in clusters.items():
        for pattern, examples in patterns.items():
            count = len(examples)
            if count < 2:  # Only suggest for recurring errors
                continue

            # Analyze the pattern
            specialists = set(e["specialist"] for e in examples)
            problems = set(e["problem"] for e in examples)

            suggestion = {
                "error_type": error_type,
                "pattern": pattern,
                "count": count,
                "specialists": list(specialists),
                "example_problems": list(problems)[:3],
            }

            # Generate specific advice based on error type
            if error_type == "SyntaxError":
                suggestion["advice"] = (
                    "Add explicit instruction: 'Ensure code has valid Python syntax. "
                    "Check for matching brackets, colons after if/for/def, and proper indentation.'"
                )
            elif error_type == "NameError":
                suggestion["advice"] = (
                    "Add instruction: 'Define all variables before use. "
                    "Import any required modules at the top of the code.'"
                )
            elif error_type == "TypeError":
                suggestion["advice"] = (
                    "Add instruction: 'Verify argument types match function signatures. "
                    "Use explicit type conversion when needed.'"
                )
            elif error_type == "IndexError":
                suggestion["advice"] = (
                    "Add instruction: 'Check array bounds before accessing. "
                    "Handle empty arrays as edge cases.'"
                )
            elif error_type == "WrongAnswer":
                suggestion["advice"] = (
                    "Add instruction: 'Test with edge cases: empty input, single element, "
                    "large values, negative numbers, duplicates.'"
                )
            elif error_type == "Timeout":
                suggestion["advice"] = (
                    "Add instruction: 'Use efficient algorithms. Avoid O(n^2) when O(n) exists. "
                    "Consider hash maps for O(1) lookup.'"
                )
            else:
                suggestion["advice"] = (
                    f"Common {error_type} pattern detected. Consider adding specific handling."
                )

            suggestions.append(suggestion)

    return sorted(suggestions, key=lambda x: x["count"], reverse=True)


def print_analysis(db_path: Path, specialist: str | None = None, top_n: int = 20, show_suggestions: bool = False):
    """Print error pattern analysis."""
    print("=" * 70)
    print("ERROR PATTERN ANALYSIS")
    print(f"Database: {db_path}")
    if specialist:
        print(f"Specialist: {specialist}")
    print("=" * 70)

    clusters = get_error_clusters(db_path, specialist)
    if not clusters:
        print("\nNo errors found in database.")
        return

    # Count total errors
    total_errors = sum(
        len(examples)
        for patterns in clusters.values()
        for examples in patterns.values()
    )
    print(f"\nTotal errors analyzed: {total_errors}")

    # Error type breakdown
    print("\n--- Error Types ---")
    type_counts = {}
    for error_type, patterns in clusters.items():
        count = sum(len(examples) for examples in patterns.values())
        type_counts[error_type] = count

    for error_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = count / total_errors * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {error_type:<20} {count:>5} ({pct:>5.1f}%) {bar}")

    # Top patterns
    print(f"\n--- Top {top_n} Error Patterns ---")
    all_patterns = []
    for error_type, patterns in clusters.items():
        for pattern, examples in patterns.items():
            all_patterns.append({
                "type": error_type,
                "pattern": pattern,
                "count": len(examples),
                "examples": examples,
            })

    all_patterns.sort(key=lambda x: -x["count"])

    for i, p in enumerate(all_patterns[:top_n], 1):
        print(f"\n{i}. [{p['type']}] {p['pattern']}")
        print(f"   Occurrences: {p['count']}")
        specialists = set(e["specialist"] for e in p["examples"])
        print(f"   Specialists: {', '.join(str(s) for s in specialists)}")
        problems = [e["problem"] for e in p["examples"][:3]]
        print(f"   Example problems: {', '.join(problems)}")

    # Specialist breakdown
    rates = get_specialist_error_rates(db_path)
    if rates:
        print("\n--- Errors by Specialist ---")
        for spec, errors in sorted(rates.items(), key=lambda x: sum(x[1].values()), reverse=True):
            if specialist and spec != specialist:
                continue
            total = sum(errors.values())
            print(f"\n  {spec} ({total} errors):")
            for err_type, count in sorted(errors.items(), key=lambda x: -x[1])[:5]:
                print(f"    - {err_type}: {count}")

    # Suggestions
    if show_suggestions:
        suggestions = generate_prompt_suggestions(clusters)
        if suggestions:
            print("\n" + "=" * 70)
            print("PROMPT IMPROVEMENT SUGGESTIONS")
            print("=" * 70)

            for i, s in enumerate(suggestions[:10], 1):
                print(f"\n{i}. {s['error_type']}: {s['pattern']}")
                print(f"   Seen {s['count']} times in: {', '.join(s['specialists'])}")
                print(f"   Suggestion: {s['advice']}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Error pattern analyzer")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Database path")
    parser.add_argument("--specialist", type=str, help="Filter by specialist")
    parser.add_argument("--top", type=int, default=20, help="Show top N patterns")
    parser.add_argument("--suggest", action="store_true", help="Generate prompt suggestions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.json:
        import json
        clusters = get_error_clusters(args.db, args.specialist)
        suggestions = generate_prompt_suggestions(clusters) if args.suggest else []
        data = {
            "clusters": {
                t: {p: len(ex) for p, ex in patterns.items()}
                for t, patterns in clusters.items()
            },
            "suggestions": suggestions,
        }
        print(json.dumps(data, indent=2))
    else:
        print_analysis(args.db, args.specialist, args.top, args.suggest)


if __name__ == "__main__":
    main()
