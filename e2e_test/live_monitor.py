#!/usr/bin/env python3
"""Real-time observability for evaluations (hi_moe-pzm).

Live monitoring of running evaluations with auto-refresh.
Shows current problem, tier, specialist, token count, elapsed time.

Usage:
    python -m e2e_test.live_monitor                    # Watch default DB
    python -m e2e_test.live_monitor --db runs/test.db  # Custom DB
    python -m e2e_test.live_monitor --interval 2       # 2 second refresh
    python -m e2e_test.live_monitor --tail 20          # Show last 20 calls
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text


DEFAULT_DB = Path("runs/hi_moe.db")


def get_recent_calls(db_path: Path, limit: int = 10) -> list[dict]:
    """Get most recent LLM calls."""
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
            latency_ms,
            success,
            created_at
        FROM calls
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_active_runs(db_path: Path) -> list[dict]:
    """Get runs that appear to be active (recent activity)."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Runs with activity in last 5 minutes
    cutoff = (datetime.now() - timedelta(minutes=5)).isoformat()

    rows = conn.execute("""
        SELECT
            run_id,
            COUNT(*) as call_count,
            SUM(tokens_in) as total_tokens_in,
            SUM(tokens_out) as total_tokens_out,
            SUM(latency_ms) as total_latency_ms,
            MAX(created_at) as last_activity,
            MIN(created_at) as started_at,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
            COUNT(DISTINCT problem_id) as problems_attempted
        FROM calls
        WHERE created_at > ?
        GROUP BY run_id
        ORDER BY last_activity DESC
    """, (cutoff,)).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def get_summary_stats(db_path: Path) -> dict:
    """Get overall summary statistics."""
    if not db_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    row = conn.execute("""
        SELECT
            COUNT(*) as total_calls,
            SUM(tokens_in) as total_tokens_in,
            SUM(tokens_out) as total_tokens_out,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
            COUNT(DISTINCT run_id) as total_runs,
            COUNT(DISTINCT problem_id) as total_problems
        FROM calls
    """).fetchone()

    # Recent activity (last hour)
    hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
    recent = conn.execute("""
        SELECT COUNT(*) as recent_calls
        FROM calls
        WHERE created_at > ?
    """, (hour_ago,)).fetchone()

    conn.close()

    return {
        **dict(row),
        "recent_calls": recent["recent_calls"] if recent else 0,
    }


def get_specialist_breakdown(db_path: Path) -> list[dict]:
    """Get breakdown by specialist."""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            specialist,
            COUNT(*) as calls,
            SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
            SUM(tokens_in + tokens_out) as tokens
        FROM calls
        WHERE specialist IS NOT NULL
        GROUP BY specialist
        ORDER BY calls DESC
    """).fetchall()
    conn.close()

    return [dict(r) for r in rows]


def create_dashboard(db_path: Path, tail: int = 10) -> Layout:
    """Create the dashboard layout."""
    layout = Layout()

    # Get data
    stats = get_summary_stats(db_path)
    active_runs = get_active_runs(db_path)
    recent_calls = get_recent_calls(db_path, tail)
    specialists = get_specialist_breakdown(db_path)

    # Header
    header_text = Text()
    header_text.append("📊 HI-MOE LIVE MONITOR", style="bold cyan")
    header_text.append(f"  |  DB: {db_path}", style="dim")
    header_text.append(f"  |  {datetime.now().strftime('%H:%M:%S')}", style="dim")

    # Summary stats panel
    if stats:
        total_tokens = (stats.get("total_tokens_in") or 0) + (stats.get("total_tokens_out") or 0)
        success_rate = (stats.get("successes") or 0) / max(stats.get("total_calls") or 1, 1) * 100

        summary = Text()
        summary.append(f"Calls: {stats.get('total_calls', 0):,}", style="bold")
        summary.append(f"  |  Tokens: {total_tokens:,}")
        summary.append(f"  |  Success: {success_rate:.0f}%")
        summary.append(f"  |  Runs: {stats.get('total_runs', 0)}")
        summary.append(f"  |  Problems: {stats.get('total_problems', 0)}")
        summary.append(f"  |  Last hour: {stats.get('recent_calls', 0)} calls", style="green" if stats.get("recent_calls", 0) > 0 else "dim")
    else:
        summary = Text("No data yet", style="dim")

    # Active runs table
    runs_table = Table(title="🔥 Active Runs (last 5 min)", show_header=True, header_style="bold magenta")
    runs_table.add_column("Run ID", style="cyan", max_width=20)
    runs_table.add_column("Calls", justify="right")
    runs_table.add_column("Problems", justify="right")
    runs_table.add_column("Tokens", justify="right")
    runs_table.add_column("Success", justify="right")
    runs_table.add_column("Last Activity", style="dim")

    for run in active_runs[:5]:
        tokens = (run.get("total_tokens_in") or 0) + (run.get("total_tokens_out") or 0)
        successes = run.get("successes") or 0
        calls = run.get("call_count") or 1
        rate = successes / calls * 100

        # Parse last activity
        last = run.get("last_activity", "")
        if last:
            try:
                last_dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
                ago = (datetime.now(last_dt.tzinfo) if last_dt.tzinfo else datetime.now()) - last_dt
                last_str = f"{ago.seconds}s ago" if ago.seconds < 60 else f"{ago.seconds // 60}m ago"
            except Exception:
                last_str = last[-8:]
        else:
            last_str = "-"

        runs_table.add_row(
            (run.get("run_id") or "-")[:20],
            str(run.get("call_count", 0)),
            str(run.get("problems_attempted", 0)),
            f"{tokens:,}",
            f"{rate:.0f}%",
            last_str,
        )

    if not active_runs:
        runs_table.add_row("-", "-", "-", "-", "-", "No active runs")

    # Recent calls table
    calls_table = Table(title=f"📝 Recent Calls (last {tail})", show_header=True, header_style="bold blue")
    calls_table.add_column("ID", style="dim", width=5)
    calls_table.add_column("Problem", max_width=25)
    calls_table.add_column("Tier", width=10)
    calls_table.add_column("Specialist", width=10)
    calls_table.add_column("Tokens", justify="right", width=8)
    calls_table.add_column("Latency", justify="right", width=8)
    calls_table.add_column("Status", width=6)

    for call in recent_calls:
        tokens = (call.get("tokens_in") or 0) + (call.get("tokens_out") or 0)
        latency = call.get("latency_ms") or 0
        success = call.get("success")

        status_style = "green" if success else "red"
        status_text = "✓" if success else "✗"

        calls_table.add_row(
            str(call.get("id", "-")),
            (call.get("problem_id") or "-")[:25],
            call.get("tier") or "-",
            call.get("specialist") or "-",
            f"{tokens:,}",
            f"{latency:.0f}ms",
            Text(status_text, style=status_style),
        )

    if not recent_calls:
        calls_table.add_row("-", "No calls yet", "-", "-", "-", "-", "-")

    # Specialist breakdown
    spec_table = Table(title="👥 Specialists", show_header=True, header_style="bold yellow")
    spec_table.add_column("Specialist", style="cyan")
    spec_table.add_column("Calls", justify="right")
    spec_table.add_column("Success", justify="right")
    spec_table.add_column("Tokens", justify="right")

    for spec in specialists[:6]:
        calls = spec.get("calls") or 1
        successes = spec.get("successes") or 0
        rate = successes / calls * 100
        spec_table.add_row(
            spec.get("specialist") or "-",
            str(calls),
            f"{rate:.0f}%",
            f"{spec.get('tokens', 0):,}",
        )

    # Combine into layout
    layout.split_column(
        Layout(Panel(header_text, style="bold"), size=3),
        Layout(Panel(summary), size=3),
        Layout(name="main"),
    )

    layout["main"].split_row(
        Layout(name="left"),
        Layout(name="right", ratio=2),
    )

    layout["left"].split_column(
        Layout(runs_table),
        Layout(spec_table),
    )

    layout["right"].update(calls_table)

    return layout


def run_monitor(db_path: Path, interval: float = 1.0, tail: int = 10):
    """Run the live monitor."""
    console = Console()

    console.print(f"[bold cyan]Starting live monitor...[/]")
    console.print(f"[dim]DB: {db_path} | Refresh: {interval}s | Press Ctrl+C to exit[/]")
    console.print()

    try:
        with Live(create_dashboard(db_path, tail), console=console, refresh_per_second=1) as live:
            while True:
                time.sleep(interval)
                live.update(create_dashboard(db_path, tail))
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/]")


def main():
    parser = argparse.ArgumentParser(description="Real-time evaluation monitor")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Database path")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds")
    parser.add_argument("--tail", type=int, default=10, help="Number of recent calls to show")
    parser.add_argument("--once", action="store_true", help="Print once and exit (no live refresh)")
    args = parser.parse_args()

    if args.once:
        console = Console()
        console.print(create_dashboard(args.db, args.tail))
    else:
        run_monitor(args.db, args.interval, args.tail)


if __name__ == "__main__":
    main()
