#!/usr/bin/env python3
"""SQLite database for vLLM call logging and results (hi_moe-ofp).

Replaces JSONL with queryable SQLite. Supports multi-instance writes via WAL mode.

Usage:
    from e2e_test.call_db import CallDB

    db = CallDB()

    # Log a call
    db.log_call(
        run_id="run-123",
        problem_id="two_sum",
        tier="fleet",
        tokens_in=100,
        tokens_out=500,
        latency_ms=5000,
        success=True,
    )

    # Query
    stats = db.get_stats()
    calls = db.get_calls(problem_id="two_sum")
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path("runs/hi_moe.db")


@dataclass
class CallRecord:
    """A single vLLM call."""
    id: int
    run_id: str
    problem_id: str
    tier: str
    specialist: str | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    success: bool
    error: str | None
    created_at: str


@dataclass
class RunRecord:
    """A single problem run."""
    id: int
    run_id: str
    problem_id: str
    status: str  # pending, running, success, failed
    total_calls: int
    total_tokens: int
    total_time_ms: int
    result_code: str | None
    error: str | None
    created_at: str
    updated_at: str


class CallDB:
    """SQLite database for call logging."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript("""
                -- Enable WAL mode for concurrent writes
                PRAGMA journal_mode=WAL;
                PRAGMA busy_timeout=5000;

                -- Calls table
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    tier TEXT,
                    specialist TEXT,
                    adapter TEXT,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    latency_ms INTEGER DEFAULT 0,
                    success BOOLEAN DEFAULT 1,
                    error TEXT,
                    input_preview TEXT,
                    output_preview TEXT,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_calls_run ON calls(run_id);
                CREATE INDEX IF NOT EXISTS idx_calls_problem ON calls(problem_id);
                CREATE INDEX IF NOT EXISTS idx_calls_created ON calls(created_at);

                -- Runs table
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    problem_id TEXT NOT NULL,
                    instance_id TEXT,
                    status TEXT DEFAULT 'pending',
                    total_calls INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_time_ms INTEGER DEFAULT 0,
                    result_code TEXT,
                    error TEXT,
                    metadata TEXT,  -- JSON
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_runs_problem ON runs(problem_id);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
                CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);

                -- Stats view
                CREATE VIEW IF NOT EXISTS call_stats AS
                SELECT
                    problem_id,
                    COUNT(*) as call_count,
                    SUM(tokens_in) as total_tokens_in,
                    SUM(tokens_out) as total_tokens_out,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                    MIN(created_at) as first_call,
                    MAX(created_at) as last_call
                FROM calls
                GROUP BY problem_id;
            """)

    @contextmanager
    def _connect(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Logging methods
    # -------------------------------------------------------------------------

    def log_call(
        self,
        run_id: str,
        problem_id: str,
        tier: str | None = None,
        specialist: str | None = None,
        adapter: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: int = 0,
        success: bool = True,
        error: str | None = None,
        input_preview: str | None = None,
        output_preview: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Log a vLLM call."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calls (
                    run_id, problem_id, tier, specialist, adapter,
                    tokens_in, tokens_out, latency_ms, success, error,
                    input_preview, output_preview, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, problem_id, tier, specialist, adapter,
                    tokens_in, tokens_out, latency_ms, success, error,
                    input_preview[:500] if input_preview else None,
                    output_preview[:500] if output_preview else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def start_run(
        self,
        run_id: str,
        problem_id: str,
        instance_id: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Start a new run."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (run_id, problem_id, instance_id, status, metadata)
                VALUES (?, ?, ?, 'running', ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status='running',
                    updated_at=CURRENT_TIMESTAMP
                """,
                (run_id, problem_id, instance_id, json.dumps(metadata) if metadata else None),
            )
            return cursor.lastrowid

    def end_run(
        self,
        run_id: str,
        success: bool,
        total_calls: int = 0,
        total_tokens: int = 0,
        total_time_ms: int = 0,
        result_code: str | None = None,
        error: str | None = None,
    ):
        """End a run with results."""
        status = "success" if success else "failed"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs SET
                    status=?,
                    total_calls=?,
                    total_tokens=?,
                    total_time_ms=?,
                    result_code=?,
                    error=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE run_id=?
                """,
                (status, total_calls, total_tokens, total_time_ms,
                 result_code[:10000] if result_code else None, error, run_id),
            )

    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------

    def get_calls(
        self,
        run_id: str | None = None,
        problem_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get calls matching criteria."""
        with self._connect() as conn:
            query = "SELECT * FROM calls WHERE 1=1"
            params = []

            if run_id:
                query += " AND run_id = ?"
                params.append(run_id)
            if problem_id:
                query += " AND problem_id = ?"
                params.append(problem_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_runs(
        self,
        status: str | None = None,
        problem_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get runs matching criteria."""
        with self._connect() as conn:
            query = "SELECT * FROM runs WHERE 1=1"
            params = []

            if status:
                query += " AND status = ?"
                params.append(status)
            if problem_id:
                query += " AND problem_id = ?"
                params.append(problem_id)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        with self._connect() as conn:
            # Overall stats
            overall = conn.execute("""
                SELECT
                    COUNT(*) as total_calls,
                    SUM(tokens_in) as total_tokens_in,
                    SUM(tokens_out) as total_tokens_out,
                    AVG(latency_ms) as avg_latency_ms,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
                FROM calls
            """).fetchone()

            run_stats = conn.execute("""
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successful_runs,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed_runs,
                    SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) as running_runs
                FROM runs
            """).fetchone()

            # By problem
            by_problem = conn.execute("SELECT * FROM call_stats").fetchall()

            return {
                "calls": dict(overall) if overall else {},
                "runs": dict(run_stats) if run_stats else {},
                "by_problem": [dict(r) for r in by_problem],
            }

    def get_cost_estimate(self, cost_per_1k_tokens: float = 0.002) -> dict:
        """Estimate costs from token usage."""
        with self._connect() as conn:
            result = conn.execute("""
                SELECT
                    SUM(tokens_in) as total_in,
                    SUM(tokens_out) as total_out,
                    SUM(tokens_in + tokens_out) as total
                FROM calls
            """).fetchone()

            total_tokens = result["total"] or 0
            return {
                "total_tokens": total_tokens,
                "tokens_in": result["total_in"] or 0,
                "tokens_out": result["total_out"] or 0,
                "estimated_cost": (total_tokens / 1000) * cost_per_1k_tokens,
            }

    # -------------------------------------------------------------------------
    # Import from JSONL
    # -------------------------------------------------------------------------

    def import_jsonl(self, jsonl_path: Path) -> int:
        """Import calls from a JSONL trajectory file."""
        imported = 0
        run_id = None
        problem_id = None

        with open(jsonl_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")

                if entry_type == "run_start":
                    run_id = entry.get("run_id")
                    problem_id = entry.get("problem_id")
                    self.start_run(run_id, problem_id or "unknown")

                elif entry_type == "vllm_call" and run_id:
                    self.log_call(
                        run_id=run_id,
                        problem_id=problem_id or "unknown",
                        tier=entry.get("tier"),
                        tokens_in=entry.get("tokens_in", 0),
                        tokens_out=entry.get("tokens_out", 0),
                        latency_ms=int(entry.get("latency_ms", 0)),
                        success=entry.get("status") == "success",
                        error=entry.get("error"),
                    )
                    imported += 1

                elif entry_type == "run_end" and run_id:
                    self.end_run(
                        run_id=run_id,
                        success=entry.get("status") == "completed",
                        total_calls=entry.get("total_calls", 0),
                        total_time_ms=int(entry.get("elapsed_ms", 0)),
                        error=entry.get("error"),
                    )

        return imported


def main():
    """Show database stats and recent activity."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM call database")
    parser.add_argument("--import-all", action="store_true", help="Import all JSONL files")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--recent", type=int, default=10, help="Show N recent calls")
    args = parser.parse_args()

    db = CallDB()

    if args.import_all:
        total = 0
        for f in Path("runs").rglob("*.jsonl"):
            imported = db.import_jsonl(f)
            if imported:
                print(f"Imported {imported} calls from {f.name}")
                total += imported
        print(f"\nTotal imported: {total} calls")
        return

    if args.stats:
        stats = db.get_stats()
        print("=== Call Statistics ===")
        print(f"Total calls: {stats['calls'].get('total_calls', 0)}")
        print(f"Tokens: {stats['calls'].get('total_tokens_in', 0):,} in, "
              f"{stats['calls'].get('total_tokens_out', 0):,} out")
        print(f"Avg latency: {stats['calls'].get('avg_latency_ms', 0):.0f}ms")
        print()
        print("=== Run Statistics ===")
        print(f"Total runs: {stats['runs'].get('total_runs', 0)}")
        print(f"Successful: {stats['runs'].get('successful_runs', 0)}")
        print(f"Failed: {stats['runs'].get('failed_runs', 0)}")
        print()
        cost = db.get_cost_estimate()
        print(f"=== Cost Estimate ===")
        print(f"Total tokens: {cost['total_tokens']:,}")
        print(f"Est. cost: ${cost['estimated_cost']:.2f}")
        return

    # Default: show recent calls
    print(f"Recent {args.recent} calls:")
    for call in db.get_calls(limit=args.recent):
        status = "✓" if call["success"] else "✗"
        print(f"  {status} {call['problem_id']:20s} {call['latency_ms']:5d}ms "
              f"{call['tokens_out']:5d} tok  {call['tier'] or 'unknown'}")


if __name__ == "__main__":
    main()
