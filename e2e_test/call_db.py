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

                -- =================================================================
                -- TRAINING DATA TABLES (hi_moe-828)
                -- =================================================================

                -- Code validation results for SFT training
                CREATE TABLE IF NOT EXISTS validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    call_id INTEGER REFERENCES calls(id),
                    run_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,

                    -- Validation outcome
                    passed BOOLEAN DEFAULT 0,
                    tests_total INTEGER DEFAULT 0,
                    tests_passed INTEGER DEFAULT 0,

                    -- Extracted artifacts
                    extracted_code TEXT,        -- Clean code from response
                    execution_output TEXT,      -- stdout/stderr
                    error_type TEXT,            -- SyntaxError, RuntimeError, WrongAnswer, etc.
                    error_message TEXT,

                    -- Quality signals
                    execution_time_ms INTEGER,  -- How long code took to run
                    memory_used_bytes INTEGER,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_validations_call ON validations(call_id);
                CREATE INDEX IF NOT EXISTS idx_validations_passed ON validations(passed);

                -- Preference pairs for DPO training
                CREATE TABLE IF NOT EXISTS comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_id TEXT NOT NULL,
                    comparison_group TEXT NOT NULL,  -- Groups attempts on same problem

                    chosen_call_id INTEGER REFERENCES calls(id),
                    rejected_call_id INTEGER REFERENCES calls(id),

                    -- Why chosen was better
                    reason TEXT,  -- faster, correct, cleaner, etc.
                    margin REAL,  -- How much better (0-1)

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_comparisons_problem ON comparisons(problem_id);
                CREATE INDEX IF NOT EXISTS idx_comparisons_group ON comparisons(comparison_group);

                -- Self-healing sequences (error → fix)
                CREATE TABLE IF NOT EXISTS retries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    attempt_number INTEGER DEFAULT 1,

                    -- Link to calls
                    failed_call_id INTEGER REFERENCES calls(id),
                    retry_call_id INTEGER REFERENCES calls(id),

                    -- Error context passed to retry
                    error_context TEXT,
                    fix_strategy TEXT,  -- same_specialist, different_specialist, escalate

                    -- Outcome
                    retry_succeeded BOOLEAN DEFAULT 0,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_retries_run ON retries(run_id);

                -- Routing decisions for router training
                CREATE TABLE IF NOT EXISTS routing_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,

                    -- Input features
                    problem_embedding TEXT,     -- JSON array of floats (optional)
                    problem_keywords TEXT,      -- JSON array of keywords
                    estimated_difficulty REAL,  -- 0-1 scale

                    -- Decision
                    selected_specialist TEXT,
                    confidence REAL,            -- 0-1
                    alternative_specialists TEXT,  -- JSON array

                    -- Outcome (filled after execution)
                    decision_correct BOOLEAN,
                    actual_specialist_needed TEXT,

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_routing_problem ON routing_decisions(problem_id);
                CREATE INDEX IF NOT EXISTS idx_routing_specialist ON routing_decisions(selected_specialist);

                -- =================================================================
                -- TRAINING DATA VIEWS
                -- =================================================================

                -- SFT training pairs: successful (input, output)
                CREATE VIEW IF NOT EXISTS training_sft AS
                SELECT
                    c.id as call_id,
                    c.problem_id,
                    c.tier,
                    c.specialist,
                    c.input_preview as input,
                    c.output_preview as output,
                    v.extracted_code as code,
                    v.tests_passed,
                    v.tests_total,
                    c.created_at
                FROM calls c
                JOIN validations v ON v.call_id = c.id
                WHERE v.passed = 1
                ORDER BY c.created_at DESC;

                -- DPO training pairs: (chosen, rejected) per problem
                CREATE VIEW IF NOT EXISTS training_dpo AS
                SELECT
                    comp.problem_id,
                    comp.comparison_group,
                    chosen.input_preview as chosen_input,
                    chosen.output_preview as chosen_output,
                    rejected.input_preview as rejected_input,
                    rejected.output_preview as rejected_output,
                    comp.reason,
                    comp.margin
                FROM comparisons comp
                JOIN calls chosen ON chosen.id = comp.chosen_call_id
                JOIN calls rejected ON rejected.id = comp.rejected_call_id;

                -- Router training: (problem, specialist, success)
                CREATE VIEW IF NOT EXISTS training_router AS
                SELECT
                    r.problem_id,
                    r.problem_keywords,
                    r.estimated_difficulty,
                    r.selected_specialist,
                    r.confidence,
                    r.decision_correct,
                    COUNT(c.id) as attempts,
                    SUM(CASE WHEN c.success THEN 1 ELSE 0 END) as successes
                FROM routing_decisions r
                LEFT JOIN calls c ON c.run_id = r.run_id AND c.specialist = r.selected_specialist
                GROUP BY r.id;

                -- Self-healing training: error → successful fix
                CREATE VIEW IF NOT EXISTS training_selfheal AS
                SELECT
                    r.problem_id,
                    failed.output_preview as failed_output,
                    r.error_context,
                    r.fix_strategy,
                    retry.output_preview as successful_output,
                    rv.extracted_code as fixed_code
                FROM retries r
                JOIN calls failed ON failed.id = r.failed_call_id
                JOIN calls retry ON retry.id = r.retry_call_id
                LEFT JOIN validations rv ON rv.call_id = r.retry_call_id
                WHERE r.retry_succeeded = 1;
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
    # Training data logging (hi_moe-828)
    # -------------------------------------------------------------------------

    def log_validation(
        self,
        call_id: int,
        run_id: str,
        problem_id: str,
        passed: bool,
        tests_total: int = 0,
        tests_passed: int = 0,
        extracted_code: str | None = None,
        execution_output: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        execution_time_ms: int | None = None,
    ) -> int:
        """Log code validation result for SFT training."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO validations (
                    call_id, run_id, problem_id, passed,
                    tests_total, tests_passed, extracted_code,
                    execution_output, error_type, error_message, execution_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (call_id, run_id, problem_id, passed,
                 tests_total, tests_passed,
                 extracted_code[:50000] if extracted_code else None,
                 execution_output[:5000] if execution_output else None,
                 error_type, error_message, execution_time_ms),
            )
            return cursor.lastrowid

    def log_comparison(
        self,
        problem_id: str,
        comparison_group: str,
        chosen_call_id: int,
        rejected_call_id: int,
        reason: str | None = None,
        margin: float = 0.5,
    ) -> int:
        """Log preference pair for DPO training."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO comparisons (
                    problem_id, comparison_group, chosen_call_id,
                    rejected_call_id, reason, margin
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (problem_id, comparison_group, chosen_call_id,
                 rejected_call_id, reason, margin),
            )
            return cursor.lastrowid

    def log_retry(
        self,
        run_id: str,
        problem_id: str,
        attempt_number: int,
        failed_call_id: int,
        retry_call_id: int,
        error_context: str | None = None,
        fix_strategy: str | None = None,
        retry_succeeded: bool = False,
    ) -> int:
        """Log self-healing retry for error→fix training."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO retries (
                    run_id, problem_id, attempt_number,
                    failed_call_id, retry_call_id,
                    error_context, fix_strategy, retry_succeeded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, problem_id, attempt_number,
                 failed_call_id, retry_call_id,
                 error_context[:5000] if error_context else None,
                 fix_strategy, retry_succeeded),
            )
            return cursor.lastrowid

    def log_routing_decision(
        self,
        run_id: str,
        problem_id: str,
        selected_specialist: str,
        confidence: float = 0.5,
        problem_keywords: list[str] | None = None,
        estimated_difficulty: float | None = None,
        alternative_specialists: list[str] | None = None,
    ) -> int:
        """Log routing decision for router training."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO routing_decisions (
                    run_id, problem_id, selected_specialist, confidence,
                    problem_keywords, estimated_difficulty, alternative_specialists
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, problem_id, selected_specialist, confidence,
                 json.dumps(problem_keywords) if problem_keywords else None,
                 estimated_difficulty,
                 json.dumps(alternative_specialists) if alternative_specialists else None),
            )
            return cursor.lastrowid

    def update_routing_outcome(
        self,
        run_id: str,
        decision_correct: bool,
        actual_specialist_needed: str | None = None,
    ):
        """Update routing decision with outcome after execution."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE routing_decisions SET
                    decision_correct = ?,
                    actual_specialist_needed = ?
                WHERE run_id = ?
                """,
                (decision_correct, actual_specialist_needed, run_id),
            )

    # -------------------------------------------------------------------------
    # Training data export
    # -------------------------------------------------------------------------

    def export_sft_data(self, output_path: Path, min_tests_passed: int = 1) -> int:
        """Export successful examples for SFT training."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM training_sft
                WHERE tests_passed >= ?
                """,
                (min_tests_passed,),
            ).fetchall()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + "\n")
        return len(rows)

    def export_dpo_data(self, output_path: Path) -> int:
        """Export preference pairs for DPO training."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM training_dpo").fetchall()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for row in rows:
                f.write(json.dumps(dict(row)) + "\n")
        return len(rows)

    def get_training_stats(self) -> dict:
        """Get statistics on available training data."""
        with self._connect() as conn:
            sft = conn.execute("SELECT COUNT(*) FROM training_sft").fetchone()[0]
            dpo = conn.execute("SELECT COUNT(*) FROM training_dpo").fetchone()[0]
            selfheal = conn.execute("SELECT COUNT(*) FROM training_selfheal").fetchone()[0]
            router = conn.execute("SELECT COUNT(*) FROM training_router").fetchone()[0]

        return {
            "sft_examples": sft,
            "dpo_pairs": dpo,
            "selfheal_sequences": selfheal,
            "router_decisions": router,
        }

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
