"""Trajectory logging for vLLM calls.

Implements issue hi_moe-iz9: Log every vLLM call for training data collection.

Log format (JSONL):
- ts: ISO timestamp
- task_id: Current task ID
- call_id: Unique call identifier
- tier: Which tier made the call (architect/dispatcher/fleet)
- specialist: Which specialist (if applicable)
- lora: Which LoRA adapter (if any)
- input: Messages sent to model
- output: Model response
- tokens_in: Prompt tokens
- tokens_out: Completion tokens
- latency_ms: Call latency
- status: success/error
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class VLLMCallRecord:
    """Record of a single vLLM API call."""

    ts: str  # ISO timestamp
    task_id: str
    call_id: str
    tier: str  # architect, dispatcher, fleet
    specialist: str | None = None
    lora: str | None = None
    input: list[dict] | None = None  # Messages
    output: str | None = None  # Response text
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0
    status: str = "success"  # success, error
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class TrajectoryLogger:
    """Logs vLLM call trajectories to JSONL files.

    Each run creates a new file in the runs/ directory.
    Files are named: {run_id}.jsonl

    Usage:
        logger = TrajectoryLogger(runs_dir="./runs")
        logger.start_run("run-123", {"problem_id": "two-sum"})

        # Log individual calls
        logger.log_call(VLLMCallRecord(...))

        # End run
        logger.end_run({"success": True})
    """

    def __init__(self, runs_dir: Path | str = "./runs"):
        """Initialize trajectory logger.

        Args:
            runs_dir: Directory to store trajectory files
        """
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self._current_run_id: str | None = None
        self._current_file: Path | None = None
        self._call_count: int = 0

    def start_run(self, run_id: str, metadata: dict | None = None) -> None:
        """Start a new run trajectory.

        Args:
            run_id: Unique run identifier
            metadata: Optional run metadata (problem info, etc.)
        """
        self._current_run_id = run_id
        self._current_file = self.runs_dir / f"{run_id}.jsonl"
        self._call_count = 0

        # Write run start record
        start_record = {
            "type": "run_start",
            "run_id": run_id,
            "ts": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        self._write_record(start_record)

        logger.info(f"[TrajectoryLogger] Started run {run_id}")

    def log_call(self, record: VLLMCallRecord) -> None:
        """Log a vLLM call record.

        Args:
            record: The call record to log
        """
        if not self._current_file:
            logger.warning("[TrajectoryLogger] No active run, call not logged")
            return

        self._call_count += 1
        self._write_record({"type": "vllm_call", **record.to_dict()})

    def end_run(self, result: dict | None = None) -> None:
        """End the current run trajectory.

        Args:
            result: Optional run result summary
        """
        if not self._current_file:
            return

        end_record = {
            "type": "run_end",
            "run_id": self._current_run_id,
            "ts": datetime.utcnow().isoformat(),
            "total_calls": self._call_count,
            **(result or {}),
        }
        self._write_record(end_record)

        logger.info(
            f"[TrajectoryLogger] Ended run {self._current_run_id} "
            f"with {self._call_count} calls"
        )

        self._current_run_id = None
        self._current_file = None
        self._call_count = 0

    def _write_record(self, record: dict) -> None:
        """Write a record to the current trajectory file."""
        if not self._current_file:
            return

        with open(self._current_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @property
    def current_run_id(self) -> str | None:
        """Get current run ID."""
        return self._current_run_id


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients that can be wrapped for logging."""

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        ...


class LoggingLLMClient:
    """Wrapper that adds trajectory logging to any LLM client.

    Intercepts all generate() calls and logs them to a TrajectoryLogger.
    """

    def __init__(
        self,
        client: LLMClientProtocol,
        trajectory_logger: TrajectoryLogger,
        default_tier: str = "unknown",
    ):
        """Initialize logging wrapper.

        Args:
            client: The underlying LLM client
            trajectory_logger: Logger for recording calls
            default_tier: Default tier name for calls
        """
        self.client = client
        self.logger = trajectory_logger
        self.default_tier = default_tier

        # Context for current call (set by caller)
        self._current_task_id: str | None = None
        self._current_tier: str | None = None
        self._current_specialist: str | None = None
        self._current_lora: str | None = None

    def set_context(
        self,
        task_id: str | None = None,
        tier: str | None = None,
        specialist: str | None = None,
        lora: str | None = None,
    ) -> None:
        """Set context for subsequent calls.

        Args:
            task_id: Current task ID
            tier: Current tier (architect/dispatcher/fleet)
            specialist: Current specialist
            lora: LoRA adapter in use
        """
        if task_id is not None:
            self._current_task_id = task_id
        if tier is not None:
            self._current_tier = tier
        if specialist is not None:
            self._current_specialist = specialist
        if lora is not None:
            self._current_lora = lora

    def clear_context(self) -> None:
        """Clear call context."""
        self._current_task_id = None
        self._current_tier = None
        self._current_specialist = None
        self._current_lora = None

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate completion with logging.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional arguments passed to underlying client

        Returns:
            Generated text
        """
        call_id = f"call-{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()
        ts = datetime.utcnow().isoformat()

        try:
            # Call underlying client
            result = await self.client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            # Estimate tokens (actual counts would come from response)
            tokens_in = sum(len(m.get("content", "")) // 4 for m in messages)
            tokens_out = len(result) // 4

            # Log successful call
            record = VLLMCallRecord(
                ts=ts,
                task_id=self._current_task_id or "unknown",
                call_id=call_id,
                tier=self._current_tier or self.default_tier,
                specialist=self._current_specialist,
                lora=self._current_lora,
                input=messages,
                output=result,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                status="success",
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            self.logger.log_call(record)

            return result

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000

            # Log failed call
            record = VLLMCallRecord(
                ts=ts,
                task_id=self._current_task_id or "unknown",
                call_id=call_id,
                tier=self._current_tier or self.default_tier,
                specialist=self._current_specialist,
                lora=self._current_lora,
                input=messages,
                output=None,
                tokens_in=sum(len(m.get("content", "")) // 4 for m in messages),
                tokens_out=0,
                latency_ms=latency_ms,
                status="error",
                error=str(e),
            )
            self.logger.log_call(record)

            raise


def create_logging_client(
    client: LLMClientProtocol,
    runs_dir: Path | str = "./runs",
) -> tuple[LoggingLLMClient, TrajectoryLogger]:
    """Create a logging-wrapped LLM client.

    Args:
        client: The underlying LLM client
        runs_dir: Directory for trajectory files

    Returns:
        Tuple of (wrapped client, trajectory logger)
    """
    trajectory_logger = TrajectoryLogger(runs_dir)
    logging_client = LoggingLLMClient(client, trajectory_logger)
    return logging_client, trajectory_logger


# Utility functions for analyzing trajectories

def load_trajectory(file_path: Path | str) -> list[dict]:
    """Load trajectory from JSONL file.

    Args:
        file_path: Path to trajectory file

    Returns:
        List of records
    """
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def filter_successful_calls(records: list[dict]) -> list[dict]:
    """Filter to only successful vLLM calls.

    Args:
        records: All trajectory records

    Returns:
        Only successful vLLM call records
    """
    return [
        r for r in records
        if r.get("type") == "vllm_call" and r.get("status") == "success"
    ]


def compute_trajectory_stats(records: list[dict]) -> dict:
    """Compute statistics for a trajectory.

    Args:
        records: Trajectory records

    Returns:
        Statistics dict with call counts, token usage, latency
    """
    calls = [r for r in records if r.get("type") == "vllm_call"]

    if not calls:
        return {"total_calls": 0}

    total_tokens_in = sum(r.get("tokens_in", 0) for r in calls)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in calls)
    total_latency = sum(r.get("latency_ms", 0) for r in calls)

    success_calls = [r for r in calls if r.get("status") == "success"]
    error_calls = [r for r in calls if r.get("status") == "error"]

    tier_counts = {}
    for call in calls:
        tier = call.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return {
        "total_calls": len(calls),
        "successful_calls": len(success_calls),
        "error_calls": len(error_calls),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "total_latency_ms": total_latency,
        "avg_latency_ms": total_latency / len(calls) if calls else 0,
        "calls_by_tier": tier_counts,
    }
