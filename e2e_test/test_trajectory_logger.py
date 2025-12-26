"""Tests for trajectory logging.

Tests issue hi_moe-iz9: Runner trajectory logging for vLLM calls.
"""
import json
import pytest
import tempfile
from pathlib import Path

from .trajectory_logger import (
    VLLMCallRecord,
    TrajectoryLogger,
    LoggingLLMClient,
    create_logging_client,
    load_trajectory,
    filter_successful_calls,
    compute_trajectory_stats,
)
from .tiers import MockLLMClient


class TestVLLMCallRecord:
    """Tests for VLLMCallRecord dataclass."""

    def test_basic_record(self):
        record = VLLMCallRecord(
            ts="2024-01-01T00:00:00",
            task_id="test-123",
            call_id="call-abc",
            tier="fleet",
        )
        assert record.task_id == "test-123"
        assert record.status == "success"

    def test_to_dict_excludes_none(self):
        record = VLLMCallRecord(
            ts="2024-01-01T00:00:00",
            task_id="test-123",
            call_id="call-abc",
            tier="fleet",
            specialist=None,  # Should be excluded
            lora=None,  # Should be excluded
        )
        d = record.to_dict()
        assert "specialist" not in d
        assert "lora" not in d
        assert d["tier"] == "fleet"

    def test_full_record(self):
        record = VLLMCallRecord(
            ts="2024-01-01T00:00:00",
            task_id="test-123",
            call_id="call-abc",
            tier="fleet",
            specialist="python",
            lora="python-lora",
            input=[{"role": "user", "content": "test"}],
            output="response",
            tokens_in=10,
            tokens_out=20,
            latency_ms=100.5,
            status="success",
        )
        d = record.to_dict()
        assert d["specialist"] == "python"
        assert d["tokens_in"] == 10
        assert d["latency_ms"] == 100.5


class TestTrajectoryLogger:
    """Tests for TrajectoryLogger."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_creates_runs_dir(self, temp_dir):
        logger = TrajectoryLogger(temp_dir / "runs")
        assert (temp_dir / "runs").exists()

    def test_start_run(self, temp_dir):
        logger = TrajectoryLogger(temp_dir)
        logger.start_run("test-run-1", {"problem_id": "p1"})

        assert logger.current_run_id == "test-run-1"
        assert (temp_dir / "test-run-1.jsonl").exists()

    def test_log_call(self, temp_dir):
        logger = TrajectoryLogger(temp_dir)
        logger.start_run("test-run-1")

        record = VLLMCallRecord(
            ts="2024-01-01T00:00:00",
            task_id="test-123",
            call_id="call-abc",
            tier="fleet",
        )
        logger.log_call(record)

        # Read file and verify
        with open(temp_dir / "test-run-1.jsonl") as f:
            lines = [json.loads(line) for line in f]

        assert len(lines) == 2  # start + call
        assert lines[1]["type"] == "vllm_call"
        assert lines[1]["tier"] == "fleet"

    def test_end_run(self, temp_dir):
        logger = TrajectoryLogger(temp_dir)
        logger.start_run("test-run-1")
        logger.end_run({"success": True})

        with open(temp_dir / "test-run-1.jsonl") as f:
            lines = [json.loads(line) for line in f]

        assert lines[-1]["type"] == "run_end"
        assert lines[-1]["success"] is True

    def test_log_without_run_warns(self, temp_dir, caplog):
        logger = TrajectoryLogger(temp_dir)

        record = VLLMCallRecord(
            ts="2024-01-01T00:00:00",
            task_id="test-123",
            call_id="call-abc",
            tier="fleet",
        )
        logger.log_call(record)

        assert "No active run" in caplog.text


class TestLoggingLLMClient:
    """Tests for LoggingLLMClient wrapper."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_client(self):
        return MockLLMClient()

    @pytest.mark.asyncio
    async def test_wraps_client(self, temp_dir, mock_client):
        traj_logger = TrajectoryLogger(temp_dir)
        logging_client = LoggingLLMClient(mock_client, traj_logger)

        traj_logger.start_run("test-run")
        logging_client.set_context(task_id="task-1", tier="fleet")

        result = await logging_client.generate(
            [{"role": "user", "content": "Hello"}]
        )

        assert result is not None  # Should get response from mock

        traj_logger.end_run()

        # Verify call was logged
        records = load_trajectory(temp_dir / "test-run.jsonl")
        calls = [r for r in records if r.get("type") == "vllm_call"]
        assert len(calls) == 1
        assert calls[0]["tier"] == "fleet"
        assert calls[0]["task_id"] == "task-1"

    @pytest.mark.asyncio
    async def test_logs_latency(self, temp_dir, mock_client):
        traj_logger = TrajectoryLogger(temp_dir)
        logging_client = LoggingLLMClient(mock_client, traj_logger)

        traj_logger.start_run("test-run")

        await logging_client.generate([{"role": "user", "content": "Hello"}])

        traj_logger.end_run()

        records = load_trajectory(temp_dir / "test-run.jsonl")
        calls = [r for r in records if r.get("type") == "vllm_call"]
        assert calls[0]["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_context_management(self, temp_dir, mock_client):
        traj_logger = TrajectoryLogger(temp_dir)
        logging_client = LoggingLLMClient(mock_client, traj_logger)

        traj_logger.start_run("test-run")

        # Set context
        logging_client.set_context(
            task_id="task-1",
            tier="architect",
            specialist="planning",
        )

        await logging_client.generate([{"role": "user", "content": "Plan"}])

        # Change context
        logging_client.set_context(tier="fleet", specialist="python")

        await logging_client.generate([{"role": "user", "content": "Code"}])

        traj_logger.end_run()

        records = load_trajectory(temp_dir / "test-run.jsonl")
        calls = [r for r in records if r.get("type") == "vllm_call"]

        assert calls[0]["tier"] == "architect"
        assert calls[1]["tier"] == "fleet"
        assert calls[1]["specialist"] == "python"


class TestUtilityFunctions:
    """Tests for trajectory analysis utilities."""

    @pytest.fixture
    def sample_records(self):
        return [
            {"type": "run_start", "run_id": "test"},
            {
                "type": "vllm_call",
                "status": "success",
                "tier": "architect",
                "tokens_in": 100,
                "tokens_out": 50,
                "latency_ms": 200,
            },
            {
                "type": "vllm_call",
                "status": "success",
                "tier": "fleet",
                "tokens_in": 80,
                "tokens_out": 100,
                "latency_ms": 300,
            },
            {
                "type": "vllm_call",
                "status": "error",
                "tier": "fleet",
                "tokens_in": 50,
                "tokens_out": 0,
                "latency_ms": 50,
            },
            {"type": "run_end", "run_id": "test"},
        ]

    def test_filter_successful_calls(self, sample_records):
        successful = filter_successful_calls(sample_records)
        assert len(successful) == 2
        assert all(r["status"] == "success" for r in successful)

    def test_compute_trajectory_stats(self, sample_records):
        stats = compute_trajectory_stats(sample_records)

        assert stats["total_calls"] == 3
        assert stats["successful_calls"] == 2
        assert stats["error_calls"] == 1
        assert stats["total_tokens_in"] == 230
        assert stats["total_tokens_out"] == 150
        assert stats["total_tokens"] == 380
        assert stats["total_latency_ms"] == 550
        assert stats["calls_by_tier"] == {"architect": 1, "fleet": 2}

    def test_compute_stats_empty(self):
        stats = compute_trajectory_stats([])
        assert stats["total_calls"] == 0


class TestCreateLoggingClient:
    """Tests for create_logging_client factory."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_creates_both(self, temp_dir):
        mock_client = MockLLMClient()
        logging_client, traj_logger = create_logging_client(mock_client, temp_dir)

        assert isinstance(logging_client, LoggingLLMClient)
        assert isinstance(traj_logger, TrajectoryLogger)
