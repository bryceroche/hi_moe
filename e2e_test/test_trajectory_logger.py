"""Tests for trajectory logging.

Tests issue hi_moe-iz9: Runner trajectory logging for vLLM calls.
Tests issue hi_moe-r8q: Tier-specific logging for training data collection.
"""
import json
import pytest
import tempfile
from pathlib import Path

from .trajectory_logger import (
    VLLMCallRecord,
    ArchitectRecord,
    DispatcherRecord,
    FleetRecord,
    TrajectoryLogger,
    LoggingLLMClient,
    create_logging_client,
    load_trajectory,
    filter_successful_calls,
    compute_trajectory_stats,
    filter_architect_records,
    filter_dispatcher_records,
    filter_fleet_records,
    compute_tier_stats,
    extract_training_pairs,
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


# Tests for hi_moe-r8q: Tier-specific logging for training data collection


class TestArchitectRecord:
    """Tests for ArchitectRecord dataclass."""

    def test_basic_record(self):
        record = ArchitectRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            goal="Solve two sum problem",
        )
        assert record.task_id == "task-123"
        assert record.goal == "Solve two sum problem"

    def test_to_dict_includes_type(self):
        record = ArchitectRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            goal="Solve problem",
        )
        d = record.to_dict()
        assert d["type"] == "architect_decision"
        assert d["goal"] == "Solve problem"

    def test_full_record(self):
        record = ArchitectRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            goal="Solve two sum problem",
            plan="1. Use hash map\n2. Find complement",
            delegation={"task_id": "task-123-impl", "objective": "Implement solution"},
            success_criteria=["Function returns correct indices", "O(n) complexity"],
            outcome_status="completed",
            outcome_summary="Completed with code",
        )
        d = record.to_dict()
        assert d["plan"] == "1. Use hash map\n2. Find complement"
        assert d["delegation"]["task_id"] == "task-123-impl"
        assert len(d["success_criteria"]) == 2


class TestDispatcherRecord:
    """Tests for DispatcherRecord dataclass."""

    def test_basic_record(self):
        record = DispatcherRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Implement solution",
            routing_decision="structured_plan",
        )
        assert record.routing_decision == "structured_plan"

    def test_to_dict_includes_type(self):
        record = DispatcherRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Implement solution",
            routing_decision="heuristic",
        )
        d = record.to_dict()
        assert d["type"] == "dispatcher_routing"

    def test_structured_plan_record(self):
        record = DispatcherRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Implement solution",
            routing_decision="structured_plan",
            plan_steps=[
                {"description": "Analyze approach", "specialist": "math"},
                {"description": "Implement code", "specialist": "python"},
            ],
            rationale="LLM generated 2-step plan",
            outcome_status="completed",
        )
        d = record.to_dict()
        assert len(d["plan_steps"]) == 2
        assert d["plan_steps"][0]["specialist"] == "math"


class TestFleetRecord:
    """Tests for FleetRecord dataclass."""

    def test_basic_record(self):
        record = FleetRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Write function",
            specialist="python",
        )
        assert record.specialist == "python"
        assert record.status == "success"

    def test_to_dict_includes_type(self):
        record = FleetRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Write function",
            specialist="python",
        )
        d = record.to_dict()
        assert d["type"] == "fleet_execution"

    def test_full_record(self):
        record = FleetRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Write two sum function",
            specialist="python",
            prompt_used="You are a Python expert...",
            output_code="def twoSum(nums, target):\n    pass",
            output_raw="```python\ndef twoSum(nums, target):\n    pass\n```",
            execution_time_ms=150.5,
            status="success",
        )
        d = record.to_dict()
        assert "def twoSum" in d["output_code"]
        assert d["execution_time_ms"] == 150.5


class TestTierLogging:
    """Tests for TrajectoryLogger tier-specific logging."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_log_architect(self, temp_dir):
        logger = TrajectoryLogger(temp_dir)
        logger.start_run("test-run")

        record = ArchitectRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            goal="Solve problem",
            plan="Step 1, Step 2",
            outcome_status="completed",
        )
        logger.log_architect(record)
        logger.end_run()

        records = load_trajectory(temp_dir / "test-run.jsonl")
        architect_records = filter_architect_records(records)
        assert len(architect_records) == 1
        assert architect_records[0]["goal"] == "Solve problem"

    def test_log_dispatcher(self, temp_dir):
        logger = TrajectoryLogger(temp_dir)
        logger.start_run("test-run")

        record = DispatcherRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Implement solution",
            routing_decision="structured_plan",
            plan_steps=[{"description": "Code it", "specialist": "python"}],
            outcome_status="completed",
        )
        logger.log_dispatcher(record)
        logger.end_run()

        records = load_trajectory(temp_dir / "test-run.jsonl")
        dispatcher_records = filter_dispatcher_records(records)
        assert len(dispatcher_records) == 1
        assert dispatcher_records[0]["routing_decision"] == "structured_plan"

    def test_log_fleet(self, temp_dir):
        logger = TrajectoryLogger(temp_dir)
        logger.start_run("test-run")

        record = FleetRecord(
            ts="2024-01-01T00:00:00",
            task_id="task-123",
            task_objective="Write function",
            specialist="python",
            output_code="def solve(): pass",
            status="success",
        )
        logger.log_fleet(record)
        logger.end_run()

        records = load_trajectory(temp_dir / "test-run.jsonl")
        fleet_records = filter_fleet_records(records)
        assert len(fleet_records) == 1
        assert fleet_records[0]["specialist"] == "python"


class TestTierAnalysisFunctions:
    """Tests for tier-specific analysis utilities."""

    @pytest.fixture
    def sample_tier_records(self):
        return [
            {"type": "run_start", "run_id": "test"},
            {
                "type": "architect_decision",
                "task_id": "task-1",
                "goal": "Solve problem",
                "outcome_status": "completed",
            },
            {
                "type": "dispatcher_routing",
                "task_id": "task-1",
                "routing_decision": "structured_plan",
                "outcome_status": "completed",
            },
            {
                "type": "fleet_execution",
                "task_id": "task-1",
                "specialist": "python",
                "status": "success",
            },
            {
                "type": "dispatcher_routing",
                "task_id": "task-2",
                "routing_decision": "heuristic",
                "outcome_status": "failed",
            },
            {
                "type": "fleet_execution",
                "task_id": "task-2",
                "specialist": "math",
                "status": "error",
            },
            {"type": "run_end", "run_id": "test"},
        ]

    def test_filter_architect_records(self, sample_tier_records):
        result = filter_architect_records(sample_tier_records)
        assert len(result) == 1
        assert result[0]["goal"] == "Solve problem"

    def test_filter_dispatcher_records(self, sample_tier_records):
        result = filter_dispatcher_records(sample_tier_records)
        assert len(result) == 2

    def test_filter_fleet_records(self, sample_tier_records):
        result = filter_fleet_records(sample_tier_records)
        assert len(result) == 2
        specialists = [r["specialist"] for r in result]
        assert "python" in specialists
        assert "math" in specialists

    def test_compute_tier_stats(self, sample_tier_records):
        stats = compute_tier_stats(sample_tier_records)

        assert stats["architect_decisions"] == 1
        assert stats["architect_success_rate"] == 1.0
        assert stats["dispatcher_routings"] == 2
        assert stats["dispatcher_success_rate"] == 0.5
        assert stats["structured_routing_count"] == 1
        assert stats["heuristic_routing_count"] == 1
        assert stats["fleet_executions"] == 2
        assert stats["fleet_success_rate"] == 0.5
        assert stats["specialist_usage"] == {"python": 1, "math": 1}

    def test_compute_tier_stats_empty(self):
        stats = compute_tier_stats([])
        assert stats["architect_decisions"] == 0
        assert stats["fleet_executions"] == 0


class TestExtractTrainingPairs:
    """Tests for extract_training_pairs function."""

    @pytest.fixture
    def training_records(self):
        return [
            {
                "type": "architect_decision",
                "task_id": "task-1",
                "goal": "Solve two sum",
                "plan": "Use hash map",
                "delegation": {"task_id": "impl", "objective": "Implement"},
                "success_criteria": ["Correct output"],
                "outcome_status": "completed",
                "outcome_summary": "Success",
                "metadata": {"context": {"problem": "two sum"}},
            },
            {
                "type": "dispatcher_routing",
                "task_id": "task-1",
                "task_objective": "Implement solution",
                "routing_decision": "structured_plan",
                "plan_steps": [{"description": "Code", "specialist": "python"}],
                "rationale": "Single step",
                "context_summary": "Two sum problem",
                "outcome_status": "completed",
            },
            {
                "type": "fleet_execution",
                "task_id": "task-1",
                "task_objective": "Write code",
                "specialist": "python",
                "prompt_used": "You are a Python expert",
                "output_code": "def twoSum(): pass",
                "status": "success",
                "validation_result": {"passed": True},
            },
            # Failed records should not be included
            {
                "type": "architect_decision",
                "task_id": "task-2",
                "goal": "Solve problem",
                "outcome_status": "failed",
            },
            {
                "type": "fleet_execution",
                "task_id": "task-2",
                "task_objective": "Write code",
                "specialist": "python",
                "status": "error",
            },
        ]

    def test_extract_architect_pairs(self, training_records):
        pairs = extract_training_pairs(training_records, "architect")
        assert len(pairs) == 1
        assert pairs[0]["input"]["goal"] == "Solve two sum"
        assert pairs[0]["output"]["plan"] == "Use hash map"

    def test_extract_dispatcher_pairs(self, training_records):
        pairs = extract_training_pairs(training_records, "dispatcher")
        assert len(pairs) == 1
        assert pairs[0]["input"]["task"] == "Implement solution"
        assert pairs[0]["output"]["routing_decision"] == "structured_plan"

    def test_extract_fleet_pairs(self, training_records):
        pairs = extract_training_pairs(training_records, "fleet")
        assert len(pairs) == 1
        assert pairs[0]["input"]["specialist"] == "python"
        assert "def twoSum" in pairs[0]["output"]["code"]
        assert pairs[0]["validation"]["passed"] is True

    def test_extract_ignores_failed(self, training_records):
        # Failed architect and fleet records should be filtered out
        architect_pairs = extract_training_pairs(training_records, "architect")
        fleet_pairs = extract_training_pairs(training_records, "fleet")
        assert len(architect_pairs) == 1  # Only the successful one
        assert len(fleet_pairs) == 1  # Only the successful one


# Tests for hi_moe-vo8: Trajectory data collection for training


from .trajectory_logger import (
    TrainingExample,
    TrajectoryDataCollector,
    collect_training_data,
)


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""

    def test_basic_example(self):
        example = TrainingExample(
            domain="python",
            problem="Write a function to add two numbers",
            reasoning="Using basic arithmetic",
            solution="def add(a, b):\n    return a + b",
        )
        assert example.domain == "python"
        assert "add" in example.solution

    def test_to_dict(self):
        example = TrainingExample(
            domain="python",
            problem="Write a function",
            reasoning="Approach step by step",
            solution="def solve(): pass",
        )
        d = example.to_dict()
        assert d["domain"] == "python"
        assert d["problem"] == "Write a function"
        assert d["reasoning"] == "Approach step by step"
        assert d["solution"] == "def solve(): pass"


class TestTrajectoryDataCollector:
    """Tests for TrajectoryDataCollector class."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_trajectory_file(self, temp_dir):
        """Create a sample trajectory JSONL file."""
        records = [
            {"type": "run_start", "run_id": "test-run"},
            {
                "type": "fleet_execution",
                "task_id": "task-1",
                "task_objective": "Write two sum function",
                "specialist": "python",
                "prompt_used": "You are a Python expert. Solve step by step.",
                "output_code": "def twoSum(nums, target):\n    seen = {}\n    for i, n in enumerate(nums):\n        if target - n in seen:\n            return [seen[target-n], i]\n        seen[n] = i",
                "status": "success",
                "validation_result": {"passed": True, "total_passed": 3},
            },
            {
                "type": "fleet_execution",
                "task_id": "task-2",
                "task_objective": "Write fibonacci",
                "specialist": "python",
                "output_code": "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
                "status": "success",
                "validation_result": {"passed": True},
            },
            {
                "type": "fleet_execution",
                "task_id": "task-3",
                "task_objective": "Failed task",
                "specialist": "python",
                "output_code": "broken",
                "status": "success",
                "validation_result": {"passed": False, "total_failed": 2},
            },
            {
                "type": "dispatcher_routing",
                "task_id": "task-1",
                "task_objective": "Implement two sum",
                "routing_decision": "structured_plan",
                "routing_strategy": "python_direct",
                "plan_steps": ["Analyze problem", "Implement solution"],
                "rationale": "Simple implementation task",
                "context_summary": "Two sum problem from LeetCode",
                "outcome_status": "completed",
            },
            {"type": "run_end", "run_id": "test-run"},
        ]
        file_path = temp_dir / "test-run.jsonl"
        with open(file_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return temp_dir

    def test_load_all_trajectories(self, sample_trajectory_file):
        collector = TrajectoryDataCollector(sample_trajectory_file)
        records = collector.load_all_trajectories()

        assert len(records) == 6  # All records from the file
        fleet_records = [r for r in records if r.get("type") == "fleet_execution"]
        assert len(fleet_records) == 3

    def test_load_trajectories_missing_dir(self, temp_dir):
        collector = TrajectoryDataCollector(temp_dir / "nonexistent")
        records = collector.load_all_trajectories()
        assert records == []

    def test_collect_fleet_examples_with_validation(self, sample_trajectory_file):
        collector = TrajectoryDataCollector(sample_trajectory_file)
        examples = collector.collect_fleet_examples(require_validation=True)

        # Only 2 examples passed validation
        assert len(examples) == 2
        assert all(isinstance(e, TrainingExample) for e in examples)
        assert examples[0].domain == "python"
        assert "twoSum" in examples[0].solution

    def test_collect_fleet_examples_without_validation(self, sample_trajectory_file):
        collector = TrajectoryDataCollector(sample_trajectory_file)
        examples = collector.collect_fleet_examples(require_validation=False, min_code_length=1)

        # All 3 fleet records included (validation not required, min_code_length=1)
        assert len(examples) == 3

    def test_collect_fleet_examples_min_code_length(self, sample_trajectory_file):
        collector = TrajectoryDataCollector(sample_trajectory_file)
        examples = collector.collect_fleet_examples(
            require_validation=False,
            min_code_length=50,  # Excludes short code
        )

        # Only includes code longer than 50 chars
        for example in examples:
            assert len(example.solution) >= 50

    def test_collect_dispatcher_examples(self, sample_trajectory_file):
        collector = TrajectoryDataCollector(sample_trajectory_file)
        examples = collector.collect_dispatcher_examples()

        assert len(examples) == 1
        assert examples[0].domain == "routing"
        assert "ROUTING: structured_plan" in examples[0].solution
        assert "python_direct" in examples[0].solution

    def test_write_training_file(self, temp_dir):
        collector = TrajectoryDataCollector(temp_dir)
        examples = [
            TrainingExample(
                domain="python",
                problem="Test problem",
                reasoning="Test reasoning",
                solution="def test(): pass",
            ),
        ]

        output_path = temp_dir / "output" / "training.jsonl"
        count = collector.write_training_file(examples, output_path)

        assert count == 1
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 1
        assert lines[0]["domain"] == "python"
        assert lines[0]["solution"] == "def test(): pass"

    def test_specialist_to_domain(self, temp_dir):
        collector = TrajectoryDataCollector(temp_dir)

        assert collector._specialist_to_domain("python") == "python"
        assert collector._specialist_to_domain("math") == "math"
        assert collector._specialist_to_domain("algorithm") == "algorithms"
        assert collector._specialist_to_domain("unknown") == "python"  # Default

    def test_build_reasoning(self, temp_dir):
        collector = TrajectoryDataCollector(temp_dir)

        # Test with step by step hint
        reasoning = collector._build_reasoning(
            prompt="Solve this step by step",
            task="Two sum problem",
            specialist="python",
        )
        assert "step by step" in reasoning.lower()
        assert "python" in reasoning.lower()

        # Test with edge cases hint
        reasoning = collector._build_reasoning(
            prompt="Consider edge cases carefully",
            task="Sort array",
            specialist="algorithm",
        )
        assert "edge cases" in reasoning.lower()


class TestCollectTrainingDataConvenience:
    """Tests for collect_training_data convenience function."""

    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as runs_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                # Create a sample run
                records = [
                    {"type": "run_start", "run_id": "run-1"},
                    {
                        "type": "fleet_execution",
                        "task_id": "task-1",
                        "task_objective": "Write code",
                        "specialist": "python",
                        "output_code": "def solve(): return 42",
                        "status": "success",
                        "validation_result": {"passed": True},
                    },
                    {
                        "type": "dispatcher_routing",
                        "task_id": "task-1",
                        "task_objective": "Route task",
                        "routing_decision": "heuristic",
                        "routing_strategy": "python_direct",
                        "routing_signals": ["simple_task"],
                        "rationale": "Direct to python",
                        "outcome_status": "completed",
                    },
                    {"type": "run_end", "run_id": "run-1"},
                ]
                with open(Path(runs_dir) / "run-1.jsonl", "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")

                yield Path(runs_dir), Path(output_dir)

    def test_collect_all(self, temp_dirs):
        runs_dir, output_dir = temp_dirs

        summary = collect_training_data(
            runs_dir=runs_dir,
            output_dir=output_dir,
            include_fleet=True,
            include_dispatcher=True,
        )

        assert summary["fleet_examples"] == 1
        assert summary["dispatcher_examples"] == 1
        assert (output_dir / "fleet_training.jsonl").exists()
        assert (output_dir / "dispatcher_training.jsonl").exists()

    def test_collect_fleet_only(self, temp_dirs):
        runs_dir, output_dir = temp_dirs

        summary = collect_training_data(
            runs_dir=runs_dir,
            output_dir=output_dir,
            include_fleet=True,
            include_dispatcher=False,
        )

        assert summary["fleet_examples"] == 1
        assert summary["dispatcher_examples"] == 0
        assert (output_dir / "fleet_training.jsonl").exists()
        assert not (output_dir / "dispatcher_training.jsonl").exists()

    def test_collect_empty_runs(self):
        with tempfile.TemporaryDirectory() as runs_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                summary = collect_training_data(
                    runs_dir=runs_dir,
                    output_dir=output_dir,
                )
                assert summary["fleet_examples"] == 0
                assert summary["dispatcher_examples"] == 0
