"""Tests for Runner control loop.

Tests issue hi_moe-xv1: Runner control loop implementation.
"""
import pytest
import tempfile
from pathlib import Path

from .runner import (
    Runner,
    RunResult,
    RunStatus,
    TaskContext,
    RetryConfig,
    create_runner,
)
from .tiers import MockLLMClient, TaskStatus


class TestTaskContext:
    """Tests for TaskContext state management."""

    def test_get_set(self):
        ctx = TaskContext(run_id="test-123")
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_default(self):
        ctx = TaskContext(run_id="test-123")
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

    def test_append(self):
        ctx = TaskContext(run_id="test-123")
        ctx.append("logs", "first")
        ctx.append("logs", "second")
        assert ctx.get("logs") == ["first", "second"]

    def test_to_dict(self):
        ctx = TaskContext(run_id="test-123")
        ctx.set("foo", "bar")
        result = ctx.to_dict()
        assert result["run_id"] == "test-123"
        assert result["foo"] == "bar"


class TestRetryConfig:
    """Tests for RetryConfig defaults."""

    def test_defaults(self):
        config = RetryConfig()
        assert config.fleet_retries == 2
        assert config.dispatcher_retries == 1
        assert config.architect_retries == 1
        assert config.top_level_retries == 1

    def test_custom_values(self):
        config = RetryConfig(
            fleet_retries=5,
            top_level_retries=3,
        )
        assert config.fleet_retries == 5
        assert config.top_level_retries == 3


class TestRunner:
    """Tests for Runner orchestration."""

    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()

    @pytest.fixture
    def runner(self, mock_llm):
        return Runner(llm=mock_llm)

    @pytest.fixture
    def sample_problem(self):
        return {
            "id": "test-1",
            "title": "Two Sum",
            "statement": "Find two numbers that add up to target",
            "function_name": "twoSum",
            "function_signature": "def twoSum(nums: list[int], target: int) -> list[int]",
            "test_cases": [
                {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1]},
            ],
        }

    @pytest.mark.asyncio
    async def test_run_basic(self, runner, sample_problem):
        """Test basic run through hierarchy."""
        result = await runner.run(sample_problem)

        assert isinstance(result, RunResult)
        assert result.run_id.startswith("run-test-1-")
        assert result.problem == sample_problem
        assert result.context is not None

    @pytest.mark.asyncio
    async def test_run_creates_context(self, runner, sample_problem):
        """Test that run creates and populates TaskContext."""
        result = await runner.run(sample_problem)

        ctx = result.context
        assert ctx is not None
        assert ctx.get("problem") == sample_problem
        assert ctx.get("status") is not None

    @pytest.mark.asyncio
    async def test_run_generates_code(self, runner, sample_problem):
        """Test that run produces code output."""
        result = await runner.run(sample_problem)

        # With mock LLM, should get some code
        assert result.outcome is not None
        # Code might be in result or nested in outcome
        if result.code:
            assert "def" in result.code or "python" in result.code.lower()

    @pytest.mark.asyncio
    async def test_run_tracks_elapsed_time(self, runner, sample_problem):
        """Test that elapsed time is tracked."""
        result = await runner.run(sample_problem)

        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_run_with_code_runner(self, mock_llm, sample_problem):
        """Test run with code validation."""

        def mock_code_runner(code: str, test_cases: list) -> dict:
            return {
                "passed": True,
                "total_cases": len(test_cases),
                "passed_cases": len(test_cases),
            }

        runner = Runner(llm=mock_llm, code_runner=mock_code_runner)
        result = await runner.run(sample_problem)

        assert result.validation is not None
        assert result.validation.get("passed") is True


class TestRunnerWithLogging:
    """Tests for Runner trajectory logging."""

    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_problem(self):
        return {
            "id": "log-test",
            "title": "Logging Test",
            "statement": "Test problem for logging",
        }

    @pytest.mark.asyncio
    async def test_saves_trajectory(self, temp_log_dir, sample_problem):
        """Test that trajectory is saved to file."""
        runner = Runner(
            llm=MockLLMClient(),
            log_dir=temp_log_dir,
        )

        result = await runner.run(sample_problem)

        # Check trajectory file was created
        files = list(temp_log_dir.glob("*.jsonl"))
        assert len(files) == 1
        assert result.run_id in files[0].name

    @pytest.mark.asyncio
    async def test_trajectory_contains_metadata(self, temp_log_dir, sample_problem):
        """Test trajectory file contains run metadata."""
        import json

        runner = Runner(
            llm=MockLLMClient(),
            log_dir=temp_log_dir,
        )

        result = await runner.run(sample_problem)

        # Read trajectory file
        traj_file = list(temp_log_dir.glob("*.jsonl"))[0]
        with open(traj_file) as f:
            lines = [json.loads(line) for line in f]

        # First line should be run start (hi_moe-iz9 format)
        metadata = lines[0]
        assert metadata["type"] == "run_start"
        assert metadata["run_id"] == result.run_id
        assert metadata["problem_id"] == "log-test"


class TestCreateRunner:
    """Tests for create_runner factory."""

    @pytest.mark.asyncio
    async def test_create_mock_runner(self):
        """Test creating runner with mock LLM."""
        runner = await create_runner(mock=True)
        assert isinstance(runner, Runner)

    @pytest.mark.asyncio
    async def test_create_runner_requires_endpoint(self):
        """Test that endpoint is required when not mock."""
        with pytest.raises(ValueError, match="endpoint required"):
            await create_runner(mock=False)

    @pytest.mark.asyncio
    async def test_create_runner_with_log_dir(self):
        """Test creating runner with log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = await create_runner(mock=True, log_dir=tmpdir)
            assert runner.log_dir == Path(tmpdir)


class TestRoutingStrategy:
    """Tests for routing strategy detection (hi_moe-gr7)."""

    def test_math_first_detection(self):
        """Test math-first strategy detected for algorithm tasks."""
        from .tiers import RoutingDispatcher, SpecializedFleet, MockLLMClient, Task

        fleet = SpecializedFleet(MockLLMClient())
        dispatcher = RoutingDispatcher(fleet)

        task = Task(
            task_id="test-1",
            objective="Implement an optimal algorithm for sorting",
        )
        specialist, strategy, signals = dispatcher._select_specialist(task)

        assert strategy == "math_first"
        assert "math:algorithm" in signals or "math:optimal" in signals

    def test_python_direct_detection(self):
        """Test python-direct strategy for implementation tasks."""
        from .tiers import RoutingDispatcher, SpecializedFleet, MockLLMClient, Task

        fleet = SpecializedFleet(MockLLMClient())
        dispatcher = RoutingDispatcher(fleet)

        task = Task(
            task_id="test-2",
            objective="Write a Python function that adds two numbers",
        )
        specialist, strategy, signals = dispatcher._select_specialist(task)

        assert strategy == "python_direct"
        assert "python:python" in signals or "python:function" in signals or "python:write" in signals

    def test_signals_track_keywords(self):
        """Test that routing signals capture matched keywords."""
        from .tiers import RoutingDispatcher, SpecializedFleet, MockLLMClient, Task

        fleet = SpecializedFleet(MockLLMClient())
        dispatcher = RoutingDispatcher(fleet)

        # Task with only debug keywords (no code/python)
        task = Task(
            task_id="test-3",
            objective="Fix the bug in the application",
        )
        specialist, strategy, signals = dispatcher._select_specialist(task)

        assert "debug:fix" in signals or "debug:bug" in signals
        assert specialist == "debugging"

    def test_mixed_signals_priority(self):
        """Test priority when multiple keyword types match."""
        from .tiers import RoutingDispatcher, SpecializedFleet, MockLLMClient, Task

        fleet = SpecializedFleet(MockLLMClient())
        dispatcher = RoutingDispatcher(fleet)

        # Math comes first in processing, so it should be selected
        task = Task(
            task_id="test-4",
            objective="Analyze the algorithm complexity and implement in Python",
        )
        specialist, strategy, signals = dispatcher._select_specialist(task)

        # Should have both math and python signals
        assert any("math:" in s for s in signals)
        assert any("python:" in s for s in signals)
        # Math should win because it's processed first
        assert specialist == "math"
        assert strategy == "math_first"


class TestSelfHealing:
    """Tests for self-healing with code validation (hi_moe-f5d)."""

    def test_format_validation_error_wrong_answer(self):
        """Test error formatting for wrong answer."""
        from .tiers import SpecializedFleet, MockLLMClient

        fleet = SpecializedFleet(MockLLMClient())

        validation = {
            "passed": False,
            "total_passed": 1,
            "total_failed": 1,
            "test_results": [
                {"test_id": "test-1", "status": "passed"},
                {
                    "test_id": "test-2",
                    "status": "wrong_answer",
                    "expected_output": "10",
                    "actual_output": "6",
                },
            ],
        }

        error = fleet._format_validation_error(validation)

        assert "Code validation failed" in error
        assert "1/2 tests" in error
        assert "test-2" in error
        assert "WRONG_ANSWER" in error
        assert "Expected: 10" in error
        assert "Got:      6" in error

    def test_format_validation_error_runtime_error(self):
        """Test error formatting for runtime error."""
        from .tiers import SpecializedFleet, MockLLMClient

        fleet = SpecializedFleet(MockLLMClient())

        validation = {
            "passed": False,
            "total_passed": 0,
            "total_failed": 1,
            "test_results": [
                {
                    "test_id": "test-1",
                    "status": "runtime_error",
                    "error_message": "ZeroDivisionError: division by zero",
                },
            ],
        }

        error = fleet._format_validation_error(validation)

        assert "Code validation failed" in error
        assert "0/1 tests" in error
        assert "RUNTIME_ERROR" in error
        assert "ZeroDivisionError" in error

    def test_format_validation_error_timeout(self):
        """Test error formatting for timeout."""
        from .tiers import SpecializedFleet, MockLLMClient

        fleet = SpecializedFleet(MockLLMClient())

        validation = {
            "passed": False,
            "total_passed": 0,
            "total_failed": 1,
            "test_results": [
                {
                    "test_id": "test-1",
                    "status": "timeout",
                    "error_message": "Timeout after 5s",
                },
            ],
        }

        error = fleet._format_validation_error(validation)

        assert "TIMEOUT" in error
        assert "Timeout after 5s" in error

    @pytest.mark.asyncio
    async def test_fleet_with_code_runner_pass(self):
        """Test Fleet passes validation with correct code."""
        from .tiers import SpecializedFleet, MockLLMClient, Task, TaskStatus

        def mock_runner(code, test_cases):
            return {"passed": True, "total_passed": 1, "total_failed": 0, "test_results": []}

        fleet = SpecializedFleet(MockLLMClient(), code_runner=mock_runner)

        task = Task(
            task_id="test-1",
            objective="Write a function",
            context={"test_cases": [{"input": "5", "expected": "10"}]},
        )

        outcome = await fleet.execute(task, "python")

        assert outcome.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fleet_with_code_runner_fail(self):
        """Test Fleet fails validation with wrong code."""
        from .tiers import SpecializedFleet, MockLLMClient, Task, TaskStatus

        def mock_runner(code, test_cases):
            return {
                "passed": False,
                "total_passed": 0,
                "total_failed": 1,
                "test_results": [
                    {"test_id": "test-1", "status": "wrong_answer", "expected_output": "10", "actual_output": "6"}
                ],
            }

        fleet = SpecializedFleet(MockLLMClient(), code_runner=mock_runner)

        task = Task(
            task_id="test-1",
            objective="Write a function",
            context={"test_cases": [{"input": "5", "expected": "10"}]},
        )

        # Should fail because validation fails and retries are exhausted
        outcome = await fleet.execute(task, "python")

        assert outcome.status == TaskStatus.FAILED
        assert "Code validation failed" in outcome.error

    @pytest.mark.asyncio
    async def test_fleet_no_code_runner(self):
        """Test Fleet works without code runner (no validation)."""
        from .tiers import SpecializedFleet, MockLLMClient, Task, TaskStatus

        fleet = SpecializedFleet(MockLLMClient())  # No code_runner

        task = Task(
            task_id="test-1",
            objective="Write a function",
            context={"test_cases": [{"input": "5", "expected": "10"}]},
        )

        outcome = await fleet.execute(task, "python")

        # Should complete without validation
        assert outcome.status == TaskStatus.COMPLETED
