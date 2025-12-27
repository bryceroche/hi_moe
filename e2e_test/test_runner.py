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


class TestMultiTurnContext:
    """Tests for multi-turn context (hi_moe-ceg)."""

    def test_specialist_stats_success_rate(self):
        """Test specialist success rate calculation."""
        from .tiers import SpecialistStats

        stats = SpecialistStats()
        assert stats.success_rate == 0.5  # Default when no data

        stats.record_success(100)
        assert stats.success_rate == 1.0

        stats.record_failure(100)
        assert stats.success_rate == 0.5

        stats.record_success(100)
        stats.record_success(100)
        assert stats.success_rate == 0.75  # 3/4

    def test_conversation_context_record_outcome(self):
        """Test recording outcomes to context."""
        from .tiers import ConversationContext, Task, Outcome, TaskStatus

        ctx = ConversationContext(session_id="test-session")

        task = Task(task_id="task-1", objective="Write a function")
        outcome = Outcome(
            task_id="task-1",
            status=TaskStatus.COMPLETED,
            execution_time_ms=100,
        )

        ctx.record_outcome(task, outcome, "python", code="def foo(): pass")

        assert "python" in ctx.specialist_stats
        assert ctx.specialist_stats["python"].successes == 1
        assert len(ctx.solution_history) == 1
        assert ctx.solution_history[0].success is True

    def test_conversation_context_specialist_preference(self):
        """Test specialist preference ordering by success rate."""
        from .tiers import ConversationContext, Task, Outcome, TaskStatus

        ctx = ConversationContext(session_id="test-session")

        # Python: 2 success, 0 failure = 100%
        for i in range(2):
            ctx.record_outcome(
                Task(task_id=f"task-p{i}", objective="python task"),
                Outcome(task_id=f"task-p{i}", status=TaskStatus.COMPLETED, execution_time_ms=100),
                "python",
            )

        # Math: 1 success, 1 failure = 50%
        ctx.record_outcome(
            Task(task_id="task-m1", objective="math task"),
            Outcome(task_id="task-m1", status=TaskStatus.COMPLETED, execution_time_ms=100),
            "math",
        )
        ctx.record_outcome(
            Task(task_id="task-m2", objective="math task"),
            Outcome(task_id="task-m2", status=TaskStatus.FAILED, execution_time_ms=100),
            "math",
        )

        # Debug: 0 success, 1 failure = 0%
        ctx.record_outcome(
            Task(task_id="task-d1", objective="debug task"),
            Outcome(task_id="task-d1", status=TaskStatus.FAILED, execution_time_ms=100),
            "debugging",
        )

        preference = ctx.get_specialist_preference()
        assert preference[0] == "python"  # 100% success rate
        assert preference[1] == "math"     # 50% success rate, 2 attempts
        assert preference[2] == "debugging"  # 0% success rate

    def test_conversation_context_relevant_solutions(self):
        """Test finding relevant previous solutions."""
        from .tiers import ConversationContext, Task, Outcome, TaskStatus

        ctx = ConversationContext(session_id="test-session")

        # Add some solutions
        ctx.record_outcome(
            Task(task_id="task-1", objective="Write a function to sort numbers"),
            Outcome(task_id="task-1", status=TaskStatus.COMPLETED, execution_time_ms=100),
            "python",
            code="def sort_numbers(nums): return sorted(nums)",
        )
        ctx.record_outcome(
            Task(task_id="task-2", objective="Implement a binary search"),
            Outcome(task_id="task-2", status=TaskStatus.COMPLETED, execution_time_ms=100),
            "algorithms",
            code="def binary_search(arr, x): pass",
        )

        # Query for sorting related
        relevant = ctx.get_relevant_solutions("Sort an array of numbers")
        assert len(relevant) >= 1
        assert any("sort" in sol.code for sol in relevant)

    def test_conversation_context_max_history(self):
        """Test that solution history is capped at max_history."""
        from .tiers import ConversationContext, Task, Outcome, TaskStatus

        ctx = ConversationContext(session_id="test-session", max_history=3)

        # Add 5 solutions
        for i in range(5):
            ctx.record_outcome(
                Task(task_id=f"task-{i}", objective=f"Task {i}"),
                Outcome(task_id=f"task-{i}", status=TaskStatus.COMPLETED, execution_time_ms=100),
                "python",
                code=f"code_{i}",
            )

        assert len(ctx.solution_history) == 3
        # Should keep the last 3
        assert ctx.solution_history[0].code == "code_2"
        assert ctx.solution_history[2].code == "code_4"

    def test_dispatcher_with_context_preference(self):
        """Test dispatcher uses context for specialist preference."""
        from .tiers import RoutingDispatcher, SpecializedFleet, MockLLMClient, Task, ConversationContext

        ctx = ConversationContext(session_id="test-session")

        # Build up history: math has 100% success
        from .tiers import Outcome, TaskStatus
        ctx.record_outcome(
            Task(task_id="t1", objective="math task"),
            Outcome(task_id="t1", status=TaskStatus.COMPLETED, execution_time_ms=100),
            "math",
        )
        ctx.record_outcome(
            Task(task_id="t2", objective="math task 2"),
            Outcome(task_id="t2", status=TaskStatus.COMPLETED, execution_time_ms=100),
            "math",
        )

        fleet = SpecializedFleet(MockLLMClient())
        dispatcher = RoutingDispatcher(fleet, conversation_context=ctx)

        # Task with no specific keywords - should prefer math due to history
        task = Task(task_id="t3", objective="Do a general task")
        specialist, strategy, signals = dispatcher._select_specialist(task)

        # Context preference should be in signals
        assert any("context:preferred" in s for s in signals)

    def test_context_summary(self):
        """Test context summary generation."""
        from .tiers import ConversationContext, Task, Outcome, TaskStatus

        ctx = ConversationContext(session_id="test-session")
        ctx.record_outcome(
            Task(task_id="t1", objective="task"),
            Outcome(task_id="t1", status=TaskStatus.COMPLETED, execution_time_ms=100),
            "python",
            code="code",
        )
        ctx.add_message("user", "Hello")

        summary = ctx.get_context_summary()
        assert summary["session_id"] == "test-session"
        assert summary["total_tasks"] == 1
        assert summary["solutions_stored"] == 1
        assert summary["messages"] == 1
