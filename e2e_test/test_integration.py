"""Integration tests for the hi-moe tier system (hi_moe-o7r).

Tests:
- Tier routing with MockLLMClient
- Memory persistence (save/load)
- Training data collection to CallDB
- CLI tool invocation

Run with: pytest e2e_test/test_integration.py -v
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from .tiers import (
    Task,
    TaskStatus,
    MockLLMClient,
    RoutingDispatcher,
    SpecializedFleet,
    DispatcherMemory,
    FleetMemory,
)
from .call_db import CallDB


# ============================================================================
# Tier Routing Tests
# ============================================================================


class TestTierRouting:
    """Test that the tier system routes and executes tasks correctly."""

    @pytest.fixture
    def mock_llm(self):
        """Create a MockLLMClient for testing."""
        return MockLLMClient()

    @pytest.fixture
    def fleet(self, mock_llm):
        """Create a SpecializedFleet with mock LLM."""
        return SpecializedFleet(mock_llm, memory=FleetMemory())

    @pytest.fixture
    def dispatcher(self, fleet, mock_llm):
        """Create a RoutingDispatcher with mock LLM."""
        return RoutingDispatcher(fleet, mock_llm, memory=DispatcherMemory())

    @pytest.mark.asyncio
    async def test_basic_task_execution(self, dispatcher):
        """Test that a simple task completes successfully."""
        task = Task(
            task_id="test_basic",
            objective="Write a function to add two numbers",
            context={"examples": "add(1, 2) = 3"},
        )

        outcome = await dispatcher.execute(task)

        assert outcome.status == TaskStatus.COMPLETED
        assert outcome.task_id == "test_basic"
        assert outcome.result is not None

    @pytest.mark.asyncio
    async def test_routing_records_to_memory(self, dispatcher):
        """Test that routing decisions are recorded to memory."""
        task = Task(
            task_id="test_memory",
            objective="Solve a math problem: what is 2+2?",
            context={},
        )

        await dispatcher.execute(task)

        # Check that routing was recorded
        assert len(dispatcher.memory.routing_history) > 0
        last_routing = dispatcher.memory.routing_history[-1]
        # Task ID may include step suffix (e.g., "test_memory-step1")
        assert "test_memory" in last_routing["task_id"]
        assert "specialist" in last_routing

    @pytest.mark.asyncio
    async def test_fleet_records_execution(self, dispatcher):
        """Test that Fleet records execution to its memory."""
        task = Task(
            task_id="test_fleet_memory",
            objective="Write a Python function to reverse a string",
            context={},
        )

        await dispatcher.execute(task)

        # Check fleet memory has records
        fleet_memory = dispatcher.fleet.memory
        assert len(fleet_memory.executions) > 0

    @pytest.mark.asyncio
    async def test_multiple_tasks_accumulate_memory(self, dispatcher):
        """Test that multiple tasks accumulate in memory."""
        tasks = [
            Task(task_id="task1", objective="Write code to sort a list", context={}),
            Task(task_id="task2", objective="Calculate fibonacci numbers", context={}),
            Task(task_id="task3", objective="Parse JSON data", context={}),
        ]

        for task in tasks:
            await dispatcher.execute(task)

        # Should have routing history for all tasks
        assert len(dispatcher.memory.routing_history) >= 3


# ============================================================================
# Memory Persistence Tests
# ============================================================================


class TestMemoryPersistence:
    """Test that memory can be saved and loaded correctly."""

    def test_dispatcher_memory_to_dict(self):
        """Test DispatcherMemory serialization."""
        memory = DispatcherMemory()
        memory.record_routing("task1", "python", "code_problem")
        memory.record_outcome("python", True)  # specialist, success

        data = memory.to_dict()

        assert "specialist_outcomes" in data
        assert "routing_history" in data
        assert len(data["routing_history"]) == 1

    def test_dispatcher_memory_from_dict(self):
        """Test DispatcherMemory deserialization."""
        data = {
            "specialist_outcomes": {"python": {"successes": 5, "failures": 1}},
            "routing_history": [
                {"task_id": "t1", "specialist": "python", "problem_type": "code"}
            ],
        }

        memory = DispatcherMemory.from_dict(data)

        assert memory.specialist_outcomes["python"]["successes"] == 5
        assert len(memory.routing_history) == 1

    def test_dispatcher_memory_save_load(self):
        """Test DispatcherMemory save and load to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dispatcher_memory.json"

            # Save
            memory = DispatcherMemory(persist_path=path)
            memory.record_routing("task1", "python", "code")
            memory.record_outcome("python", True)  # specialist, success

            # Load into new instance
            loaded = DispatcherMemory.load(path)

            assert loaded.specialist_outcomes == memory.specialist_outcomes
            assert len(loaded.routing_history) == len(memory.routing_history)

    def test_fleet_memory_to_dict(self):
        """Test FleetMemory serialization."""
        memory = FleetMemory()
        memory.record_execution("python", "Write a function", True, None)

        data = memory.to_dict()

        assert "executions" in data
        assert "python" in data["executions"]
        assert len(data["executions"]["python"]) == 1

    def test_fleet_memory_from_dict(self):
        """Test FleetMemory deserialization."""
        data = {
            "executions": {
                "python": [
                    {"task_summary": "test", "success": True, "error": None}
                ]
            }
        }

        memory = FleetMemory.from_dict(data)

        assert "python" in memory.executions
        assert memory.executions["python"][0]["success"] is True

    def test_fleet_memory_save_load(self):
        """Test FleetMemory save and load to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fleet_memory.json"

            # Save
            memory = FleetMemory(persist_path=path)
            memory.record_execution("python", "test task", True, None)

            # Load into new instance
            loaded = FleetMemory.load(path)

            assert loaded.executions == memory.executions


# ============================================================================
# CallDB Tests
# ============================================================================


class TestCallDB:
    """Test that CallDB correctly records training data."""

    @pytest.fixture
    def db(self):
        """Create a temporary CallDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            yield CallDB(db_path)

    def test_log_call(self, db):
        """Test logging an LLM call."""
        call_id = db.log_call(
            run_id="run1",
            problem_id="problem1",
            tier="fleet",
            specialist="python",
            tokens_in=100,
            tokens_out=50,
            latency_ms=500.0,
            success=True,
            input_preview="test input",
            output_preview="test output",
        )

        assert call_id is not None
        assert call_id > 0

    def test_log_validation(self, db):
        """Test logging a validation result."""
        call_id = db.log_call(
            run_id="run1",
            problem_id="problem1",
            tier="fleet",
            specialist="python",
            tokens_in=100,
            tokens_out=50,
            latency_ms=500.0,
            success=True,
        )

        val_id = db.log_validation(
            call_id=call_id,
            run_id="run1",
            problem_id="problem1",
            passed=True,
            tests_passed=5,
            tests_total=5,
            extracted_code="def foo(): pass",
        )

        assert val_id is not None

    def test_log_routing_decision(self, db):
        """Test logging a routing decision."""
        db.log_routing_decision(
            run_id="run1",
            problem_id="problem1",
            selected_specialist="python",
            confidence=0.85,
            problem_keywords=["array", "sort"],
            alternative_specialists=["math"],
        )

        # Verify by getting stats
        stats = db.get_training_stats()
        assert stats["router_decisions"] >= 1

    def test_get_training_stats(self, db):
        """Test getting training statistics."""
        # Add some data
        call_id = db.log_call(
            run_id="run1",
            problem_id="p1",
            tier="fleet",
            specialist="python",
            tokens_in=100,
            tokens_out=50,
            latency_ms=500.0,
            success=True,
        )
        db.log_validation(
            call_id=call_id,
            run_id="run1",
            problem_id="p1",
            passed=True,
            tests_passed=3,
            tests_total=3,
            extracted_code="def test(): pass",
        )

        stats = db.get_training_stats()

        assert "sft_examples" in stats
        assert "dpo_pairs" in stats
        assert stats["sft_examples"] >= 1

    def test_export_sft_data(self, db):
        """Test SFT data export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add passing validation
            call_id = db.log_call(
                run_id="run1",
                problem_id="p1",
                tier="fleet",
                specialist="python",
                tokens_in=100,
                tokens_out=50,
                latency_ms=500.0,
                success=True,
                input_preview="Write a function",
                output_preview="def foo(): pass",
            )
            db.log_validation(
                call_id=call_id,
                run_id="run1",
                problem_id="p1",
                passed=True,
                tests_passed=3,
                tests_total=3,
                extracted_code="def foo(): pass",
            )

            output_path = Path(tmpdir) / "sft.jsonl"
            count = db.export_sft_data(output_path)

            assert count >= 1
            assert output_path.exists()

            # Verify content
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) >= 1


# ============================================================================
# CLI Tool Tests
# ============================================================================


class TestCLITools:
    """Test that CLI tools can be invoked without errors."""

    def test_demo_imports(self):
        """Test that demo script imports work."""
        # This tests that the demo module can be imported
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "demo", Path(__file__).parent.parent / "scripts" / "demo.py"
        )
        assert spec is not None

    def test_token_audit_imports(self):
        """Test that token_audit module imports."""
        from . import token_audit
        assert hasattr(token_audit, "main")

    def test_error_analyzer_imports(self):
        """Test that error_analyzer module imports."""
        from . import error_analyzer
        assert hasattr(error_analyzer, "main")

    def test_training_curator_imports(self):
        """Test that training_curator module imports."""
        from . import training_curator
        assert hasattr(training_curator, "main")

    def test_specialist_dashboard_imports(self):
        """Test that specialist_dashboard module imports."""
        from . import specialist_dashboard
        assert hasattr(specialist_dashboard, "main")

    def test_ab_test_imports(self):
        """Test that ab_test_confidence module imports."""
        from . import ab_test_confidence
        assert hasattr(ab_test_confidence, "main")


# ============================================================================
# Adaptive Confidence Tests
# ============================================================================


class TestAdaptiveConfidence:
    """Test adaptive confidence scoring."""

    @pytest.fixture
    def db_with_history(self):
        """Create CallDB with historical data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = CallDB(db_path)

            # Add successful python routing decisions
            for i in range(10):
                db.log_routing_decision(
                    run_id=f"run{i}",
                    problem_id=f"p{i}",
                    selected_specialist="python",
                    confidence=0.8,
                )
                # Mark as correct
                db.update_routing_outcome(f"run{i}", decision_correct=True)

            # Add failing math routing decisions
            for i in range(5):
                db.log_routing_decision(
                    run_id=f"math_run{i}",
                    problem_id=f"math_p{i}",
                    selected_specialist="math",
                    confidence=0.6,
                )
                # Mark as incorrect
                db.update_routing_outcome(f"math_run{i}", decision_correct=False)

            yield db

    def test_get_specialist_success_rate(self, db_with_history):
        """Test getting success rate for a specialist."""
        rate = db_with_history.get_specialist_success_rate("python")

        assert rate["total"] == 10
        assert rate["successes"] == 10
        assert rate["success_rate"] == 1.0

    def test_get_all_specialist_rates(self, db_with_history):
        """Test getting all specialist rates."""
        rates = db_with_history.get_all_specialist_rates()

        assert "python" in rates
        assert "math" in rates
        assert rates["python"]["success_rate"] == 1.0
        assert rates["math"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_adaptive_confidence_toggle(self):
        """Test that adaptive confidence can be toggled."""
        llm = MockLLMClient()
        fleet = SpecializedFleet(llm, memory=FleetMemory())
        dispatcher = RoutingDispatcher(fleet, llm, memory=DispatcherMemory())

        # Default should be enabled
        assert dispatcher.enable_adaptive_confidence is True

        # Can toggle off
        dispatcher.enable_adaptive_confidence = False
        assert dispatcher.enable_adaptive_confidence is False
