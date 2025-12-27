"""Outcome schemas for Fleet→Architect handoff (hi_moe-qwo).

Defines structured result types so the Architect can evaluate
whether the Fleet's work satisfied the original goal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ValidationStatus(Enum):
    """Status of code validation."""
    ALL_PASSED = "all_passed"
    SOME_PASSED = "some_passed"
    ALL_FAILED = "all_failed"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    NOT_VALIDATED = "not_validated"


@dataclass
class ValidationSummary:
    """Structured summary of code validation results.

    Provides enough detail for Architect to decide:
    - Is the solution correct? (passed)
    - What's the failure mode? (status, first_failure)
    - Is re-planning likely to help? (failure_type)
    """
    status: ValidationStatus
    passed: bool  # Convenience: all tests passed
    tests_passed: int = 0
    tests_failed: int = 0
    total_time_ms: float = 0
    first_failure: str | None = None  # First test failure message
    failure_type: str | None = None  # "wrong_answer", "runtime_error", "timeout"

    @classmethod
    def from_validation_dict(cls, validation: dict | None) -> "ValidationSummary":
        """Create from CodeRunner validation result dict."""
        if not validation:
            return cls(
                status=ValidationStatus.NOT_VALIDATED,
                passed=True,  # Assume success if no validation
            )

        passed_count = validation.get("total_passed", 0)
        failed_count = validation.get("total_failed", 0)
        all_passed = validation.get("passed", False)

        # Determine status
        if all_passed:
            status = ValidationStatus.ALL_PASSED
        elif passed_count > 0:
            status = ValidationStatus.SOME_PASSED
        elif validation.get("error"):
            error = validation.get("error", "").lower()
            if "timeout" in error:
                status = ValidationStatus.TIMEOUT
            else:
                status = ValidationStatus.RUNTIME_ERROR
        else:
            status = ValidationStatus.ALL_FAILED

        # Extract first failure info
        first_failure = None
        failure_type = None
        test_results = validation.get("test_results", [])
        for result in test_results:
            if result.get("status") != "passed":
                failure_type = result.get("status")
                if failure_type == "wrong_answer":
                    expected = result.get("expected_output", "")[:50]
                    actual = result.get("actual_output", "")[:50]
                    first_failure = f"Expected: {expected}, Got: {actual}"
                else:
                    first_failure = result.get("error_message", "Unknown error")[:100]
                break

        return cls(
            status=status,
            passed=all_passed,
            tests_passed=passed_count,
            tests_failed=failed_count,
            total_time_ms=validation.get("total_time_ms", 0),
            first_failure=first_failure,
            failure_type=failure_type,
        )

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {
            "status": self.status.value,
            "passed": self.passed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "total_time_ms": self.total_time_ms,
            "first_failure": self.first_failure,
            "failure_type": self.failure_type,
        }


@dataclass
class FleetResult:
    """Structured result from Fleet tier execution.

    Replaces the untyped dict that Fleet was returning.
    """
    code: str | None = None
    raw_response: str | None = None
    validation: ValidationSummary | None = None
    specialist: str | None = None
    adapter: str | None = None

    @property
    def is_valid(self) -> bool:
        """True if code passed validation (or wasn't validated)."""
        if self.validation is None:
            return True
        return self.validation.passed

    @property
    def has_code(self) -> bool:
        """True if code was generated."""
        return bool(self.code and self.code.strip())

    def to_dict(self) -> dict:
        """Serialize for compatibility with existing code."""
        return {
            "code": self.code,
            "raw_response": self.raw_response,
            "validation": self.validation.to_dict() if self.validation else None,
            "specialist": self.specialist,
            "adapter": self.adapter,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FleetResult":
        """Create from dict (for legacy compatibility)."""
        validation = None
        if data.get("validation"):
            validation = ValidationSummary.from_validation_dict(data["validation"])

        return cls(
            code=data.get("code"),
            raw_response=data.get("raw_response"),
            validation=validation,
            specialist=data.get("specialist"),
            adapter=data.get("adapter"),
        )


@dataclass
class StepResult:
    """Result from a single Dispatcher step."""
    step_number: int
    description: str
    specialist: str
    status: str  # "completed" or "failed"
    result: FleetResult | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            "step": self.step_number,
            "description": self.description,
            "specialist": self.specialist,
            "status": self.status,
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
        }


@dataclass
class DispatcherResult:
    """Structured result from Dispatcher tier execution.

    Captures the full execution of a plan with multiple steps.
    """
    plan: list[dict] = field(default_factory=list)  # Original plan steps
    steps_completed: int = 0
    steps_total: int = 0
    final_result: FleetResult | None = None
    step_results: list[StepResult] = field(default_factory=list)
    routing_strategy: str | None = None  # "math_first" or "python_direct"

    @property
    def all_steps_passed(self) -> bool:
        """True if all steps completed successfully."""
        return self.steps_completed == self.steps_total and self.steps_total > 0

    @property
    def code(self) -> str | None:
        """Extract final code from results."""
        if self.final_result:
            return self.final_result.code
        return None

    def to_dict(self) -> dict:
        """Serialize for JSON."""
        return {
            "plan": self.plan,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "final_result": self.final_result.to_dict() if self.final_result else None,
            "step_results": [s.to_dict() for s in self.step_results],
            "routing_strategy": self.routing_strategy,
        }


@dataclass
class OutcomeEvaluation:
    """Architect's evaluation of an outcome.

    Helps Architect decide whether to:
    - Accept the result (goal_achieved=True)
    - Retry with same approach (should_retry=True)
    - Re-plan with different approach (should_replan=True)
    """
    goal_achieved: bool
    confidence: float  # 0-1 how confident in the assessment
    should_retry: bool = False
    should_replan: bool = False
    reason: str = ""

    @classmethod
    def evaluate(
        cls,
        result: FleetResult | DispatcherResult | None,
        had_tests: bool = False,
    ) -> "OutcomeEvaluation":
        """Evaluate whether outcome achieved the goal.

        Args:
            result: The tier result to evaluate
            had_tests: Whether test cases were available
        """
        if result is None:
            return cls(
                goal_achieved=False,
                confidence=1.0,
                should_retry=True,
                reason="No result returned",
            )

        # Handle FleetResult
        if isinstance(result, FleetResult):
            if not result.has_code:
                return cls(
                    goal_achieved=False,
                    confidence=1.0,
                    should_retry=True,
                    reason="No code generated",
                )

            if result.validation:
                if result.validation.passed:
                    return cls(
                        goal_achieved=True,
                        confidence=1.0,
                        reason="All tests passed",
                    )
                elif result.validation.status == ValidationStatus.SOME_PASSED:
                    return cls(
                        goal_achieved=False,
                        confidence=0.8,
                        should_retry=True,
                        reason=f"Partial success: {result.validation.tests_passed}/{result.validation.tests_passed + result.validation.tests_failed} tests passed",
                    )
                elif result.validation.failure_type == "runtime_error":
                    return cls(
                        goal_achieved=False,
                        confidence=0.9,
                        should_retry=True,
                        reason=f"Runtime error: {result.validation.first_failure}",
                    )
                else:
                    return cls(
                        goal_achieved=False,
                        confidence=0.7,
                        should_retry=True,
                        should_replan=result.validation.tests_passed == 0,
                        reason=f"Tests failed: {result.validation.first_failure}",
                    )
            else:
                # No validation - assume success but lower confidence
                return cls(
                    goal_achieved=True,
                    confidence=0.5 if had_tests else 0.7,
                    reason="Code generated (not validated)" if had_tests else "Code generated",
                )

        # Handle DispatcherResult
        if isinstance(result, DispatcherResult):
            if result.all_steps_passed and result.final_result:
                return cls.evaluate(result.final_result, had_tests)
            else:
                failed_step = next(
                    (s for s in result.step_results if s.status == "failed"), None
                )
                return cls(
                    goal_achieved=False,
                    confidence=0.8,
                    should_retry=True,
                    should_replan=result.steps_completed < result.steps_total // 2,
                    reason=f"Step {failed_step.step_number if failed_step else '?'} failed: {failed_step.error if failed_step else 'Unknown'}",
                )

        # Unknown result type
        return cls(
            goal_achieved=False,
            confidence=0.3,
            should_replan=True,
            reason=f"Unknown result type: {type(result)}",
        )
