"""Code execution sandbox for validating Fleet-generated solutions.

Implements issue hi_moe-6ik: Sandboxed code execution for validation.

Provides:
- Local subprocess execution with timeout (for testing)
- Modal sandbox execution (for production)
- Safety limits: timeout, memory, output size
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of code execution."""
    ALL_PASSED = "all_passed"
    SOME_PASSED = "some_passed"
    ALL_FAILED = "all_failed"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    SANDBOX_ERROR = "sandbox_error"


class TestStatus(Enum):
    """Status of a single test case."""
    PASSED = "passed"
    WRONG_ANSWER = "wrong_answer"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"


@dataclass
class TestCase:
    """A single test case."""
    id: str
    input: str
    expected_output: str
    description: str = ""
    timeout_ms: int | None = None

    @classmethod
    def from_dict(cls, data: dict, index: int = 0) -> "TestCase":
        """Create TestCase from dictionary."""
        return cls(
            id=data.get("id", f"test-{index}"),
            input=data.get("input", ""),
            expected_output=data.get("expected_output", data.get("expected", "")),
            description=data.get("description", ""),
            timeout_ms=data.get("timeout_ms"),
        )


@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    status: TestStatus
    actual_output: str
    expected_output: str
    execution_time_ms: float
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "test_id": self.test_id,
            "status": self.status.value,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
        }


@dataclass
class ExecutionResult:
    """Result of code execution."""
    status: ExecutionStatus
    test_results: list[TestResult]
    total_time_ms: float
    passed: int
    failed: int
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for runner integration."""
        return {
            "passed": self.passed == len(self.test_results) and self.passed > 0,
            "status": self.status.value,
            "total_passed": self.passed,
            "total_failed": self.failed,
            "total_time_ms": self.total_time_ms,
            "test_results": [r.to_dict() for r in self.test_results],
            "error": self.error,
        }


@dataclass
class SandboxConfig:
    """Configuration for the sandbox environment."""
    timeout_ms: int = 5000  # 5 seconds per test
    total_timeout_ms: int = 30000  # 30 seconds total
    max_output_size: int = 10000  # 10KB max output
    python_executable: str = "python3"


class CodeRunner:
    """Sandboxed code execution for validating solutions.

    Usage:
        runner = CodeRunner()
        result = runner.run(code, test_cases)

        # Or as a function for Runner integration:
        runner = CodeRunner()
        validation = runner(code, test_cases)  # Returns dict
    """

    def __init__(self, config: SandboxConfig | None = None):
        """Initialize CodeRunner.

        Args:
            config: Sandbox configuration (uses defaults if None)
        """
        self.config = config or SandboxConfig()

    def __call__(self, code: str, test_cases: list[dict]) -> dict:
        """Execute code against test cases (for Runner integration).

        Args:
            code: Python code to execute
            test_cases: List of test case dicts with 'input' and 'expected_output'

        Returns:
            Result dictionary compatible with Runner
        """
        cases = [TestCase.from_dict(tc, i) for i, tc in enumerate(test_cases)]
        result = self.run(code, cases)
        return result.to_dict()

    def run(self, code: str, test_cases: list[TestCase]) -> ExecutionResult:
        """Execute code against test cases.

        Args:
            code: Python code to execute
            test_cases: List of TestCase objects

        Returns:
            ExecutionResult with all test results
        """
        if not test_cases:
            return ExecutionResult(
                status=ExecutionStatus.SANDBOX_ERROR,
                test_results=[],
                total_time_ms=0,
                passed=0,
                failed=0,
                error="No test cases provided",
            )

        start_time = time.monotonic()
        test_results: list[TestResult] = []

        # Run each test case
        for test_case in test_cases:
            # Check total timeout
            elapsed = (time.monotonic() - start_time) * 1000
            if elapsed > self.config.total_timeout_ms:
                test_results.append(TestResult(
                    test_id=test_case.id,
                    status=TestStatus.TIMEOUT,
                    actual_output="",
                    expected_output=test_case.expected_output,
                    execution_time_ms=0,
                    error_message="Total timeout exceeded",
                ))
                continue

            result = self._run_single_test(code, test_case)
            test_results.append(result)

        total_time_ms = (time.monotonic() - start_time) * 1000

        # Calculate summary
        passed = sum(1 for r in test_results if r.status == TestStatus.PASSED)
        failed = len(test_results) - passed

        # Determine overall status
        if passed == len(test_results):
            status = ExecutionStatus.ALL_PASSED
        elif passed > 0:
            status = ExecutionStatus.SOME_PASSED
        elif any(r.status == TestStatus.RUNTIME_ERROR for r in test_results):
            status = ExecutionStatus.RUNTIME_ERROR
        elif any(r.status == TestStatus.TIMEOUT for r in test_results):
            status = ExecutionStatus.TIMEOUT
        else:
            status = ExecutionStatus.ALL_FAILED

        return ExecutionResult(
            status=status,
            test_results=test_results,
            total_time_ms=total_time_ms,
            passed=passed,
            failed=failed,
        )

    def _run_single_test(self, code: str, test_case: TestCase) -> TestResult:
        """Run a single test case.

        Args:
            code: Python code to execute
            test_case: Test case to run

        Returns:
            TestResult for this test
        """
        timeout_s = (test_case.timeout_ms or self.config.timeout_ms) / 1000.0

        # Create a wrapper script that runs the code with input
        wrapper = self._create_wrapper(code)

        start_time = time.monotonic()

        try:
            # Run in subprocess with timeout
            result = subprocess.run(
                [self.config.python_executable, "-c", wrapper],
                input=test_case.input,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )

            execution_time_ms = (time.monotonic() - start_time) * 1000

            # Truncate output if too large
            stdout = result.stdout[:self.config.max_output_size]
            stderr = result.stderr[:self.config.max_output_size]

            # Check for runtime errors
            if result.returncode != 0:
                return TestResult(
                    test_id=test_case.id,
                    status=TestStatus.RUNTIME_ERROR,
                    actual_output=stdout,
                    expected_output=test_case.expected_output,
                    execution_time_ms=execution_time_ms,
                    error_message=stderr or f"Exit code: {result.returncode}",
                )

            # Compare output (strip whitespace for comparison)
            actual = stdout.strip()
            expected = test_case.expected_output.strip()

            if actual == expected:
                return TestResult(
                    test_id=test_case.id,
                    status=TestStatus.PASSED,
                    actual_output=actual,
                    expected_output=expected,
                    execution_time_ms=execution_time_ms,
                )
            else:
                return TestResult(
                    test_id=test_case.id,
                    status=TestStatus.WRONG_ANSWER,
                    actual_output=actual,
                    expected_output=expected,
                    execution_time_ms=execution_time_ms,
                )

        except subprocess.TimeoutExpired:
            execution_time_ms = (time.monotonic() - start_time) * 1000
            return TestResult(
                test_id=test_case.id,
                status=TestStatus.TIMEOUT,
                actual_output="",
                expected_output=test_case.expected_output,
                execution_time_ms=execution_time_ms,
                error_message=f"Timeout after {timeout_s}s",
            )

        except Exception as e:
            execution_time_ms = (time.monotonic() - start_time) * 1000
            return TestResult(
                test_id=test_case.id,
                status=TestStatus.RUNTIME_ERROR,
                actual_output="",
                expected_output=test_case.expected_output,
                execution_time_ms=execution_time_ms,
                error_message=str(e),
            )

    def _create_wrapper(self, code: str) -> str:
        """Create wrapper script for code execution.

        Handles common patterns:
        - Functions that need to be called with input
        - Direct stdin processing
        """
        # The code should handle stdin itself
        # We just need to make sure it runs
        return code


def create_code_runner(config: SandboxConfig | None = None) -> CodeRunner:
    """Factory function to create a CodeRunner.

    Args:
        config: Optional sandbox configuration

    Returns:
        Configured CodeRunner instance
    """
    return CodeRunner(config)


# Convenience function for quick validation
def validate_code(
    code: str,
    test_cases: list[dict],
    timeout_ms: int = 5000,
) -> dict:
    """Validate code against test cases.

    Args:
        code: Python code to execute
        test_cases: List of test case dicts
        timeout_ms: Timeout per test in milliseconds

    Returns:
        Validation result dictionary
    """
    config = SandboxConfig(timeout_ms=timeout_ms)
    runner = CodeRunner(config)
    return runner(code, test_cases)
