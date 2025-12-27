"""Tests for CodeRunner sandbox execution.

Tests issue hi_moe-6ik: Code execution sandbox for validation.
"""
import pytest

from .code_runner import (
    CodeRunner,
    SandboxConfig,
    TestCase,
    ExecutionStatus,
    TestStatus,
    validate_code,
)


class TestCodeRunner:
    """Tests for CodeRunner class."""

    def test_simple_passing_test(self):
        """Test code that passes a simple test case."""
        runner = CodeRunner()

        code = """
x = int(input())
print(x * 2)
"""
        test_cases = [
            TestCase(id="test-1", input="5", expected_output="10"),
        ]

        result = runner.run(code, test_cases)

        assert result.status == ExecutionStatus.ALL_PASSED
        assert result.passed == 1
        assert result.failed == 0
        assert result.test_results[0].status == TestStatus.PASSED

    def test_wrong_answer(self):
        """Test code that produces wrong output."""
        runner = CodeRunner()

        code = """
x = int(input())
print(x + 1)  # Wrong: should be x * 2
"""
        test_cases = [
            TestCase(id="test-1", input="5", expected_output="10"),
        ]

        result = runner.run(code, test_cases)

        assert result.status == ExecutionStatus.ALL_FAILED
        assert result.passed == 0
        assert result.failed == 1
        assert result.test_results[0].status == TestStatus.WRONG_ANSWER
        assert result.test_results[0].actual_output == "6"

    def test_multiple_test_cases(self):
        """Test running multiple test cases."""
        runner = CodeRunner()

        code = """
x = int(input())
print(x * 2)
"""
        test_cases = [
            TestCase(id="test-1", input="5", expected_output="10"),
            TestCase(id="test-2", input="0", expected_output="0"),
            TestCase(id="test-3", input="-3", expected_output="-6"),
        ]

        result = runner.run(code, test_cases)

        assert result.status == ExecutionStatus.ALL_PASSED
        assert result.passed == 3
        assert result.failed == 0

    def test_partial_success(self):
        """Test code that passes some but not all tests."""
        runner = CodeRunner()

        code = """
x = int(input())
if x > 0:
    print(x * 2)
else:
    print(0)  # Bug: doesn't handle negative correctly
"""
        test_cases = [
            TestCase(id="test-1", input="5", expected_output="10"),
            TestCase(id="test-2", input="0", expected_output="0"),
            TestCase(id="test-3", input="-3", expected_output="-6"),
        ]

        result = runner.run(code, test_cases)

        assert result.status == ExecutionStatus.SOME_PASSED
        assert result.passed == 2
        assert result.failed == 1

    def test_runtime_error(self):
        """Test code that raises an exception."""
        runner = CodeRunner()

        code = """
x = int(input())
y = 1 / x  # Division by zero when x=0
print(y)
"""
        test_cases = [
            TestCase(id="test-1", input="0", expected_output="inf"),
        ]

        result = runner.run(code, test_cases)

        assert result.status == ExecutionStatus.RUNTIME_ERROR
        assert result.test_results[0].status == TestStatus.RUNTIME_ERROR
        assert "ZeroDivision" in result.test_results[0].error_message

    def test_timeout(self):
        """Test code that exceeds timeout."""
        config = SandboxConfig(timeout_ms=100)  # 100ms timeout
        runner = CodeRunner(config)

        code = """
import time
time.sleep(10)  # Sleep for 10 seconds
print("done")
"""
        test_cases = [
            TestCase(id="test-1", input="", expected_output="done"),
        ]

        result = runner.run(code, test_cases)

        assert result.test_results[0].status == TestStatus.TIMEOUT
        assert "Timeout" in result.test_results[0].error_message

    def test_empty_test_cases(self):
        """Test with no test cases."""
        runner = CodeRunner()

        code = "print('hello')"
        result = runner.run(code, [])

        assert result.status == ExecutionStatus.SANDBOX_ERROR
        assert "No test cases" in result.error

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        runner = CodeRunner()

        code = """
x = int(input())
print(x * 2)
"""
        test_cases = [
            TestCase(id="test-1", input="5\n", expected_output="10\n"),
        ]

        result = runner.run(code, test_cases)

        # Should pass - whitespace is stripped for comparison
        assert result.status == ExecutionStatus.ALL_PASSED


class TestTestCase:
    """Tests for TestCase dataclass."""

    def test_from_dict_basic(self):
        """Test creating TestCase from dict."""
        data = {
            "input": "5",
            "expected_output": "10",
        }
        tc = TestCase.from_dict(data, 0)

        assert tc.id == "test-0"
        assert tc.input == "5"
        assert tc.expected_output == "10"

    def test_from_dict_with_expected_key(self):
        """Test creating TestCase with 'expected' key (common format)."""
        data = {
            "input": "5",
            "expected": "10",  # Alternative key
        }
        tc = TestCase.from_dict(data, 1)

        assert tc.expected_output == "10"

    def test_from_dict_with_all_fields(self):
        """Test creating TestCase with all fields."""
        data = {
            "id": "custom-id",
            "input": "5",
            "expected_output": "10",
            "description": "Test doubling",
            "timeout_ms": 1000,
        }
        tc = TestCase.from_dict(data)

        assert tc.id == "custom-id"
        assert tc.description == "Test doubling"
        assert tc.timeout_ms == 1000


class TestCodeRunnerCallable:
    """Tests for CodeRunner as callable (Runner integration)."""

    def test_callable_interface(self):
        """Test using CodeRunner as a function."""
        runner = CodeRunner()

        code = """
x = int(input())
print(x * 2)
"""
        test_cases = [
            {"input": "5", "expected_output": "10"},
            {"input": "3", "expected_output": "6"},
        ]

        result = runner(code, test_cases)

        assert isinstance(result, dict)
        assert result["passed"] is True
        assert result["total_passed"] == 2
        assert result["status"] == "all_passed"

    def test_callable_with_failure(self):
        """Test callable returns correct dict on failure."""
        runner = CodeRunner()

        code = "print('wrong')"
        test_cases = [{"input": "", "expected_output": "right"}]

        result = runner(code, test_cases)

        assert result["passed"] is False
        assert result["total_failed"] == 1


class TestValidateCode:
    """Tests for validate_code convenience function."""

    def test_validate_code_basic(self):
        """Test the convenience function."""
        code = """
x = int(input())
print(x + 1)
"""
        test_cases = [
            {"input": "4", "expected_output": "5"},
        ]

        result = validate_code(code, test_cases)

        assert result["passed"] is True

    def test_validate_code_with_timeout(self):
        """Test convenience function with custom timeout."""
        code = """
import time
time.sleep(1)
print("done")
"""
        test_cases = [{"input": "", "expected_output": "done"}]

        result = validate_code(code, test_cases, timeout_ms=50)

        assert result["passed"] is False
        assert "timeout" in result["test_results"][0]["status"]


class TestFunctionExecution:
    """Tests for executing code that defines and calls functions."""

    def test_function_with_main(self):
        """Test code that defines a function and calls it."""
        runner = CodeRunner()

        code = """
def add(a, b):
    return a + b

x, y = map(int, input().split())
print(add(x, y))
"""
        test_cases = [
            TestCase(id="test-1", input="3 5", expected_output="8"),
            TestCase(id="test-2", input="10 20", expected_output="30"),
        ]

        result = runner.run(code, test_cases)

        assert result.status == ExecutionStatus.ALL_PASSED

    def test_fibonacci(self):
        """Test a fibonacci function."""
        runner = CodeRunner()

        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

n = int(input())
print(fibonacci(n))
"""
        test_cases = [
            TestCase(id="test-1", input="0", expected_output="0"),
            TestCase(id="test-2", input="1", expected_output="1"),
            TestCase(id="test-3", input="5", expected_output="5"),
            TestCase(id="test-4", input="10", expected_output="55"),
        ]

        result = runner.run(code, test_cases)

        assert result.status == ExecutionStatus.ALL_PASSED
        assert result.passed == 4
