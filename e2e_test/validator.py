"""Validate generated solutions against test cases."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    passed: bool
    total_cases: int
    passed_cases: int
    failed_cases: list[dict]
    error: str | None = None


def validate_solution(
    code: str,
    test_cases: list[dict],
    function_name: str,
    timeout_seconds: int = 5,
) -> ValidationResult:
    """Run solution against test cases."""

    # Create test harness
    harness = create_test_harness(code, test_cases, function_name)

    # Write to temp file and execute
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(harness)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python3", temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode != 0:
            return ValidationResult(
                passed=False,
                total_cases=len(test_cases),
                passed_cases=0,
                failed_cases=[],
                error=result.stderr,
            )

        # Parse results
        output = json.loads(result.stdout)
        return ValidationResult(
            passed=output["all_passed"],
            total_cases=output["total"],
            passed_cases=output["passed"],
            failed_cases=output["failures"],
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(
            passed=False,
            total_cases=len(test_cases),
            passed_cases=0,
            failed_cases=[],
            error="Timeout: solution took too long",
        )
    except json.JSONDecodeError as e:
        return ValidationResult(
            passed=False,
            total_cases=len(test_cases),
            passed_cases=0,
            failed_cases=[],
            error=f"Failed to parse test output: {e}",
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            total_cases=len(test_cases),
            passed_cases=0,
            failed_cases=[],
            error=str(e),
        )
    finally:
        Path(temp_path).unlink(missing_ok=True)


def create_test_harness(code: str, test_cases: list[dict], function_name: str) -> str:
    """Create a test harness that runs all test cases."""

    # Use repr to get Python-compatible literals (True/False instead of true/false)
    test_cases_repr = repr(test_cases)

    return f'''
import json
import sys

# Solution code
{code}

# Test harness
def run_tests():
    test_cases = {test_cases_repr}
    results = {{"total": len(test_cases), "passed": 0, "failures": []}}

    for i, tc in enumerate(test_cases):
        try:
            # Call the solution function
            actual = {function_name}(**tc["input"])

            # Check result (handle list order for two_sum type problems)
            expected = tc["expected"]
            if isinstance(expected, list) and isinstance(actual, list):
                passed = sorted(actual) == sorted(expected)
            else:
                passed = actual == expected

            if passed:
                results["passed"] += 1
            else:
                results["failures"].append({{
                    "case": i,
                    "input": tc["input"],
                    "expected": expected,
                    "actual": actual,
                }})
        except Exception as e:
            results["failures"].append({{
                "case": i,
                "input": tc["input"],
                "error": str(e),
            }})

    results["all_passed"] = results["passed"] == results["total"]
    print(json.dumps(results))

if __name__ == "__main__":
    run_tests()
'''
