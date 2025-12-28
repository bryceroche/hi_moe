#!/usr/bin/env python3
"""Insight extraction from run results (hi_moe-3k7).

Identifies valuable patterns from context windows before they're discarded:
- Successful code patterns
- Debugging breakthroughs (error → fix)
- Effective routing decisions
- Problem-solving strategies

Usage:
    from e2e_test.insight_extractor import InsightExtractor
    from e2e_test.call_db import CallDB

    db = CallDB()
    extractor = InsightExtractor(db)

    # Extract insights from a completed run
    insights = extractor.extract_from_run(run_id, run_result)

    # Or manually extract specific insight types
    extractor.extract_successful_pattern(run_id, code, validation_result)
    extractor.extract_debugging_breakthrough(run_id, error, fix_code)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .call_db import CallDB

logger = logging.getLogger(__name__)


# Insight types
INSIGHT_SUCCESSFUL_PATTERN = "successful_pattern"
INSIGHT_DEBUGGING_BREAKTHROUGH = "debugging_breakthrough"
INSIGHT_ROUTING_DECISION = "routing_decision"
INSIGHT_STRATEGY = "strategy"

# Categories
CATEGORY_CODE = "code"
CATEGORY_ALGORITHM = "algorithm"
CATEGORY_OPTIMIZATION = "optimization"
CATEGORY_ERROR_HANDLING = "error_handling"
CATEGORY_DATA_STRUCTURE = "data_structure"
CATEGORY_MATH = "math"


@dataclass
class ExtractedInsight:
    """An insight extracted from run data."""

    insight_type: str
    title: str
    confidence: float
    category: str | None = None
    description: str | None = None
    context: str | None = None
    code_snippet: str | None = None
    error_before: str | None = None
    solution: str | None = None
    tags: list[str] | None = None


class InsightExtractor:
    """Extracts valuable insights from run results for persistence.

    The extractor analyzes completed runs and identifies patterns worth
    preserving before context windows are discarded.
    """

    def __init__(self, db: CallDB):
        """Initialize extractor.

        Args:
            db: CallDB instance for logging insights
        """
        self.db = db

        # Thresholds for insight extraction
        self.min_tests_for_pattern = 1  # Minimum tests passed to consider
        self.min_confidence_for_storage = 0.3  # Minimum confidence to store

    def extract_from_run(
        self,
        run_id: str,
        problem_id: str,
        code: str | None,
        passed: bool,
        tests_passed: int = 0,
        tests_total: int = 0,
        error: str | None = None,
        retry_history: list[dict] | None = None,
        routing_decision: dict | None = None,
    ) -> list[int]:
        """Extract all relevant insights from a completed run.

        Args:
            run_id: Run identifier
            problem_id: Problem identifier
            code: Final generated code
            passed: Whether all tests passed
            tests_passed: Number of tests passed
            tests_total: Total number of tests
            error: Error message if failed
            retry_history: List of retry attempts (for debugging insights)
            routing_decision: Routing decision metadata

        Returns:
            List of insight IDs that were stored
        """
        insight_ids = []

        # Extract successful patterns
        if passed and code and tests_passed >= self.min_tests_for_pattern:
            insight_id = self.extract_successful_pattern(
                run_id=run_id,
                problem_id=problem_id,
                code=code,
                tests_passed=tests_passed,
                tests_total=tests_total,
            )
            if insight_id:
                insight_ids.append(insight_id)

        # Extract debugging breakthroughs from retry history
        if retry_history:
            for retry in retry_history:
                if retry.get("succeeded"):
                    insight_id = self.extract_debugging_breakthrough(
                        run_id=run_id,
                        problem_id=problem_id,
                        error_before=retry.get("error"),
                        fix_code=retry.get("code"),
                        fix_strategy=retry.get("strategy"),
                    )
                    if insight_id:
                        insight_ids.append(insight_id)

        # Extract routing decisions that worked
        if passed and routing_decision:
            insight_id = self.extract_routing_insight(
                run_id=run_id,
                problem_id=problem_id,
                routing_decision=routing_decision,
                success=True,
            )
            if insight_id:
                insight_ids.append(insight_id)

        logger.info(f"[InsightExtractor] Extracted {len(insight_ids)} insights from run {run_id}")
        return insight_ids

    def extract_successful_pattern(
        self,
        run_id: str,
        problem_id: str,
        code: str,
        tests_passed: int,
        tests_total: int,
    ) -> int | None:
        """Extract a successful code pattern.

        Args:
            run_id: Run identifier
            problem_id: Problem identifier
            code: Successful code
            tests_passed: Number of tests passed
            tests_total: Total tests

        Returns:
            Insight ID if stored, None otherwise
        """
        # Calculate confidence based on test coverage
        if tests_total > 0:
            confidence = tests_passed / tests_total
        else:
            confidence = 0.5

        # Boost confidence for perfect scores
        if tests_passed == tests_total and tests_total > 0:
            confidence = min(1.0, confidence + 0.2)

        if confidence < self.min_confidence_for_storage:
            return None

        # Detect category from code patterns
        category = self._detect_code_category(code)

        # Generate title from code structure
        title = self._generate_pattern_title(code, problem_id)

        # Extract key patterns/techniques used
        tags = self._extract_code_tags(code)

        insight_id = self.db.log_insight(
            insight_type=INSIGHT_SUCCESSFUL_PATTERN,
            title=title,
            run_id=run_id,
            problem_id=problem_id,
            category=category,
            confidence=confidence,
            description=f"Successfully solved {problem_id} with {tests_passed}/{tests_total} tests passing",
            code_snippet=code,
            tests_passed=tests_passed,
            tags=tags,
        )

        logger.debug(f"[InsightExtractor] Stored successful pattern: {title} (confidence: {confidence:.2f})")
        return insight_id

    def extract_debugging_breakthrough(
        self,
        run_id: str,
        problem_id: str,
        error_before: str | None,
        fix_code: str | None,
        fix_strategy: str | None = None,
    ) -> int | None:
        """Extract a debugging breakthrough (error → fix).

        Args:
            run_id: Run identifier
            problem_id: Problem identifier
            error_before: Error that was encountered
            fix_code: Code that fixed the error
            fix_strategy: Strategy used to fix

        Returns:
            Insight ID if stored, None otherwise
        """
        if not error_before or not fix_code:
            return None

        # Higher confidence for common error types that were fixed
        confidence = 0.6
        error_type = self._classify_error(error_before)

        if error_type in ["SyntaxError", "NameError", "TypeError"]:
            confidence = 0.7  # Common errors worth learning from
        elif error_type in ["IndexError", "KeyError", "ValueError"]:
            confidence = 0.8  # Boundary/validation errors are valuable
        elif error_type == "WrongAnswer":
            confidence = 0.9  # Logic fixes are highly valuable

        title = f"Fixed {error_type}: {self._summarize_error(error_before)}"

        insight_id = self.db.log_insight(
            insight_type=INSIGHT_DEBUGGING_BREAKTHROUGH,
            title=title[:100],
            run_id=run_id,
            problem_id=problem_id,
            category=CATEGORY_ERROR_HANDLING,
            confidence=confidence,
            description=f"Fixed error using {fix_strategy or 'retry'} strategy",
            error_before=error_before[:2000] if error_before else None,
            solution=fix_code,
            tags=[error_type, fix_strategy or "retry"],
        )

        logger.debug(f"[InsightExtractor] Stored debugging breakthrough: {title[:50]}...")
        return insight_id

    def extract_routing_insight(
        self,
        run_id: str,
        problem_id: str,
        routing_decision: dict,
        success: bool,
    ) -> int | None:
        """Extract a routing decision insight.

        Args:
            run_id: Run identifier
            problem_id: Problem identifier
            routing_decision: Routing metadata
            success: Whether the routing led to success

        Returns:
            Insight ID if stored, None otherwise
        """
        if not success:
            return None  # Only store successful routing decisions

        specialist = routing_decision.get("specialist")
        strategy = routing_decision.get("strategy")
        signals = routing_decision.get("signals", [])

        # Confidence based on how decisive the routing was
        confidence_score = routing_decision.get("confidence", 0.5)
        confidence = min(0.9, confidence_score + 0.2)  # Boost for success

        title = f"Route to {specialist}: {strategy or 'default'}"

        insight_id = self.db.log_insight(
            insight_type=INSIGHT_ROUTING_DECISION,
            title=title,
            run_id=run_id,
            problem_id=problem_id,
            category=None,
            confidence=confidence,
            description=f"Successfully routed to {specialist} specialist",
            context=f"Signals: {', '.join(signals)}" if signals else None,
            tags=[specialist, strategy] if strategy else [specialist],
            metadata=routing_decision,
        )

        logger.debug(f"[InsightExtractor] Stored routing insight: {title}")
        return insight_id

    def _detect_code_category(self, code: str) -> str:
        """Detect the category of code based on patterns.

        Args:
            code: Python code to analyze

        Returns:
            Category string
        """
        code_lower = code.lower()

        # Check for algorithm patterns
        if any(kw in code_lower for kw in ["sort", "binary_search", "merge", "quick"]):
            return CATEGORY_ALGORITHM
        if any(kw in code_lower for kw in ["dp", "memo", "cache", "lru_cache"]):
            return CATEGORY_OPTIMIZATION
        if any(kw in code_lower for kw in ["dict", "set", "deque", "heap", "tree"]):
            return CATEGORY_DATA_STRUCTURE
        if any(kw in code_lower for kw in ["math", "sqrt", "log", "factorial", "prime"]):
            return CATEGORY_MATH

        return CATEGORY_CODE

    def _generate_pattern_title(self, code: str, problem_id: str) -> str:
        """Generate a descriptive title for a code pattern.

        Args:
            code: Python code
            problem_id: Problem identifier

        Returns:
            Title string
        """
        # Try to extract function name
        func_match = re.search(r"def\s+(\w+)\s*\(", code)
        func_name = func_match.group(1) if func_match else "solution"

        # Detect key techniques
        techniques = []
        if "for" in code and "for" in code[code.find("for") + 4:]:
            techniques.append("nested loops")
        if "while" in code:
            techniques.append("iteration")
        if "return" in code and code.count("return") > 1:
            techniques.append("early return")
        if "if" in code and "else" in code:
            techniques.append("conditional")
        if re.search(r"\bdef\s+\w+.*:.*\bdef\s+\w+", code, re.DOTALL):
            techniques.append("helper function")

        if techniques:
            return f"{func_name}: {', '.join(techniques[:2])}"
        return f"{func_name} for {problem_id}"

    def _extract_code_tags(self, code: str) -> list[str]:
        """Extract tags from code for searchability.

        Args:
            code: Python code

        Returns:
            List of tags
        """
        tags = []

        # Detect common patterns
        patterns = {
            "list_comprehension": r"\[.*\bfor\b.*\bin\b.*\]",
            "dictionary_comprehension": r"\{.*:\s*.*\bfor\b.*\bin\b.*\}",
            "lambda": r"\blambda\b",
            "recursion": r"def\s+(\w+).*\1\s*\(",
            "generator": r"\byield\b",
            "context_manager": r"\bwith\b",
            "exception_handling": r"\btry\b.*\bexcept\b",
            "type_hints": r":\s*(int|str|float|list|dict|bool)\b",
        }

        for tag, pattern in patterns.items():
            if re.search(pattern, code, re.DOTALL):
                tags.append(tag)

        # Detect imports
        imports = re.findall(r"(?:from|import)\s+([\w.]+)", code)
        for imp in imports[:3]:  # Limit to first 3
            tags.append(f"uses_{imp.split('.')[0]}")

        return tags[:10]  # Limit total tags

    def _classify_error(self, error: str) -> str:
        """Classify an error message.

        Args:
            error: Error message

        Returns:
            Error type string
        """
        error_types = [
            "SyntaxError", "NameError", "TypeError", "ValueError",
            "IndexError", "KeyError", "AttributeError", "RuntimeError",
            "ZeroDivisionError", "RecursionError",
        ]

        for error_type in error_types:
            if error_type in error:
                return error_type

        if "expected" in error.lower() or "wrong" in error.lower():
            return "WrongAnswer"
        if "timeout" in error.lower():
            return "Timeout"

        return "UnknownError"

    def _summarize_error(self, error: str) -> str:
        """Create a brief summary of an error.

        Args:
            error: Full error message

        Returns:
            Brief summary
        """
        # Get first line of error
        first_line = error.split("\n")[0].strip()

        # Truncate if too long
        if len(first_line) > 80:
            return first_line[:77] + "..."

        return first_line


def extract_insights_from_trajectory(
    db: CallDB,
    trajectory_file: str,
) -> int:
    """Extract insights from a trajectory file.

    Convenience function for batch processing.

    Args:
        db: CallDB instance
        trajectory_file: Path to trajectory JSONL file

    Returns:
        Number of insights extracted
    """
    import json
    from pathlib import Path

    extractor = InsightExtractor(db)
    path = Path(trajectory_file)

    if not path.exists():
        logger.warning(f"Trajectory file not found: {trajectory_file}")
        return 0

    insights_count = 0
    run_id = None
    problem_id = None

    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type")

            if record_type == "run_start":
                run_id = record.get("run_id")
                problem_id = record.get("problem_id")

            elif record_type == "fleet_execution" and run_id:
                if record.get("status") == "success":
                    validation = record.get("validation_result", {})
                    if validation.get("passed"):
                        insight_id = extractor.extract_successful_pattern(
                            run_id=run_id,
                            problem_id=problem_id or "unknown",
                            code=record.get("output_code", ""),
                            tests_passed=validation.get("tests_passed", 0),
                            tests_total=validation.get("tests_total", 0),
                        )
                        if insight_id:
                            insights_count += 1

    logger.info(f"[extract_insights_from_trajectory] Extracted {insights_count} insights from {path.name}")
    return insights_count
