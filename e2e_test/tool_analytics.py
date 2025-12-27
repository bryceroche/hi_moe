"""Tool usage tracking and analytics (hi_moe-5ip).

Tracks which tools/specialists are used vs unused to inform future
tool construction. Identifies gaps where problems reveal missing tools.

Key capabilities:
- Track tool invocations and success rates
- Identify unused tools (candidates for removal)
- Detect patterns suggesting missing tools
- Generate analytics reports
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolInvocation:
    """Record of a single tool invocation."""
    tool_name: str
    task_id: str
    timestamp: str
    success: bool
    duration_ms: float = 0
    input_summary: str = ""
    output_summary: str = ""
    error: str | None = None


@dataclass
class ToolStats:
    """Aggregate statistics for a tool."""
    tool_name: str
    invocations: int = 0
    successes: int = 0
    failures: int = 0
    total_time_ms: float = 0
    last_used: str | None = None

    @property
    def success_rate(self) -> float:
        if self.invocations == 0:
            return 0.0
        return self.successes / self.invocations

    @property
    def avg_time_ms(self) -> float:
        if self.invocations == 0:
            return 0.0
        return self.total_time_ms / self.invocations

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "invocations": self.invocations,
            "success_rate": f"{self.success_rate:.1%}",
            "avg_time_ms": round(self.avg_time_ms, 1),
            "last_used": self.last_used,
        }


@dataclass
class MissingToolSignal:
    """Signal suggesting a missing tool capability."""
    task_id: str
    timestamp: str
    signal_type: str  # "routing_gap", "repeated_failure", "fallback_pattern"
    description: str
    suggested_tool: str | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "signal_type": self.signal_type,
            "description": self.description,
            "suggested_tool": self.suggested_tool,
            "evidence": self.evidence,
        }


# Known tools in the system
KNOWN_TOOLS = {
    # Specialists
    "python": "Python code generation",
    "math": "Mathematical reasoning",
    "algorithms": "Algorithm design",
    "debugging": "Bug identification",
    "refactoring": "Code improvement",
    "general": "General tasks",

    # Infrastructure
    "code_runner": "Code execution and validation",
    "llm_client": "LLM API calls",
    "trajectory_logger": "Execution logging",

    # New tools from this codebase
    "learned_router": "ML-based specialist routing",
    "drift_monitor": "Plan/execution drift detection",
    "credit_assigner": "Failure attribution",
    "architect_tournament": "Prompt variant competition",
}


class ToolAnalytics:
    """Track and analyze tool usage patterns (hi_moe-5ip).

    Usage:
        analytics = ToolAnalytics()

        # Track invocations
        analytics.record_invocation("python", task_id, success=True)

        # Track gaps
        analytics.record_routing_gap(task_id, "No specialist for SQL")

        # Get insights
        report = analytics.get_report()
        unused = analytics.get_unused_tools()
        gaps = analytics.get_missing_tool_signals()
    """

    def __init__(self, known_tools: dict[str, str] | None = None):
        self.known_tools = known_tools or KNOWN_TOOLS
        self.invocations: list[ToolInvocation] = []
        self.stats: dict[str, ToolStats] = {
            name: ToolStats(tool_name=name) for name in self.known_tools
        }
        self.missing_signals: list[MissingToolSignal] = []
        self.routing_gaps: list[dict] = []

    def record_invocation(
        self,
        tool_name: str,
        task_id: str,
        success: bool,
        duration_ms: float = 0,
        input_summary: str = "",
        output_summary: str = "",
        error: str | None = None,
    ) -> None:
        """Record a tool invocation."""
        timestamp = datetime.now().isoformat()

        invocation = ToolInvocation(
            tool_name=tool_name,
            task_id=task_id,
            timestamp=timestamp,
            success=success,
            duration_ms=duration_ms,
            input_summary=input_summary[:200],
            output_summary=output_summary[:200],
            error=error,
        )
        self.invocations.append(invocation)

        # Update stats
        if tool_name not in self.stats:
            self.stats[tool_name] = ToolStats(tool_name=tool_name)

        stats = self.stats[tool_name]
        stats.invocations += 1
        stats.total_time_ms += duration_ms
        stats.last_used = timestamp
        if success:
            stats.successes += 1
        else:
            stats.failures += 1

        logger.debug(f"[ToolAnalytics] {tool_name}: {'✓' if success else '✗'} ({duration_ms:.0f}ms)")

    def record_routing_gap(
        self,
        task_id: str,
        objective: str,
        extracted_domains: list[str] | None = None,
    ) -> None:
        """Record when no specialist matched a task."""
        timestamp = datetime.now().isoformat()

        self.routing_gaps.append({
            "task_id": task_id,
            "objective": objective[:200],
            "domains": extracted_domains or [],
            "timestamp": timestamp,
        })

        # Analyze for missing tool signals
        signal = self._analyze_routing_gap(task_id, objective, extracted_domains)
        if signal:
            self.missing_signals.append(signal)
            logger.info(f"[ToolAnalytics] Missing tool signal: {signal.description}")

    def record_repeated_failure(
        self,
        tool_name: str,
        task_id: str,
        failure_count: int,
        error_pattern: str,
    ) -> None:
        """Record when a tool repeatedly fails on similar tasks."""
        if failure_count >= 3:
            signal = MissingToolSignal(
                task_id=task_id,
                timestamp=datetime.now().isoformat(),
                signal_type="repeated_failure",
                description=f"{tool_name} failed {failure_count}x on similar errors",
                suggested_tool=f"enhanced_{tool_name}",
                evidence=[f"Error pattern: {error_pattern[:100]}"],
            )
            self.missing_signals.append(signal)

    def _analyze_routing_gap(
        self,
        task_id: str,
        objective: str,
        domains: list[str] | None,
    ) -> MissingToolSignal | None:
        """Analyze a routing gap for missing tool signals."""
        objective_lower = objective.lower()

        # Domain keywords that suggest missing specialists
        domain_hints = {
            "sql": ("sql", "query", "database", "select", "join"),
            "cuda": ("cuda", "gpu", "kernel", "parallel"),
            "web": ("html", "css", "javascript", "react", "frontend"),
            "devops": ("docker", "kubernetes", "deploy", "ci/cd"),
            "testing": ("test", "unittest", "pytest", "coverage"),
        }

        for domain, keywords in domain_hints.items():
            if any(kw in objective_lower for kw in keywords):
                if domain not in self.known_tools:
                    return MissingToolSignal(
                        task_id=task_id,
                        timestamp=datetime.now().isoformat(),
                        signal_type="routing_gap",
                        description=f"Task mentions {domain} but no specialist exists",
                        suggested_tool=f"{domain}_specialist",
                        evidence=[f"Keywords matched: {[k for k in keywords if k in objective_lower]}"],
                    )

        return None

    def get_unused_tools(self, min_invocations: int = 0) -> list[str]:
        """Get tools that haven't been used or are underused."""
        unused = []
        for name, stats in self.stats.items():
            if stats.invocations <= min_invocations:
                unused.append(name)
        return unused

    def get_hot_tools(self, top_n: int = 5) -> list[tuple[str, ToolStats]]:
        """Get most frequently used tools."""
        ranked = sorted(
            self.stats.items(),
            key=lambda x: x[1].invocations,
            reverse=True,
        )
        return ranked[:top_n]

    def get_failing_tools(self, min_failure_rate: float = 0.2) -> list[tuple[str, ToolStats]]:
        """Get tools with high failure rates."""
        failing = []
        for name, stats in self.stats.items():
            if stats.invocations >= 3 and (1 - stats.success_rate) >= min_failure_rate:
                failing.append((name, stats))
        return sorted(failing, key=lambda x: x[1].success_rate)

    def get_missing_tool_signals(self) -> list[MissingToolSignal]:
        """Get signals suggesting missing tool capabilities."""
        return self.missing_signals

    def get_report(self) -> str:
        """Generate human-readable analytics report."""
        lines = [
            "=" * 60,
            "TOOL USAGE ANALYTICS",
            "=" * 60,
        ]

        # Summary
        total_invocations = sum(s.invocations for s in self.stats.values())
        active_tools = sum(1 for s in self.stats.values() if s.invocations > 0)
        lines.append(f"\nTotal invocations: {total_invocations}")
        lines.append(f"Active tools: {active_tools}/{len(self.stats)}")

        # Hot tools
        lines.append("\nMost Used Tools:")
        for name, stats in self.get_hot_tools():
            if stats.invocations > 0:
                lines.append(
                    f"  {name:20} | {stats.invocations:4} calls | "
                    f"{stats.success_rate:5.1%} success | {stats.avg_time_ms:.0f}ms avg"
                )

        # Unused tools
        unused = self.get_unused_tools()
        if unused:
            lines.append(f"\nUnused Tools ({len(unused)}):")
            for name in unused[:10]:
                desc = self.known_tools.get(name, "")
                lines.append(f"  - {name}: {desc}")

        # Failing tools
        failing = self.get_failing_tools()
        if failing:
            lines.append("\nHigh Failure Rate Tools:")
            for name, stats in failing:
                lines.append(
                    f"  {name}: {stats.success_rate:.1%} success "
                    f"({stats.failures} failures)"
                )

        # Missing tool signals
        if self.missing_signals:
            lines.append(f"\nMissing Tool Signals ({len(self.missing_signals)}):")
            for signal in self.missing_signals[-5:]:
                lines.append(f"  [{signal.signal_type}] {signal.description}")
                if signal.suggested_tool:
                    lines.append(f"    → Suggested: {signal.suggested_tool}")

        # Routing gaps
        if self.routing_gaps:
            lines.append(f"\nRouting Gaps ({len(self.routing_gaps)}):")
            for gap in self.routing_gaps[-3:]:
                lines.append(f"  - {gap['objective'][:50]}...")

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_summary(self) -> dict:
        """Get summary statistics as dict."""
        return {
            "total_invocations": sum(s.invocations for s in self.stats.values()),
            "active_tools": sum(1 for s in self.stats.values() if s.invocations > 0),
            "total_tools": len(self.stats),
            "unused_tools": len(self.get_unused_tools()),
            "failing_tools": len(self.get_failing_tools()),
            "missing_signals": len(self.missing_signals),
            "routing_gaps": len(self.routing_gaps),
        }

    def save(self, path: Path) -> None:
        """Save analytics data to file."""
        data = {
            "summary": self.get_summary(),
            "tool_stats": {name: stats.to_dict() for name, stats in self.stats.items()},
            "missing_signals": [s.to_dict() for s in self.missing_signals],
            "routing_gaps": self.routing_gaps[-100:],  # Last 100
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[ToolAnalytics] Saved to {path}")

    def reset(self) -> None:
        """Reset all analytics data."""
        self.invocations.clear()
        self.stats = {name: ToolStats(tool_name=name) for name in self.known_tools}
        self.missing_signals.clear()
        self.routing_gaps.clear()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analytics = ToolAnalytics()

    # Simulate tool usage
    tools_used = [
        ("python", True, 150),
        ("python", True, 180),
        ("python", False, 200),
        ("math", True, 300),
        ("math", True, 250),
        ("algorithms", True, 400),
        ("llm_client", True, 50),
        ("llm_client", True, 45),
        ("llm_client", True, 55),
        ("code_runner", True, 100),
        ("code_runner", False, 150),
    ]

    for i, (tool, success, duration) in enumerate(tools_used):
        analytics.record_invocation(
            tool_name=tool,
            task_id=f"task-{i}",
            success=success,
            duration_ms=duration,
        )

    # Simulate routing gaps
    analytics.record_routing_gap(
        task_id="task-20",
        objective="Write a SQL query to join users and orders tables",
        extracted_domains=[],
    )

    analytics.record_routing_gap(
        task_id="task-21",
        objective="Create a Docker container for the application",
        extracted_domains=[],
    )

    print(analytics.get_report())

    # Show programmatic access
    print("\n--- Programmatic Access ---")
    print(f"Unused tools: {analytics.get_unused_tools()}")
    print(f"Missing signals: {[s.suggested_tool for s in analytics.get_missing_tool_signals()]}")
