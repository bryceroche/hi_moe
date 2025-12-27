"""Credit assignment for tier failure attribution (hi_moe-o46).

When tasks fail, identifies which tier caused the failure through
instrumentation and analysis. Enables targeted improvement by
attributing failures to specific tiers.

Approach:
1. Instrument each tier transition with success/failure signals
2. Track cascading failures through the tier stack
3. Analyze patterns to identify root cause tier
4. Generate per-tier failure reports
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tier(Enum):
    """Tier levels in the hierarchy."""
    MONITOR = "monitor"      # Tier 4
    ARCHITECT = "architect"  # Tier 3
    DISPATCHER = "dispatcher"  # Tier 2
    FLEET = "fleet"          # Tier 1


class FailureType(Enum):
    """Types of failures that can occur."""
    PLANNING_ERROR = "planning_error"  # Bad plan from Architect
    ROUTING_ERROR = "routing_error"    # Wrong specialist selection
    EXECUTION_ERROR = "execution_error"  # Code generation failed
    VALIDATION_ERROR = "validation_error"  # Code didn't pass tests
    TIMEOUT = "timeout"                # Exceeded time limit
    RETRY_EXHAUSTED = "retry_exhausted"  # All retries failed
    CONTEXT_ERROR = "context_error"    # Missing/invalid context
    UNKNOWN = "unknown"


@dataclass
class TierEvent:
    """An event at a specific tier during task execution."""
    task_id: str
    tier: Tier
    action: str  # "start", "success", "failure", "retry"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: dict = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0


@dataclass
class FailureAttribution:
    """Attribution of a failure to a specific tier."""
    task_id: str
    root_cause_tier: Tier
    failure_type: FailureType
    confidence: float  # 0-1
    evidence: list[str]
    contributing_tiers: list[Tier] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "root_cause_tier": self.root_cause_tier.value,
            "failure_type": self.failure_type.value,
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "contributing_tiers": [t.value for t in self.contributing_tiers],
        }


@dataclass
class TierStats:
    """Statistics for a single tier."""
    tier: Tier
    total_invocations: int = 0
    successes: int = 0
    failures: int = 0
    retries: int = 0
    total_time_ms: float = 0
    failure_types: dict[str, int] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_invocations == 0:
            return 0.0
        return self.successes / self.total_invocations

    @property
    def avg_time_ms(self) -> float:
        if self.total_invocations == 0:
            return 0.0
        return self.total_time_ms / self.total_invocations

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "invocations": self.total_invocations,
            "success_rate": f"{self.success_rate:.1%}",
            "failures": self.failures,
            "retries": self.retries,
            "avg_time_ms": round(self.avg_time_ms, 1),
            "failure_types": self.failure_types,
        }


class CreditAssigner:
    """Assigns credit/blame to tiers for task outcomes.

    Usage:
        assigner = CreditAssigner()

        # Record events as task executes
        assigner.record_event(task_id, Tier.ARCHITECT, "start")
        assigner.record_event(task_id, Tier.ARCHITECT, "success")
        assigner.record_event(task_id, Tier.DISPATCHER, "start")
        assigner.record_event(task_id, Tier.FLEET, "failure", error="validation failed")

        # Analyze failures
        attribution = assigner.analyze_failure(task_id)

        # Get tier-level stats
        stats = assigner.get_tier_stats()
    """

    def __init__(self):
        self.events: dict[str, list[TierEvent]] = defaultdict(list)
        self.attributions: list[FailureAttribution] = []
        self.tier_stats: dict[Tier, TierStats] = {
            tier: TierStats(tier=tier) for tier in Tier
        }

    def record_event(
        self,
        task_id: str,
        tier: Tier,
        action: str,
        error: str | None = None,
        details: dict | None = None,
        duration_ms: float = 0,
    ) -> None:
        """Record a tier event during task execution."""
        event = TierEvent(
            task_id=task_id,
            tier=tier,
            action=action,
            error=error,
            details=details or {},
            duration_ms=duration_ms,
        )
        self.events[task_id].append(event)

        # Update tier stats
        stats = self.tier_stats[tier]
        if action == "start":
            stats.total_invocations += 1
        elif action == "success":
            stats.successes += 1
            stats.total_time_ms += duration_ms
        elif action == "failure":
            stats.failures += 1
            stats.total_time_ms += duration_ms
            if error:
                failure_type = self._classify_error(error)
                stats.failure_types[failure_type] = stats.failure_types.get(failure_type, 0) + 1
        elif action == "retry":
            stats.retries += 1

        logger.debug(f"[CreditAssigner] {tier.value}:{action} for {task_id}")

    def _classify_error(self, error: str) -> str:
        """Classify error message into failure type."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return FailureType.TIMEOUT.value
        elif "retry" in error_lower or "exhausted" in error_lower:
            return FailureType.RETRY_EXHAUSTED.value
        elif "validation" in error_lower or "test" in error_lower:
            return FailureType.VALIDATION_ERROR.value
        elif "route" in error_lower or "specialist" in error_lower:
            return FailureType.ROUTING_ERROR.value
        elif "plan" in error_lower:
            return FailureType.PLANNING_ERROR.value
        elif "context" in error_lower or "missing" in error_lower:
            return FailureType.CONTEXT_ERROR.value
        elif "execution" in error_lower or "runtime" in error_lower:
            return FailureType.EXECUTION_ERROR.value
        else:
            return FailureType.UNKNOWN.value

    def analyze_failure(self, task_id: str) -> FailureAttribution | None:
        """Analyze events to attribute failure to a tier.

        Returns attribution if task failed, None if it succeeded.
        """
        events = self.events.get(task_id, [])
        if not events:
            return None

        # Check if task ultimately failed
        final_events = [e for e in events if e.action in ("success", "failure")]
        if not final_events:
            return None

        # Find the last event
        last_event = final_events[-1]
        if last_event.action == "success":
            return None

        # Analyze failure chain
        evidence = []
        contributing_tiers = []
        failure_type = FailureType.UNKNOWN
        root_cause_tier = last_event.tier
        confidence = 0.5

        # Look for first failure in the chain
        first_failure = None
        for event in events:
            if event.action == "failure":
                first_failure = event
                break

        if first_failure:
            root_cause_tier = first_failure.tier
            failure_type = FailureType(self._classify_error(first_failure.error or ""))
            evidence.append(f"First failure at {first_failure.tier.value}: {first_failure.error}")
            confidence = 0.7

        # Check for retry patterns
        retries_by_tier = defaultdict(int)
        for event in events:
            if event.action == "retry":
                retries_by_tier[event.tier] += 1

        if retries_by_tier:
            most_retried = max(retries_by_tier, key=retries_by_tier.get)
            evidence.append(f"Most retries at {most_retried.value}: {retries_by_tier[most_retried]}")
            if retries_by_tier[most_retried] >= 2:
                root_cause_tier = most_retried
                confidence = 0.8

        # Identify contributing tiers (any tier that failed before success)
        for event in events:
            if event.action == "failure" and event.tier not in contributing_tiers:
                contributing_tiers.append(event.tier)

        # Specific pattern matching
        if self._detect_planning_failure(events):
            root_cause_tier = Tier.ARCHITECT
            failure_type = FailureType.PLANNING_ERROR
            evidence.append("Pattern: multiple downstream failures after planning")
            confidence = 0.85

        if self._detect_routing_failure(events):
            root_cause_tier = Tier.DISPATCHER
            failure_type = FailureType.ROUTING_ERROR
            evidence.append("Pattern: specialist mismatch or routing issues")
            confidence = 0.8

        if self._detect_execution_failure(events):
            root_cause_tier = Tier.FLEET
            failure_type = FailureType.EXECUTION_ERROR
            evidence.append("Pattern: code execution or validation failure")
            confidence = 0.9

        attribution = FailureAttribution(
            task_id=task_id,
            root_cause_tier=root_cause_tier,
            failure_type=failure_type,
            confidence=confidence,
            evidence=evidence,
            contributing_tiers=contributing_tiers,
        )

        self.attributions.append(attribution)
        logger.info(
            f"[CreditAssigner] Attributed {task_id} failure to {root_cause_tier.value} "
            f"({failure_type.value}, confidence: {confidence:.0%})"
        )

        return attribution

    def _detect_planning_failure(self, events: list[TierEvent]) -> bool:
        """Detect if failure was due to bad planning."""
        architect_success = False
        downstream_failures = 0

        for event in events:
            if event.tier == Tier.ARCHITECT and event.action == "success":
                architect_success = True
            if architect_success and event.action == "failure":
                downstream_failures += 1

        return architect_success and downstream_failures >= 2

    def _detect_routing_failure(self, events: list[TierEvent]) -> bool:
        """Detect if failure was due to routing."""
        for event in events:
            if event.tier == Tier.DISPATCHER and event.action == "failure":
                error = event.error or ""
                if "specialist" in error.lower() or "route" in error.lower():
                    return True
        return False

    def _detect_execution_failure(self, events: list[TierEvent]) -> bool:
        """Detect if failure was due to code execution."""
        for event in events:
            if event.tier == Tier.FLEET and event.action == "failure":
                error = event.error or ""
                if any(kw in error.lower() for kw in ["validation", "test", "runtime", "error"]):
                    return True
        return False

    def get_tier_stats(self) -> dict[str, dict]:
        """Get statistics for each tier."""
        return {tier.value: stats.to_dict() for tier, stats in self.tier_stats.items()}

    def get_attribution_summary(self) -> dict:
        """Get summary of failure attributions."""
        by_tier = defaultdict(int)
        by_type = defaultdict(int)

        for attr in self.attributions:
            by_tier[attr.root_cause_tier.value] += 1
            by_type[attr.failure_type.value] += 1

        return {
            "total_failures": len(self.attributions),
            "by_tier": dict(by_tier),
            "by_type": dict(by_type),
            "avg_confidence": (
                sum(a.confidence for a in self.attributions) / len(self.attributions)
                if self.attributions else 0
            ),
        }

    def get_report(self) -> str:
        """Generate human-readable credit assignment report."""
        lines = [
            "=" * 60,
            "CREDIT ASSIGNMENT REPORT",
            "=" * 60,
        ]

        # Tier stats
        lines.append("\nTier Performance:")
        for tier in Tier:
            stats = self.tier_stats[tier]
            lines.append(
                f"  {tier.value:12} | {stats.success_rate:5.1%} success | "
                f"{stats.failures} failures | {stats.retries} retries"
            )

        # Attribution summary
        summary = self.get_attribution_summary()
        if summary["total_failures"] > 0:
            lines.append(f"\nFailure Attribution ({summary['total_failures']} total):")
            lines.append("  By tier:")
            for tier, count in summary["by_tier"].items():
                pct = count / summary["total_failures"] * 100
                lines.append(f"    {tier:12}: {count} ({pct:.0f}%)")

            lines.append("  By type:")
            for ftype, count in summary["by_type"].items():
                lines.append(f"    {ftype:20}: {count}")

        # Recent attributions
        if self.attributions:
            lines.append("\nRecent Attributions:")
            for attr in self.attributions[-5:]:
                lines.append(
                    f"  [{attr.confidence:.0%}] {attr.task_id}: "
                    f"{attr.root_cause_tier.value} ({attr.failure_type.value})"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save credit assignment data to file."""
        data = {
            "tier_stats": self.get_tier_stats(),
            "summary": self.get_attribution_summary(),
            "attributions": [a.to_dict() for a in self.attributions],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[CreditAssigner] Saved report to {path}")

    def reset(self) -> None:
        """Reset all data."""
        self.events.clear()
        self.attributions.clear()
        self.tier_stats = {tier: TierStats(tier=tier) for tier in Tier}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    assigner = CreditAssigner()

    # Simulate successful task
    assigner.record_event("task-1", Tier.MONITOR, "start")
    assigner.record_event("task-1", Tier.ARCHITECT, "start")
    assigner.record_event("task-1", Tier.ARCHITECT, "success", duration_ms=100)
    assigner.record_event("task-1", Tier.DISPATCHER, "start")
    assigner.record_event("task-1", Tier.FLEET, "start")
    assigner.record_event("task-1", Tier.FLEET, "success", duration_ms=200)
    assigner.record_event("task-1", Tier.DISPATCHER, "success", duration_ms=250)
    assigner.record_event("task-1", Tier.MONITOR, "success", duration_ms=400)

    # Simulate task with Fleet failure
    assigner.record_event("task-2", Tier.MONITOR, "start")
    assigner.record_event("task-2", Tier.ARCHITECT, "start")
    assigner.record_event("task-2", Tier.ARCHITECT, "success", duration_ms=100)
    assigner.record_event("task-2", Tier.DISPATCHER, "start")
    assigner.record_event("task-2", Tier.FLEET, "start")
    assigner.record_event("task-2", Tier.FLEET, "failure", error="Validation failed: test case 2")
    assigner.record_event("task-2", Tier.FLEET, "retry")
    assigner.record_event("task-2", Tier.FLEET, "failure", error="Validation failed: test case 2")
    assigner.record_event("task-2", Tier.DISPATCHER, "failure", error="Fleet exhausted retries")
    assigner.record_event("task-2", Tier.MONITOR, "failure", error="All attempts failed")

    # Simulate task with planning failure
    assigner.record_event("task-3", Tier.MONITOR, "start")
    assigner.record_event("task-3", Tier.ARCHITECT, "start")
    assigner.record_event("task-3", Tier.ARCHITECT, "success", duration_ms=100)
    assigner.record_event("task-3", Tier.DISPATCHER, "start")
    assigner.record_event("task-3", Tier.FLEET, "failure", error="Wrong approach for problem")
    assigner.record_event("task-3", Tier.DISPATCHER, "failure")
    assigner.record_event("task-3", Tier.ARCHITECT, "retry")
    assigner.record_event("task-3", Tier.ARCHITECT, "success")
    assigner.record_event("task-3", Tier.DISPATCHER, "failure", error="Still wrong approach")
    assigner.record_event("task-3", Tier.MONITOR, "failure")

    # Analyze failures
    for task_id in ["task-1", "task-2", "task-3"]:
        attr = assigner.analyze_failure(task_id)
        if attr:
            print(f"\n{task_id}: {attr.root_cause_tier.value} ({attr.failure_type.value})")
            for e in attr.evidence:
                print(f"  - {e}")

    print("\n" + assigner.get_report())
