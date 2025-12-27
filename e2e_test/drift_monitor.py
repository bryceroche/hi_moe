"""State drift monitoring for tier coordination (hi_moe-z68).

Tracks divergence between what Architect plans/expects and what
actually happens during execution. Detects when the system's
internal model drifts from reality.

Key metrics:
- Plan adherence: Did execution follow the plan?
- Outcome prediction: Did results match expectations?
- Context preservation: Was information lost in handoffs?
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of state drift that can occur."""
    PLAN_DEVIATION = "plan_deviation"  # Execution didn't follow plan
    OUTCOME_MISMATCH = "outcome_mismatch"  # Result differs from expectation
    CONTEXT_LOSS = "context_loss"  # Information lost in handoff
    SPECIALIST_OVERRIDE = "specialist_override"  # Different specialist used
    RETRY_ESCALATION = "retry_escalation"  # More retries than expected


@dataclass
class PlanRecord:
    """Record of what was planned."""
    task_id: str
    planned_steps: list[dict]
    expected_specialist: str | None = None
    expected_outcome: str | None = None
    confidence: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExecutionRecord:
    """Record of what actually happened."""
    task_id: str
    actual_steps: list[dict]
    actual_specialist: str
    actual_outcome: str
    success: bool
    execution_time_ms: float = 0
    retries: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DriftEvent:
    """A detected drift between plan and execution."""
    task_id: str
    drift_type: DriftType
    severity: float  # 0-1, higher is worse
    planned: Any
    actual: Any
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "drift_type": self.drift_type.value,
            "severity": self.severity,
            "planned": self.planned,
            "actual": self.actual,
            "description": self.description,
            "timestamp": self.timestamp,
        }


@dataclass
class DriftStats:
    """Aggregate drift statistics."""
    total_tasks: int = 0
    tasks_with_drift: int = 0
    drift_events: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    avg_severity: float = 0.0

    @property
    def drift_rate(self) -> float:
        """Fraction of tasks that experienced drift."""
        return self.tasks_with_drift / self.total_tasks if self.total_tasks > 0 else 0

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "tasks_with_drift": self.tasks_with_drift,
            "drift_rate": f"{self.drift_rate:.1%}",
            "drift_events": self.drift_events,
            "by_type": self.by_type,
            "avg_severity": round(self.avg_severity, 3),
        }


class DriftMonitor:
    """Monitors state drift between Architect plans and actual execution.

    Usage:
        monitor = DriftMonitor()

        # Record what was planned
        monitor.record_plan(task_id, planned_steps, expected_specialist)

        # Record what happened
        monitor.record_execution(task_id, actual_steps, actual_specialist, outcome)

        # Check for drift
        drifts = monitor.detect_drift(task_id)

        # Get overall stats
        stats = monitor.get_stats()
    """

    def __init__(self, severity_thresholds: dict[DriftType, float] | None = None):
        """Initialize drift monitor.

        Args:
            severity_thresholds: Custom severity thresholds by drift type
        """
        self.plans: dict[str, PlanRecord] = {}
        self.executions: dict[str, ExecutionRecord] = {}
        self.drift_events: list[DriftEvent] = []

        # Default severity weights
        self.severity_thresholds = severity_thresholds or {
            DriftType.PLAN_DEVIATION: 0.3,
            DriftType.OUTCOME_MISMATCH: 0.5,
            DriftType.CONTEXT_LOSS: 0.4,
            DriftType.SPECIALIST_OVERRIDE: 0.2,
            DriftType.RETRY_ESCALATION: 0.3,
        }

    def record_plan(
        self,
        task_id: str,
        planned_steps: list[dict],
        expected_specialist: str | None = None,
        expected_outcome: str | None = None,
        confidence: float = 0.5,
    ) -> None:
        """Record what Architect planned for a task."""
        self.plans[task_id] = PlanRecord(
            task_id=task_id,
            planned_steps=planned_steps,
            expected_specialist=expected_specialist,
            expected_outcome=expected_outcome,
            confidence=confidence,
        )
        logger.debug(f"[DriftMonitor] Recorded plan for {task_id}: {len(planned_steps)} steps")

    def record_execution(
        self,
        task_id: str,
        actual_steps: list[dict],
        actual_specialist: str,
        actual_outcome: str,
        success: bool,
        execution_time_ms: float = 0,
        retries: int = 0,
    ) -> None:
        """Record what actually happened during execution."""
        self.executions[task_id] = ExecutionRecord(
            task_id=task_id,
            actual_steps=actual_steps,
            actual_specialist=actual_specialist,
            actual_outcome=actual_outcome,
            success=success,
            execution_time_ms=execution_time_ms,
            retries=retries,
        )
        logger.debug(f"[DriftMonitor] Recorded execution for {task_id}: {actual_outcome}")

    def detect_drift(self, task_id: str) -> list[DriftEvent]:
        """Detect drift between plan and execution for a task.

        Returns list of drift events detected.
        """
        if task_id not in self.plans or task_id not in self.executions:
            return []

        plan = self.plans[task_id]
        execution = self.executions[task_id]
        drifts = []

        # Check specialist override
        if plan.expected_specialist and plan.expected_specialist != execution.actual_specialist:
            drift = DriftEvent(
                task_id=task_id,
                drift_type=DriftType.SPECIALIST_OVERRIDE,
                severity=self._calculate_severity(DriftType.SPECIALIST_OVERRIDE, plan, execution),
                planned=plan.expected_specialist,
                actual=execution.actual_specialist,
                description=f"Expected {plan.expected_specialist}, used {execution.actual_specialist}",
            )
            drifts.append(drift)

        # Check plan deviation (step count mismatch)
        if len(plan.planned_steps) != len(execution.actual_steps):
            drift = DriftEvent(
                task_id=task_id,
                drift_type=DriftType.PLAN_DEVIATION,
                severity=self._calculate_severity(DriftType.PLAN_DEVIATION, plan, execution),
                planned=len(plan.planned_steps),
                actual=len(execution.actual_steps),
                description=f"Planned {len(plan.planned_steps)} steps, executed {len(execution.actual_steps)}",
            )
            drifts.append(drift)

        # Check outcome mismatch
        if plan.expected_outcome:
            outcome_match = self._compare_outcomes(plan.expected_outcome, execution.actual_outcome)
            if outcome_match < 0.5:
                drift = DriftEvent(
                    task_id=task_id,
                    drift_type=DriftType.OUTCOME_MISMATCH,
                    severity=self._calculate_severity(DriftType.OUTCOME_MISMATCH, plan, execution),
                    planned=plan.expected_outcome,
                    actual=execution.actual_outcome,
                    description=f"Outcome diverged from expectation (match: {outcome_match:.0%})",
                )
                drifts.append(drift)

        # Check retry escalation
        if execution.retries > 0:
            drift = DriftEvent(
                task_id=task_id,
                drift_type=DriftType.RETRY_ESCALATION,
                severity=min(execution.retries * 0.2, 1.0),
                planned=0,
                actual=execution.retries,
                description=f"Required {execution.retries} retries",
            )
            drifts.append(drift)

        # Check for context loss in step transitions
        context_loss = self._detect_context_loss(plan.planned_steps, execution.actual_steps)
        if context_loss:
            drift = DriftEvent(
                task_id=task_id,
                drift_type=DriftType.CONTEXT_LOSS,
                severity=self._calculate_severity(DriftType.CONTEXT_LOSS, plan, execution),
                planned="full context",
                actual=context_loss,
                description=f"Context potentially lost: {context_loss}",
            )
            drifts.append(drift)

        # Store drift events
        self.drift_events.extend(drifts)

        if drifts:
            logger.warning(f"[DriftMonitor] Detected {len(drifts)} drift events for {task_id}")

        return drifts

    def _calculate_severity(
        self,
        drift_type: DriftType,
        plan: PlanRecord,
        execution: ExecutionRecord,
    ) -> float:
        """Calculate severity of a drift event."""
        base = self.severity_thresholds.get(drift_type, 0.5)

        # Adjust by plan confidence (high confidence plan deviation is worse)
        confidence_factor = plan.confidence

        # Adjust by execution success (drift that caused failure is worse)
        success_factor = 0.5 if execution.success else 1.0

        return min(base * (0.5 + confidence_factor * 0.5) * success_factor, 1.0)

    def _compare_outcomes(self, expected: str, actual: str) -> float:
        """Compare expected vs actual outcome (simple word overlap)."""
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())

        if not expected_words or not actual_words:
            return 0.5

        overlap = len(expected_words & actual_words)
        return overlap / max(len(expected_words), len(actual_words))

    def _detect_context_loss(
        self,
        planned_steps: list[dict],
        actual_steps: list[dict],
    ) -> str | None:
        """Detect if context was lost between steps."""
        # Check if actual steps reference context from plan
        for i, actual in enumerate(actual_steps):
            actual_desc = str(actual.get("description", "")).lower()

            # Look for indicators of missing context
            missing_indicators = ["unknown", "unclear", "missing", "no context", "?"]
            if any(ind in actual_desc for ind in missing_indicators):
                return f"Step {i+1} may have missing context"

        return None

    def get_stats(self) -> DriftStats:
        """Get aggregate drift statistics."""
        if not self.executions:
            return DriftStats()

        tasks_with_drift = set()
        by_type: dict[str, int] = {}
        total_severity = 0.0

        for event in self.drift_events:
            tasks_with_drift.add(event.task_id)
            type_name = event.drift_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1
            total_severity += event.severity

        n_events = len(self.drift_events)
        return DriftStats(
            total_tasks=len(self.executions),
            tasks_with_drift=len(tasks_with_drift),
            drift_events=n_events,
            by_type=by_type,
            avg_severity=total_severity / n_events if n_events > 0 else 0,
        )

    def get_drift_report(self) -> str:
        """Generate human-readable drift report."""
        stats = self.get_stats()
        lines = [
            "=" * 50,
            "DRIFT MONITORING REPORT",
            "=" * 50,
            f"Total tasks monitored: {stats.total_tasks}",
            f"Tasks with drift: {stats.tasks_with_drift} ({stats.drift_rate:.1%})",
            f"Total drift events: {stats.drift_events}",
            f"Average severity: {stats.avg_severity:.2f}",
            "",
            "Drift by type:",
        ]

        for drift_type, count in stats.by_type.items():
            lines.append(f"  - {drift_type}: {count}")

        if self.drift_events:
            lines.extend(["", "Recent drift events:"])
            for event in self.drift_events[-5:]:
                lines.append(f"  [{event.severity:.2f}] {event.task_id}: {event.description}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save drift data to file."""
        data = {
            "stats": self.get_stats().to_dict(),
            "events": [e.to_dict() for e in self.drift_events],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[DriftMonitor] Saved report to {path}")

    def reset(self) -> None:
        """Reset all monitoring data."""
        self.plans.clear()
        self.executions.clear()
        self.drift_events.clear()


# Integration with tier system
class TierDriftMonitor(DriftMonitor):
    """Drift monitor integrated with tier trajectory logging."""

    def record_architect_plan(
        self,
        task_id: str,
        plan: str,
        delegation: dict,
    ) -> None:
        """Record Architect's plan from trajectory."""
        # Parse plan into steps
        steps = []
        for line in plan.split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                steps.append({"description": line})

        expected_specialist = None
        if "python" in plan.lower():
            expected_specialist = "python"
        elif "math" in plan.lower():
            expected_specialist = "math"

        self.record_plan(
            task_id=task_id,
            planned_steps=steps,
            expected_specialist=expected_specialist,
            expected_outcome="completed" if steps else None,
        )

    def record_dispatcher_execution(
        self,
        task_id: str,
        routing_decision: str,
        specialist: str,
        outcome_status: str,
        plan_steps: list[dict] | None = None,
    ) -> None:
        """Record Dispatcher's execution from trajectory."""
        self.record_execution(
            task_id=task_id,
            actual_steps=plan_steps or [],
            actual_specialist=specialist,
            actual_outcome=outcome_status,
            success=outcome_status == "completed",
        )

        # Auto-detect drift
        self.detect_drift(task_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo
    monitor = DriftMonitor()

    # Simulate a task with drift
    monitor.record_plan(
        task_id="task-1",
        planned_steps=[
            {"description": "Analyze the algorithm"},
            {"description": "Implement the solution"},
        ],
        expected_specialist="math",
        confidence=0.8,
    )

    monitor.record_execution(
        task_id="task-1",
        actual_steps=[
            {"description": "Implement solution directly"},
        ],
        actual_specialist="python",
        actual_outcome="completed",
        success=True,
        retries=1,
    )

    drifts = monitor.detect_drift("task-1")
    print(f"Detected {len(drifts)} drift events:")
    for d in drifts:
        print(f"  - {d.drift_type.value}: {d.description} (severity: {d.severity:.2f})")

    # Simulate a task without drift
    monitor.record_plan(
        task_id="task-2",
        planned_steps=[{"description": "Write Python code"}],
        expected_specialist="python",
        confidence=0.9,
    )

    monitor.record_execution(
        task_id="task-2",
        actual_steps=[{"description": "Write Python code"}],
        actual_specialist="python",
        actual_outcome="completed",
        success=True,
    )

    monitor.detect_drift("task-2")

    print("\n" + monitor.get_drift_report())
