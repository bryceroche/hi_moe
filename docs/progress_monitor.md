# Progress Monitor Specification

> Fourth tier - meta-layer that tracks whether the system is making progress, not just planning/executing.

## Related Specifications

- **[Abstract Architect](./abstract_architect.md)**: Receives tasks from and reports outcomes to Progress Monitor
- **[Handoff Protocol](./handoff_protocol.md)**: Message formats for tier communication
- **[BeadsClient](./beads_client.md)**: Persistent state storage for progress, confidence, valence
- **[Routing Dispatcher](./routing_dispatcher.md)**: Receives routing feedback for training signal

## Required Imports

```python
from __future__ import annotations
import asyncio
import hashlib
import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from abstract_architect import AbstractArchitect, ArchitectTask, ArchitectResult
from beads_client import BeadsClient

logger = logging.getLogger(__name__)
```

## Overview

The Progress Monitor sits above the Abstract Architect as the fourth and highest tier. Its job isn't planning or execution—it's tracking whether the system is *getting somewhere*.

```
                    ┌─────────────────────────────────────────┐
                    │           PROGRESS MONITOR              │
                    │                                         │
                    │  "Are we making progress, or spinning?" │
                    │                                         │
                    │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
                    │  │Progress │  │Surprise │  │ Valence │ │
                    │  │ Tracker │  │Detector │  │ Manager │ │
                    │  └─────────┘  └─────────┘  └─────────┘ │
                    │                                         │
                    │  ┌─────────┐  ┌─────────┐              │
                    │  │ Memory  │  │ Routine │              │
                    │  │ Manager │  │Extractor│              │
                    │  └─────────┘  └─────────┘              │
                    └─────────────────────────────────────────┘
                                      │
                           Delegates tasks, receives outcomes
                                      ▼
                    ┌─────────────────────────────────────────┐
                    │          ABSTRACT ARCHITECT             │
                    │       (Strategic planning tier)         │
                    └─────────────────────────────────────────┘
                                      │
                                      ▼
                              Lower tiers...
```

## Core Responsibilities

1. **Progress Tracking**: Is the system moving toward objectives, or stuck in loops?
2. **Surprise Detection**: Flag outcomes outside confidence intervals for analysis
3. **Valence Signals**: Attach good/bad scalar to outcomes as learning gradient
4. **Novelty-Weighted Memory**: Retain surprises, decay routine events
5. **Routine Extraction**: Abstract successful patterns into reusable skills

## Data Types

### Monitor Input/Output

```python
@dataclass
class MonitorTask:
    """Task submitted to the Progress Monitor."""
    task_id: str
    objective: str
    context: dict[str, Any]
    priority: int = 2                    # 0=critical, 4=background
    deadline: datetime | None = None
    parent_task_id: str | None = None    # For subtask tracking
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorResult:
    """Result from Progress Monitor after task completion."""
    task_id: str
    status: MonitorStatus
    outcome_summary: str
    valence: float                       # -1.0 (bad) to +1.0 (good)
    surprise_level: float                # 0.0 (expected) to 1.0 (very surprising)
    progress_delta: float                # Change in progress toward objective
    lessons_learned: list[str]           # Insights to remember
    routines_extracted: list[str]        # New routines discovered
    total_time_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


class MonitorStatus(Enum):
    SUCCESS = "success"                  # Objective achieved
    PARTIAL = "partial"                  # Some progress made
    STUCK = "stuck"                      # No progress despite attempts
    FAILED = "failed"                    # Unrecoverable failure
    TIMEOUT = "timeout"                  # Deadline exceeded
    CANCELLED = "cancelled"              # Externally cancelled
```

### Progress State

```python
@dataclass
class ProgressState:
    """Tracks progress toward an objective."""
    task_id: str
    objective_embedding: list[float]     # Vector representation of goal
    initial_distance: float              # Starting distance to goal
    current_distance: float              # Current distance to goal
    attempts: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    stuck_count: int = 0                 # Consecutive non-progress steps
    last_progress_at: datetime | None = None
    trajectory: list[float] = field(default_factory=list)  # Distance over time

    @property
    def progress_ratio(self) -> float:
        """Progress as ratio from 0 (no progress) to 1 (complete)."""
        if self.initial_distance == 0:
            return 1.0
        return max(0.0, 1.0 - (self.current_distance / self.initial_distance))

    @property
    def is_stuck(self) -> bool:
        """True if system appears stuck (no progress in multiple attempts)."""
        return self.stuck_count >= 3

    @property
    def velocity(self) -> float:
        """Recent rate of progress (positive = moving forward)."""
        if len(self.trajectory) < 2:
            return 0.0
        recent = self.trajectory[-5:]  # Last 5 measurements
        if len(recent) < 2:
            return 0.0
        return (recent[0] - recent[-1]) / len(recent)  # Positive = distance decreasing
```

### Confidence Intervals

```python
@dataclass
class ConfidenceModel:
    """Statistical model for expected outcomes."""
    task_signature: str
    observation_count: int = 0
    success_rate: float = 0.5            # Prior: 50%
    mean_duration_ms: float = 10000.0    # Prior: 10 seconds
    std_duration_ms: float = 5000.0
    mean_valence: float = 0.0            # Prior: neutral
    std_valence: float = 0.5
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update(self, success: bool, duration_ms: int, valence: float) -> None:
        """Update model with new observation using online learning."""
        self.observation_count += 1
        n = self.observation_count

        # Exponential moving average (more weight to recent observations)
        alpha = min(0.3, 2.0 / (n + 1))  # Decay factor

        # Update success rate
        observed_success = 1.0 if success else 0.0
        self.success_rate = (1 - alpha) * self.success_rate + alpha * observed_success

        # Update duration stats (Welford's online algorithm)
        delta = duration_ms - self.mean_duration_ms
        self.mean_duration_ms += alpha * delta
        self.std_duration_ms = math.sqrt(
            (1 - alpha) * (self.std_duration_ms ** 2) + alpha * (delta ** 2)
        )

        # Update valence stats
        delta_v = valence - self.mean_valence
        self.mean_valence += alpha * delta_v
        self.std_valence = math.sqrt(
            (1 - alpha) * (self.std_valence ** 2) + alpha * (delta_v ** 2)
        )

        self.last_updated = datetime.utcnow()

    def is_surprising(
        self,
        success: bool,
        duration_ms: int,
        valence: float,
        threshold_sigma: float = 2.0,
    ) -> tuple[bool, str | None]:
        """Check if outcome is outside confidence interval."""
        reasons = []

        # Check success rate surprise
        expected_success = self.success_rate > 0.5
        if success != expected_success and self.observation_count >= 5:
            if success:
                reasons.append(f"Unexpected success (expected {self.success_rate:.0%} success rate)")
            else:
                reasons.append(f"Unexpected failure (expected {self.success_rate:.0%} success rate)")

        # Check duration surprise
        if self.std_duration_ms > 0:
            z_duration = abs(duration_ms - self.mean_duration_ms) / self.std_duration_ms
            if z_duration > threshold_sigma:
                if duration_ms > self.mean_duration_ms:
                    reasons.append(f"Unusually slow ({duration_ms}ms vs expected {self.mean_duration_ms:.0f}ms)")
                else:
                    reasons.append(f"Unusually fast ({duration_ms}ms vs expected {self.mean_duration_ms:.0f}ms)")

        # Check valence surprise
        if self.std_valence > 0:
            z_valence = abs(valence - self.mean_valence) / self.std_valence
            if z_valence > threshold_sigma:
                reasons.append(f"Unexpected valence ({valence:.2f} vs expected {self.mean_valence:.2f})")

        is_surprising = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else None
        return is_surprising, reason
```

### Valence Signals

```python
@dataclass
class ValenceSignal:
    """Good/bad signal attached to an outcome."""
    task_id: str
    valence: float                       # -1.0 to +1.0
    confidence: float                    # 0.0 to 1.0 (how certain)
    components: dict[str, float]         # Breakdown of valence sources
    timestamp: datetime = field(default_factory=datetime.utcnow)


class ValenceCalculator:
    """Calculates valence (good/bad) signal for outcomes."""

    # Weights for different outcome aspects
    WEIGHTS = {
        "success": 0.4,          # Did it work?
        "efficiency": 0.2,       # Was it fast?
        "quality": 0.2,          # How good was the result?
        "novelty": 0.1,          # Did we learn something new?
        "cost": 0.1,             # Resource usage
    }

    def calculate(
        self,
        result: ArchitectResult,
        confidence_model: ConfidenceModel,
        progress_state: ProgressState,
    ) -> ValenceSignal:
        """Calculate valence signal for an outcome."""
        components = {}

        # Success component: did we achieve the objective?
        if result.status.value == "success":
            components["success"] = 1.0
        elif result.status.value == "partial":
            components["success"] = 0.3
        elif result.status.value == "needs_clarification":
            components["success"] = 0.0  # Neutral - not failure
        else:
            components["success"] = -1.0

        # Efficiency component: faster than expected = good
        if confidence_model.std_duration_ms > 0:
            z_duration = (result.total_time_ms - confidence_model.mean_duration_ms) / confidence_model.std_duration_ms
            components["efficiency"] = max(-1.0, min(1.0, -z_duration * 0.5))  # Negative z = faster = positive valence
        else:
            components["efficiency"] = 0.0

        # Quality component: based on revision count (fewer = better)
        if result.metadata.get("revision_count", 0) == 0:
            components["quality"] = 1.0
        elif result.metadata.get("revision_count", 0) <= 2:
            components["quality"] = 0.5
        else:
            components["quality"] = -0.5

        # Novelty component: did we learn something?
        if result.surprise_flag:
            components["novelty"] = 0.5  # Surprising = interesting = slight positive
        else:
            components["novelty"] = 0.0

        # Cost component: based on resource usage (placeholder)
        components["cost"] = 0.0  # TODO: integrate actual resource tracking

        # Weighted sum
        valence = sum(
            self.WEIGHTS[k] * components[k]
            for k in self.WEIGHTS
        )

        # Confidence based on observation count
        confidence = min(1.0, confidence_model.observation_count / 10.0)

        return ValenceSignal(
            task_id=result.task_id,
            valence=valence,
            confidence=confidence,
            components=components,
        )
```

### Memory Events

```python
@dataclass
class MemoryEvent:
    """An event stored in memory with novelty weighting."""
    event_id: str
    event_type: str                      # "outcome", "surprise", "routine", etc.
    content: dict[str, Any]
    novelty_score: float                 # 0.0 (routine) to 1.0 (very novel)
    importance: float                    # 0.0 to 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.1              # Per-day decay

    @property
    def current_weight(self) -> float:
        """Current memory weight after decay."""
        days_since_access = (datetime.utcnow() - self.last_accessed).total_seconds() / 86400
        base_weight = self.novelty_score * self.importance
        decayed = base_weight * math.exp(-self.decay_rate * days_since_access)
        return max(0.01, decayed)  # Minimum weight to prevent complete forgetting

    def should_forget(self, threshold: float = 0.05) -> bool:
        """True if memory has decayed below threshold."""
        return self.current_weight < threshold and self.novelty_score < 0.3


class MemoryPriority(Enum):
    """Priority levels for memory retention."""
    CRITICAL = "critical"      # Never forget (e.g., safety lessons)
    HIGH = "high"              # Retain for weeks
    NORMAL = "normal"          # Retain for days
    LOW = "low"                # Decay quickly
    EPHEMERAL = "ephemeral"    # Forget within hours
```

### Routines

```python
@dataclass
class Routine:
    """A learned pattern that can be replayed."""
    routine_id: str
    name: str
    description: str
    trigger_signature: str               # When to apply this routine
    steps: list[dict[str, Any]]          # Abstract steps to replay
    success_count: int = 0
    failure_count: int = 0
    avg_duration_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime | None = None

    @property
    def reliability(self) -> float:
        """Success rate of this routine."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Prior
        return self.success_count / total
```

## Core Components

### Monitor Statistics

```python
@dataclass
class MonitorStats:
    """Runtime statistics for the monitor."""
    tasks_processed: int = 0
    successful_tasks: int = 0
    total_valence: float = 0.0
    surprise_count: int = 0

    @property
    def avg_valence(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.total_valence / self.tasks_processed

    @property
    def surprise_rate(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.surprise_count / self.tasks_processed
```

### Progress Monitor (Main Class)

```python
class ProgressMonitor:
    """
    Fourth tier: Meta-layer tracking whether the system makes progress.

    Responsibilities:
    - Track progress across tasks
    - Detect surprising outcomes
    - Manage valence signals
    - Handle novelty-weighted memory
    - Extract reusable routines
    """

    def __init__(
        self,
        architect: AbstractArchitect,
        beads: BeadsClient,
        config: MonitorConfig | None = None,
    ):
        self.architect = architect
        self.beads = beads
        self.config = config or MonitorConfig()
        self.stats = MonitorStats()

        self.progress_tracker = ProgressTracker(beads)
        self.surprise_detector = SurpriseDetector(beads)
        self.valence_manager = ValenceManager(beads)
        self.memory_manager = MemoryManager(beads, self.config)
        self.routine_extractor = RoutineExtractor(beads)

    async def execute(self, task: MonitorTask) -> MonitorResult:
        """Execute task with progress monitoring."""
        start_time = time.monotonic()
        logger.info(f"Monitor received task: {task.task_id}")

        # Initialize progress tracking
        progress_state = await self.progress_tracker.initialize(task)

        # Check for applicable routine
        routine = await self.routine_extractor.find_matching_routine(task)
        if routine and routine.reliability > self.config.routine_reliability_threshold:
            logger.info(f"Applying routine: {routine.name}")

        # Convert to architect task
        architect_task = self._to_architect_task(task, routine)

        try:
            # Execute with progress monitoring
            result = await self._execute_with_monitoring(
                architect_task, progress_state, task.deadline
            )
        except TimeoutError:
            # Handle deadline exceeded
            total_time_ms = int((time.monotonic() - start_time) * 1000)
            await self.routine_extractor.record_failure(task)
            return MonitorResult(
                task_id=task.task_id,
                status=MonitorStatus.TIMEOUT,
                outcome_summary="Task exceeded deadline",
                valence=-0.5,
                surprise_level=0.0,
                progress_delta=0.0,
                lessons_learned=["Task timed out before completion"],
                routines_extracted=[],
                total_time_ms=total_time_ms,
                metadata={"timeout": True},
            )

        # Process outcome
        monitor_result = await self._process_outcome(
            task, result, progress_state, start_time
        )

        # Record failure for routine tracking if not successful
        if monitor_result.status in (MonitorStatus.FAILED, MonitorStatus.STUCK):
            await self.routine_extractor.record_failure(task)

        # Update stats
        self.stats.tasks_processed += 1
        self.stats.total_valence += monitor_result.valence
        if monitor_result.status == MonitorStatus.SUCCESS:
            self.stats.successful_tasks += 1
        if monitor_result.surprise_level > 0.5:
            self.stats.surprise_count += 1

        # Memory management
        await self._update_memory(task, monitor_result)

        return monitor_result

    async def _execute_with_monitoring(
        self,
        task: ArchitectTask,
        progress_state: ProgressState,
        deadline: datetime | None,
    ) -> ArchitectResult:
        """Execute task with periodic progress checks."""
        # Start architect execution
        result_future = asyncio.create_task(self.architect.execute(task))

        # Monitor progress while executing
        check_interval = self.config.progress_check_interval_seconds

        while not result_future.done():
            # Check deadline
            if deadline and datetime.utcnow() > deadline:
                result_future.cancel()
                raise TimeoutError(f"Task {task.task_id} exceeded deadline")

            # Check for stuck state
            if progress_state.is_stuck:
                logger.warning(f"Task {task.task_id} appears stuck")
                # Could trigger intervention here

            await asyncio.sleep(check_interval)

        return await result_future

    async def _process_outcome(
        self,
        task: MonitorTask,
        result: ArchitectResult,
        progress_state: ProgressState,
        start_time: float,
    ) -> MonitorResult:
        """Process architect result into monitor result."""
        total_time_ms = int((time.monotonic() - start_time) * 1000)

        # Update progress state
        progress_delta = await self.progress_tracker.update(
            progress_state, result
        )

        # Get confidence model and check for surprise
        confidence_model = await self.surprise_detector.get_model(task)
        is_success = result.status.value in ("success", "partial")

        # Calculate valence
        valence_signal = self.valence_manager.calculate(
            result, confidence_model, progress_state
        )

        # Detect surprise
        is_surprising, surprise_reason = confidence_model.is_surprising(
            success=is_success,
            duration_ms=total_time_ms,
            valence=valence_signal.valence,
        )

        # Update confidence model with observation
        confidence_model.update(
            success=is_success,
            duration_ms=total_time_ms,
            valence=valence_signal.valence,
        )
        await self.surprise_detector.save_model(confidence_model)

        # Extract lessons and routines
        lessons = await self._extract_lessons(task, result, is_surprising)
        routines = await self.routine_extractor.extract_if_successful(task, result)

        # Determine status
        status = self._determine_status(result, progress_state)

        return MonitorResult(
            task_id=task.task_id,
            status=status,
            outcome_summary=result.summary,
            valence=valence_signal.valence,
            surprise_level=1.0 if is_surprising else 0.0,
            progress_delta=progress_delta,
            lessons_learned=lessons,
            routines_extracted=[r.name for r in routines],
            total_time_ms=total_time_ms,
            metadata={
                "architect_status": result.status.value,
                "surprise_reason": surprise_reason,
                "valence_components": valence_signal.components,
                "progress_ratio": progress_state.progress_ratio,
            },
        )

    def _determine_status(
        self,
        result: ArchitectResult,
        progress_state: ProgressState,
    ) -> MonitorStatus:
        """Map architect result to monitor status."""
        if result.status.value == "success":
            return MonitorStatus.SUCCESS
        elif result.status.value == "partial":
            if progress_state.velocity > 0:
                return MonitorStatus.PARTIAL
            else:
                return MonitorStatus.STUCK
        elif result.status.value == "timeout":
            return MonitorStatus.TIMEOUT
        elif progress_state.is_stuck:
            return MonitorStatus.STUCK
        else:
            return MonitorStatus.FAILED

    async def _extract_lessons(
        self,
        task: MonitorTask,
        result: ArchitectResult,
        is_surprising: bool,
    ) -> list[str]:
        """Extract lessons learned from outcome."""
        lessons = []

        if is_surprising:
            if result.surprise_reason:
                lessons.append(f"Surprise: {result.surprise_reason}")

        if result.status.value == "failure":
            lessons.append(f"Failure mode: {result.error_detail or 'unknown'}")

        # Could use LLM to generate more sophisticated lessons
        return lessons

    async def _update_memory(
        self,
        task: MonitorTask,
        result: MonitorResult,
    ) -> None:
        """Update memory with outcome, applying novelty weighting."""
        # Calculate novelty (surprising = novel)
        novelty = result.surprise_level

        # Calculate importance (high valence magnitude = important)
        importance = abs(result.valence) * 0.5 + 0.5

        # Determine priority
        if result.status == MonitorStatus.FAILED and result.valence < -0.5:
            priority = MemoryPriority.HIGH  # Remember failures
        elif result.surprise_level > 0.5:
            priority = MemoryPriority.HIGH  # Remember surprises
        elif result.status == MonitorStatus.SUCCESS:
            priority = MemoryPriority.NORMAL
        else:
            priority = MemoryPriority.LOW

        # Store memory event
        await self.memory_manager.store(
            event_type="outcome",
            content={
                "task_id": task.task_id,
                "objective": task.objective,
                "status": result.status.value,
                "valence": result.valence,
                "lessons": result.lessons_learned,
            },
            novelty_score=novelty,
            importance=importance,
            priority=priority,
        )

        # Run memory cleanup
        await self.memory_manager.cleanup()

    def _to_architect_task(
        self,
        task: MonitorTask,
        routine: Routine | None,
    ) -> ArchitectTask:
        """Convert monitor task to architect task."""
        # Copy hints to avoid mutating original task metadata
        hints = list(task.metadata.get("hints", []))
        if routine:
            hints.append(f"Consider applying routine: {routine.name}")

        return ArchitectTask(
            task_id=task.task_id,
            objective=task.objective,
            context=task.context,
            constraints=task.metadata.get("constraints", []),
            hints=hints,
            timeout_ms=self._calculate_timeout(task),
            metadata=task.metadata,
        )

    def _calculate_timeout(self, task: MonitorTask) -> int:
        """Calculate timeout for architect task."""
        if task.deadline:
            remaining = (task.deadline - datetime.utcnow()).total_seconds()
            return max(1000, int(remaining * 1000 * 0.9))  # 90% of remaining time
        return self.config.default_timeout_ms
```

### Progress Tracker

```python
class ProgressTracker:
    """Tracks progress toward objectives."""

    def __init__(self, beads: BeadsClient):
        self.beads = beads

    async def initialize(self, task: MonitorTask) -> ProgressState:
        """Initialize progress tracking for a task."""
        # Generate objective embedding (placeholder - would use actual embedding model)
        objective_embedding = self._hash_to_embedding(task.objective)

        state = ProgressState(
            task_id=task.task_id,
            objective_embedding=objective_embedding,
            initial_distance=1.0,  # Start at max distance
            current_distance=1.0,
        )

        await self.beads.set(
            f"progress/{task.task_id}",
            self._state_to_dict(state),
        )

        return state

    async def update(
        self,
        state: ProgressState,
        result: ArchitectResult,
    ) -> float:
        """Update progress state and return delta."""
        state.attempts += 1

        # Estimate new distance based on result
        if result.status.value == "success":
            new_distance = 0.0
            state.successful_steps += 1
            state.stuck_count = 0
        elif result.status.value == "partial":
            # Partial progress
            new_distance = state.current_distance * 0.7
            state.successful_steps += 1
            state.stuck_count = 0
        else:
            # No progress or regression
            new_distance = min(1.0, state.current_distance * 1.1)
            state.failed_steps += 1
            state.stuck_count += 1

        # Calculate delta (positive = progress)
        delta = state.current_distance - new_distance
        state.current_distance = new_distance
        state.trajectory.append(new_distance)

        if delta > 0:
            state.last_progress_at = datetime.utcnow()

        # Persist
        await self.beads.set(
            f"progress/{state.task_id}",
            self._state_to_dict(state),
        )

        return delta

    def _hash_to_embedding(self, text: str) -> list[float]:
        """Placeholder: hash text to pseudo-embedding."""
        h = hashlib.sha256(text.encode()).hexdigest()
        # Convert to 8 floats between -1 and 1
        return [
            (int(h[i:i+8], 16) / (2**32) - 0.5) * 2
            for i in range(0, 64, 8)
        ]

    def _state_to_dict(self, state: ProgressState) -> dict:
        """Serialize progress state."""
        return {
            "task_id": state.task_id,
            "objective_embedding": state.objective_embedding,
            "initial_distance": state.initial_distance,
            "current_distance": state.current_distance,
            "attempts": state.attempts,
            "successful_steps": state.successful_steps,
            "failed_steps": state.failed_steps,
            "stuck_count": state.stuck_count,
            "last_progress_at": state.last_progress_at.isoformat() if state.last_progress_at else None,
            "trajectory": state.trajectory[-20:],  # Keep last 20
        }
```

### Surprise Detector

```python
class SurpriseDetector:
    """Detects surprising outcomes using confidence intervals."""

    def __init__(self, beads: BeadsClient):
        self.beads = beads

    async def get_model(self, task: MonitorTask) -> ConfidenceModel:
        """Get or create confidence model for task type."""
        signature = self._task_signature(task)
        key = f"confidence/{signature}"

        data = await self.beads.get(key)
        if data:
            return ConfidenceModel(
                task_signature=signature,
                observation_count=data.get("observation_count", 0),
                success_rate=data.get("success_rate", 0.5),
                mean_duration_ms=data.get("mean_duration_ms", 10000.0),
                std_duration_ms=data.get("std_duration_ms", 5000.0),
                mean_valence=data.get("mean_valence", 0.0),
                std_valence=data.get("std_valence", 0.5),
                last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.utcnow(),
            )

        return ConfidenceModel(task_signature=signature)

    async def save_model(self, model: ConfidenceModel) -> None:
        """Persist confidence model."""
        key = f"confidence/{model.task_signature}"
        await self.beads.set(key, {
            "task_signature": model.task_signature,
            "observation_count": model.observation_count,
            "success_rate": model.success_rate,
            "mean_duration_ms": model.mean_duration_ms,
            "std_duration_ms": model.std_duration_ms,
            "mean_valence": model.mean_valence,
            "std_valence": model.std_valence,
            "last_updated": model.last_updated.isoformat(),
        })

    def _task_signature(self, task: MonitorTask) -> str:
        """Generate signature for task type (for grouping similar tasks)."""
        # Simple approach: hash first 100 chars of objective
        text = task.objective[:100].lower()
        return hashlib.sha256(text.encode()).hexdigest()[:12]
```

### Valence Manager

```python
class ValenceManager:
    """Manages valence (good/bad) signals for outcomes."""

    def __init__(self, beads: BeadsClient):
        self.beads = beads
        self.calculator = ValenceCalculator()

    def calculate(
        self,
        result: ArchitectResult,
        confidence_model: ConfidenceModel,
        progress_state: ProgressState,
    ) -> ValenceSignal:
        """Calculate valence signal for outcome."""
        return self.calculator.calculate(result, confidence_model, progress_state)

    async def get_average_valence(
        self,
        task_signature: str,
        window_hours: int = 24,
    ) -> float:
        """Get average valence for task type over time window."""
        key = f"valence/history/{task_signature}"
        history = await self.beads.get(key) or []

        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [
            v["valence"] for v in history
            if datetime.fromisoformat(v["timestamp"]) > cutoff
        ]

        if not recent:
            return 0.0
        return statistics.mean(recent)

    async def record_valence(self, signal: ValenceSignal, task_signature: str) -> None:
        """Record valence signal in history."""
        key = f"valence/history/{task_signature}"
        await self.beads.append(key, {
            "task_id": signal.task_id,
            "valence": signal.valence,
            "confidence": signal.confidence,
            "timestamp": signal.timestamp.isoformat(),
        })
```

### Memory Manager

```python
class MemoryManager:
    """Manages novelty-weighted memory with forgetting."""

    def __init__(self, beads: BeadsClient, config: MonitorConfig):
        self.beads = beads
        self.config = config

    async def store(
        self,
        event_type: str,
        content: dict[str, Any],
        novelty_score: float,
        importance: float,
        priority: MemoryPriority,
    ) -> MemoryEvent:
        """Store memory event with novelty weighting."""
        event_id = hashlib.sha256(
            f"{event_type}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        # Decay rate based on priority
        decay_rates = {
            MemoryPriority.CRITICAL: 0.0,    # Never decay
            MemoryPriority.HIGH: 0.05,       # ~2 weeks half-life
            MemoryPriority.NORMAL: 0.1,      # ~1 week half-life
            MemoryPriority.LOW: 0.3,         # ~2 days half-life
            MemoryPriority.EPHEMERAL: 1.0,   # Hours
        }

        event = MemoryEvent(
            event_id=event_id,
            event_type=event_type,
            content=content,
            novelty_score=novelty_score,
            importance=importance,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            decay_rate=decay_rates[priority],
        )

        await self.beads.set(f"memory/{event_id}", self._event_to_dict(event))
        await self.beads.append(f"memory/index/{event_type}", event_id)

        return event

    async def recall(
        self,
        event_type: str | None = None,
        min_weight: float = 0.1,
        limit: int = 10,
    ) -> list[MemoryEvent]:
        """Recall memories above weight threshold."""
        if event_type:
            event_ids = await self.beads.get(f"memory/index/{event_type}") or []
        else:
            # Get all event types
            event_ids = []
            for t in ["outcome", "surprise", "routine", "lesson"]:
                ids = await self.beads.get(f"memory/index/{t}") or []
                event_ids.extend(ids)

        events = []
        for event_id in event_ids:
            data = await self.beads.get(f"memory/{event_id}")
            if data:
                event = self._dict_to_event(data)
                if event.current_weight >= min_weight:
                    events.append(event)

        # Sort by weight and return top N
        events.sort(key=lambda e: e.current_weight, reverse=True)
        return events[:limit]

    async def cleanup(self) -> int:
        """Remove memories that have decayed below threshold."""
        removed = 0
        threshold = self.config.memory_forget_threshold

        for event_type in ["outcome", "surprise", "routine", "lesson"]:
            event_ids = await self.beads.get(f"memory/index/{event_type}") or []
            remaining = []

            for event_id in event_ids:
                data = await self.beads.get(f"memory/{event_id}")
                if data:
                    event = self._dict_to_event(data)
                    if event.should_forget(threshold):
                        await self.beads.delete(f"memory/{event_id}")
                        removed += 1
                        logger.debug(f"Forgot memory: {event_id}")
                    else:
                        remaining.append(event_id)

            await self.beads.set(f"memory/index/{event_type}", remaining)

        return removed

    def _event_to_dict(self, event: MemoryEvent) -> dict:
        """Serialize memory event."""
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "content": event.content,
            "novelty_score": event.novelty_score,
            "importance": event.importance,
            "created_at": event.created_at.isoformat(),
            "last_accessed": event.last_accessed.isoformat(),
            "access_count": event.access_count,
            "decay_rate": event.decay_rate,
        }

    def _dict_to_event(self, data: dict) -> MemoryEvent:
        """Deserialize memory event."""
        return MemoryEvent(
            event_id=data["event_id"],
            event_type=data["event_type"],
            content=data["content"],
            novelty_score=data["novelty_score"],
            importance=data["importance"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            decay_rate=data.get("decay_rate", 0.1),
        )
```

### Routine Extractor

```python
class RoutineExtractor:
    """Extracts successful patterns into reusable routines."""

    def __init__(self, beads: BeadsClient):
        self.beads = beads

    async def find_matching_routine(self, task: MonitorTask) -> Routine | None:
        """Find a routine that matches the task."""
        signature = self._task_signature(task)

        # Check for exact match first
        routine_data = await self.beads.get(f"routines/{signature}")
        if routine_data:
            routine = self._dict_to_routine(routine_data)
            if routine.reliability >= 0.7:
                return routine

        # Could add fuzzy matching here
        return None

    async def extract_if_successful(
        self,
        task: MonitorTask,
        result: ArchitectResult,
    ) -> list[Routine]:
        """Extract routine if task was successful enough."""
        if result.status.value not in ("success", "partial"):
            return []

        signature = self._task_signature(task)
        key = f"routines/{signature}"

        existing = await self.beads.get(key)
        if existing:
            # Update existing routine
            routine = self._dict_to_routine(existing)
            routine.success_count += 1
            routine.last_used = datetime.utcnow()
            # Update average duration
            n = routine.success_count + routine.failure_count
            routine.avg_duration_ms = (
                (routine.avg_duration_ms * (n - 1) + result.total_time_ms) / n
            )
        else:
            # Create new routine
            routine = Routine(
                routine_id=signature,
                name=f"Routine for: {task.objective[:50]}",
                description=task.objective,
                trigger_signature=signature,
                steps=self._extract_steps(result),
                success_count=1,
            )

        await self.beads.set(key, self._routine_to_dict(routine))
        return [routine]

    async def record_failure(self, task: MonitorTask) -> None:
        """Record routine failure."""
        signature = self._task_signature(task)
        key = f"routines/{signature}"

        existing = await self.beads.get(key)
        if existing:
            routine = self._dict_to_routine(existing)
            routine.failure_count += 1
            await self.beads.set(key, self._routine_to_dict(routine))

    def _task_signature(self, task: MonitorTask) -> str:
        """Generate signature for task type."""
        text = task.objective[:100].lower()
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def _extract_steps(self, result: ArchitectResult) -> list[dict[str, Any]]:
        """Extract abstract steps from successful result."""
        # Placeholder - would extract from actual execution trace
        return [
            {"action": "plan", "description": "Generate strategy"},
            {"action": "execute", "description": "Execute via dispatcher"},
            {"action": "validate", "description": "Validate outcome"},
        ]

    def _routine_to_dict(self, routine: Routine) -> dict:
        """Serialize routine."""
        return {
            "routine_id": routine.routine_id,
            "name": routine.name,
            "description": routine.description,
            "trigger_signature": routine.trigger_signature,
            "steps": routine.steps,
            "success_count": routine.success_count,
            "failure_count": routine.failure_count,
            "avg_duration_ms": routine.avg_duration_ms,
            "created_at": routine.created_at.isoformat(),
            "last_used": routine.last_used.isoformat() if routine.last_used else None,
        }

    def _dict_to_routine(self, data: dict) -> Routine:
        """Deserialize routine."""
        return Routine(
            routine_id=data["routine_id"],
            name=data["name"],
            description=data["description"],
            trigger_signature=data["trigger_signature"],
            steps=data["steps"],
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            avg_duration_ms=data.get("avg_duration_ms", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
        )
```

## Configuration

```python
@dataclass
class MonitorConfig:
    """Configuration for Progress Monitor."""
    # Timeouts
    default_timeout_ms: int = 300000             # 5 minutes default
    progress_check_interval_seconds: float = 5.0

    # Surprise detection
    surprise_threshold_sigma: float = 2.0        # Standard deviations for surprise

    # Memory management
    memory_forget_threshold: float = 0.05        # Below this weight, forget
    memory_cleanup_interval_hours: int = 1

    # Progress tracking
    stuck_threshold_attempts: int = 3            # Attempts without progress = stuck

    # Routine extraction
    min_successes_for_routine: int = 2           # Successes before saving routine
    routine_reliability_threshold: float = 0.7   # Reliability to apply routine
```

```yaml
# monitor_config.yaml
timeouts:
  default_ms: 300000
  progress_check_seconds: 5.0

surprise:
  threshold_sigma: 2.0

memory:
  forget_threshold: 0.05
  cleanup_interval_hours: 1
  decay_rates:
    critical: 0.0
    high: 0.05
    normal: 0.1
    low: 0.3
    ephemeral: 1.0

progress:
  stuck_threshold_attempts: 3

routines:
  min_successes: 2
  reliability_threshold: 0.7
```

## Beads Key Schema

```python
BEADS_KEYS = {
    # Progress tracking
    "progress": "progress/{task_id}",

    # Confidence models
    "confidence": "confidence/{task_signature}",

    # Valence history
    "valence_history": "valence/history/{task_signature}",

    # Memory events
    "memory_event": "memory/{event_id}",
    "memory_index": "memory/index/{event_type}",

    # Routines
    "routine": "routines/{signature}",

    # System state
    "system_health": "system/health/monitor",
    "system_stats": "system/stats/monitor",
}
```

## Usage Example

```python
async def main():
    # Initialize components
    beads = BeadsClient()
    architect = AbstractArchitect(beads=beads)
    monitor = ProgressMonitor(architect=architect, beads=beads)

    # Submit task
    task = MonitorTask(
        task_id="task-001",
        objective="Solve the two-sum problem in Python",
        context={
            "problem_description": "Given an array of integers...",
            "expected_complexity": "O(n)",
        },
        priority=2,
    )

    # Execute with monitoring
    result = await monitor.execute(task)

    print(f"Status: {result.status.value}")
    print(f"Valence: {result.valence:.2f}")
    print(f"Surprise level: {result.surprise_level:.2f}")
    print(f"Progress delta: {result.progress_delta:.2f}")
    print(f"Lessons: {result.lessons_learned}")
    print(f"Routines extracted: {result.routines_extracted}")
```

## Testing

```python
import pytest


@pytest.fixture
def beads():
    return BeadsClient(backend=MemoryBackend())


@pytest.fixture
def monitor(beads):
    architect = MockArchitect()
    return ProgressMonitor(architect=architect, beads=beads)


@pytest.mark.asyncio
async def test_successful_task(monitor):
    task = MonitorTask(
        task_id="test-1",
        objective="Simple test task",
        context={},
    )

    result = await monitor.execute(task)

    assert result.status == MonitorStatus.SUCCESS
    assert result.valence > 0


@pytest.mark.asyncio
async def test_surprise_detection(monitor, beads):
    # Build up baseline expectations
    for i in range(5):
        task = MonitorTask(
            task_id=f"baseline-{i}",
            objective="Consistent task",
            context={},
        )
        await monitor.execute(task)

    # Now do something surprising
    # (mock architect returns unexpected result)
    monitor.architect.next_result = ArchitectResult(
        status=ArchitectStatus.FAILURE,
        # ... unexpected failure
    )

    task = MonitorTask(
        task_id="surprising",
        objective="Consistent task",  # Same signature
        context={},
    )

    result = await monitor.execute(task)

    assert result.surprise_level > 0.5


@pytest.mark.asyncio
async def test_routine_extraction(monitor, beads):
    # First success - no routine yet
    task1 = MonitorTask(
        task_id="routine-1",
        objective="Repeatable task",
        context={},
    )
    result1 = await monitor.execute(task1)
    assert len(result1.routines_extracted) == 1

    # Second success - routine should be applied
    task2 = MonitorTask(
        task_id="routine-2",
        objective="Repeatable task",
        context={},
    )
    result2 = await monitor.execute(task2)

    # Check routine was found
    routine = await monitor.routine_extractor.find_matching_routine(task2)
    assert routine is not None
    assert routine.success_count == 2


@pytest.mark.asyncio
async def test_memory_decay(monitor, beads):
    # Store a low-priority memory
    await monitor.memory_manager.store(
        event_type="test",
        content={"data": "ephemeral"},
        novelty_score=0.1,
        importance=0.1,
        priority=MemoryPriority.EPHEMERAL,
    )

    # Immediate recall should work
    memories = await monitor.memory_manager.recall(event_type="test")
    assert len(memories) == 1

    # After decay (simulated), should be forgotten
    # (In real test, would mock datetime or wait)


@pytest.mark.asyncio
async def test_valence_calculation(monitor):
    confidence = ConfidenceModel(task_signature="test")
    progress = ProgressState(
        task_id="test",
        objective_embedding=[0.0] * 8,
        initial_distance=1.0,
        current_distance=0.5,
    )

    result = ArchitectResult(
        task_id="test",
        status=ArchitectStatus.SUCCESS,
        summary="Success",
        total_time_ms=5000,
    )

    signal = monitor.valence_manager.calculate(result, confidence, progress)

    assert signal.valence > 0  # Success should be positive
    assert "success" in signal.components
```

## Integration with Other Tiers

### Receiving Results from Architect

```python
# The Progress Monitor wraps the Architect
result = await self.architect.execute(task)

# Get confidence model for surprise detection
confidence_model = await self.surprise_detector.get_model(task)

# Process result
valence_signal = self.valence_manager.calculate(result, confidence_model, progress_state)
is_surprising, reason = confidence_model.is_surprising(
    success=result.status.value == "success",
    duration_ms=result.total_time_ms,
    valence=valence_signal.valence,
)
```

### Providing Training Signal to Dispatcher

```python
# After processing outcome, signal routing quality
await self.beads.append("routing/feedback", {
    "task_signature": signature,
    "routing_success": result.status == MonitorStatus.SUCCESS,
    "valence": valence.valence,
    "timestamp": datetime.utcnow().isoformat(),
})
```

### System Health Monitoring

```python
# Add stats tracking to ProgressMonitor.__init__:
@dataclass
class MonitorStats:
    """Runtime statistics for the monitor."""
    tasks_processed: int = 0
    successful_tasks: int = 0
    total_valence: float = 0.0
    surprise_count: int = 0

    @property
    def avg_valence(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.total_valence / self.tasks_processed

    @property
    def surprise_rate(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.surprise_count / self.tasks_processed


# In ProgressMonitor class:
async def update_health(self) -> None:
    """Update system health status in Beads."""
    # Count routines by listing keys
    routine_keys = await self.beads.list_keys("routines/")

    await self.beads.set("system/health/monitor", {
        "status": "healthy",
        "tasks_processed": self.stats.tasks_processed,
        "avg_valence": self.stats.avg_valence,
        "surprise_rate": self.stats.surprise_rate,
        "memory_usage": len(await self.memory_manager.recall()),
        "routines_count": len(routine_keys),
        "last_update": datetime.utcnow().isoformat(),
    })
```
