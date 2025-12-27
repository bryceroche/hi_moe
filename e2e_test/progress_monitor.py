"""Progress monitoring with valence and surprise detection.

Implements:
- hi_moe-x3j: Valence field for state tracking (positive/negative momentum)
- hi_moe-9wm: Surprise detector (unexpected outcomes)
- hi_moe-d2j: Vector trajectory tracking via embeddings

These features enable the system to:
- Track emotional/momentum state of task execution
- Detect when outcomes differ from expectations
- Trace solution paths through embedding space
"""
from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Valence Tracking (hi_moe-x3j)
# =============================================================================

@dataclass
class ValenceState:
    """Tracks positive/negative momentum of task execution.

    Valence represents the "emotional charge" of progress:
    - Positive valence: things are going well, momentum is good
    - Negative valence: struggling, frustration building
    - Neutral: steady state, no strong signal
    """
    value: float = 0.0  # -1.0 to 1.0
    confidence: float = 0.5  # How certain we are
    trend: str = "stable"  # "improving", "declining", "stable"
    history: list[float] = field(default_factory=list)
    max_history: int = 20

    def update(self, outcome: float, weight: float = 0.3) -> None:
        """Update valence based on outcome (-1 to 1)."""
        # Clamp outcome
        outcome = max(-1.0, min(1.0, outcome))

        # EMA update
        self.value = (1 - weight) * self.value + weight * outcome

        # Track history
        self.history.append(outcome)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Update trend
        if len(self.history) >= 3:
            recent = self.history[-3:]
            if all(r > 0 for r in recent):
                self.trend = "improving"
            elif all(r < 0 for r in recent):
                self.trend = "declining"
            else:
                self.trend = "stable"

        # Update confidence based on consistency
        if len(self.history) >= 5:
            variance = np.var(self.history[-5:])
            self.confidence = 1.0 / (1.0 + variance * 10)

    def get_signal(self) -> dict:
        """Get current valence signal."""
        return {
            "value": self.value,
            "confidence": self.confidence,
            "trend": self.trend,
            "interpretation": self._interpret(),
        }

    def _interpret(self) -> str:
        """Human-readable interpretation."""
        if self.value > 0.5 and self.trend == "improving":
            return "Strong positive momentum"
        elif self.value > 0.2:
            return "Positive progress"
        elif self.value < -0.5 and self.trend == "declining":
            return "Significant struggles, consider strategy change"
        elif self.value < -0.2:
            return "Some difficulties"
        else:
            return "Neutral/steady state"


class ValenceTracker:
    """Tracks valence across multiple contexts.

    Usage:
        tracker = ValenceTracker()

        # Record outcomes
        tracker.record("task-1", success=True, difficulty="hard")
        tracker.record("task-1", success=False, difficulty="medium")

        # Get valence
        state = tracker.get_valence("task-1")
        print(state.get_signal())
    """

    def __init__(self):
        self.states: dict[str, ValenceState] = {}
        self.global_state = ValenceState()

    def record(
        self,
        context_id: str,
        success: bool,
        difficulty: str = "medium",
        confidence: float = 1.0,
    ) -> ValenceState:
        """Record an outcome and update valence.

        Args:
            context_id: Task or session ID
            success: Whether the outcome was successful
            difficulty: Problem difficulty (affects valence magnitude)
            confidence: How confident in this outcome
        """
        if context_id not in self.states:
            self.states[context_id] = ValenceState()

        # Compute outcome value
        difficulty_multiplier = {
            "easy": 0.5,
            "medium": 1.0,
            "hard": 1.5,
        }.get(difficulty, 1.0)

        if success:
            outcome = 0.5 * difficulty_multiplier * confidence
        else:
            outcome = -0.5 * difficulty_multiplier * confidence

        # Update context-specific valence
        state = self.states[context_id]
        state.update(outcome)

        # Update global valence
        self.global_state.update(outcome, weight=0.1)

        logger.debug(
            f"[ValenceTracker] {context_id}: {'success' if success else 'failure'} "
            f"-> valence={state.value:.2f} ({state.trend})"
        )

        return state

    def get_valence(self, context_id: str) -> ValenceState:
        """Get valence for a context."""
        return self.states.get(context_id, ValenceState())

    def get_global_valence(self) -> ValenceState:
        """Get global valence across all contexts."""
        return self.global_state

    def should_escalate(self, context_id: str) -> bool:
        """Check if valence suggests escalation needed."""
        state = self.states.get(context_id)
        if not state:
            return False

        return state.value < -0.4 and state.trend == "declining"

    def should_celebrate(self, context_id: str) -> bool:
        """Check if valence suggests success momentum."""
        state = self.states.get(context_id)
        if not state:
            return False

        return state.value > 0.6 and state.trend == "improving"


# =============================================================================
# Surprise Detection (hi_moe-9wm)
# =============================================================================

@dataclass
class SurpriseEvent:
    """An unexpected outcome that deviates from prediction."""
    timestamp: str
    context_id: str
    expected: Any
    actual: Any
    surprise_score: float  # 0-1, higher = more surprising
    category: str  # "positive_surprise", "negative_surprise", "neutral"
    details: dict = field(default_factory=dict)


class SurpriseDetector:
    """Detects when outcomes differ significantly from expectations.

    Tracks prediction confidence and actual outcomes to identify:
    - Positive surprises: succeeded when expected to fail
    - Negative surprises: failed when expected to succeed
    - Anomalies: outcomes that don't fit patterns

    Usage:
        detector = SurpriseDetector()

        # Set expectation
        detector.set_expectation("task-1", success_prob=0.8)

        # Record actual outcome
        surprise = detector.record_outcome("task-1", success=False)
        if surprise and surprise.surprise_score > 0.5:
            print(f"Surprising outcome: {surprise.category}")
    """

    def __init__(self, surprise_threshold: float = 0.3):
        """
        Args:
            surprise_threshold: Minimum surprise score to report
        """
        self.surprise_threshold = surprise_threshold
        self.expectations: dict[str, dict] = {}
        self.history: list[SurpriseEvent] = []

        # Baseline probabilities by context type
        self.baseline_probs = {
            "easy": 0.9,
            "medium": 0.7,
            "hard": 0.4,
            "default": 0.6,
        }

    def set_expectation(
        self,
        context_id: str,
        success_prob: float | None = None,
        expected_time_ms: int | None = None,
        expected_specialist: str | None = None,
        difficulty: str = "medium",
    ) -> None:
        """Set expectations for a task."""
        if success_prob is None:
            success_prob = self.baseline_probs.get(difficulty, self.baseline_probs["default"])

        self.expectations[context_id] = {
            "success_prob": success_prob,
            "expected_time_ms": expected_time_ms,
            "expected_specialist": expected_specialist,
            "difficulty": difficulty,
            "set_at": datetime.now().isoformat(),
        }

    def record_outcome(
        self,
        context_id: str,
        success: bool,
        time_ms: int | None = None,
        specialist: str | None = None,
    ) -> SurpriseEvent | None:
        """Record actual outcome and detect surprise."""
        expectation = self.expectations.get(context_id, {})

        if not expectation:
            # No expectation set, can't detect surprise
            return None

        success_prob = expectation.get("success_prob", 0.6)
        expected_time = expectation.get("expected_time_ms")
        expected_specialist = expectation.get("expected_specialist")

        # Compute surprise score
        surprise_score = 0.0
        details = {}

        # Success/failure surprise
        if success:
            # Succeeded - surprise if low probability expected
            if success_prob < 0.5:
                surprise_score = 1.0 - success_prob
                details["success_surprise"] = f"Succeeded with only {success_prob:.0%} expected"
        else:
            # Failed - surprise if high probability expected
            if success_prob > 0.5:
                surprise_score = success_prob
                details["failure_surprise"] = f"Failed despite {success_prob:.0%} expected"

        # Time surprise
        if time_ms is not None and expected_time is not None:
            time_ratio = time_ms / expected_time
            if time_ratio > 2.0 or time_ratio < 0.5:
                time_surprise = min(abs(math.log(time_ratio)) / 2, 0.5)
                surprise_score = max(surprise_score, time_surprise)
                details["time_surprise"] = f"Expected {expected_time}ms, got {time_ms}ms"

        # Specialist surprise
        if specialist and expected_specialist and specialist != expected_specialist:
            details["specialist_surprise"] = f"Expected {expected_specialist}, used {specialist}"
            surprise_score = max(surprise_score, 0.3)

        # Only report if above threshold
        if surprise_score < self.surprise_threshold:
            return None

        # Categorize
        if success and success_prob < 0.5:
            category = "positive_surprise"
        elif not success and success_prob > 0.5:
            category = "negative_surprise"
        else:
            category = "neutral"

        event = SurpriseEvent(
            timestamp=datetime.now().isoformat(),
            context_id=context_id,
            expected={"success_prob": success_prob, **expectation},
            actual={"success": success, "time_ms": time_ms, "specialist": specialist},
            surprise_score=surprise_score,
            category=category,
            details=details,
        )

        self.history.append(event)
        logger.info(f"[SurpriseDetector] {category}: {context_id} (score={surprise_score:.2f})")

        return event

    def get_recent_surprises(self, limit: int = 10) -> list[SurpriseEvent]:
        """Get recent surprise events."""
        return self.history[-limit:]

    def get_surprise_rate(self, window: int = 20) -> float:
        """Get rate of surprising outcomes in recent history.

        Returns the proportion of recent tasks that were surprising.
        Note: This requires external tracking of total outcomes, not just surprises.
        For now, returns proportion of history that is surprising (always 1.0 if only surprises stored).
        """
        if not self.history:
            return 0.0
        # Return count relative to window size as a density metric
        return min(len(self.history), window) / window

    def analyze_patterns(self) -> dict:
        """Analyze surprise patterns."""
        if not self.history:
            return {"message": "No surprises recorded"}

        positive = [s for s in self.history if s.category == "positive_surprise"]
        negative = [s for s in self.history if s.category == "negative_surprise"]

        return {
            "total_surprises": len(self.history),
            "positive_surprises": len(positive),
            "negative_surprises": len(negative),
            "avg_surprise_score": sum(s.surprise_score for s in self.history) / len(self.history),
            "surprise_rate": self.get_surprise_rate(),
        }


# =============================================================================
# Vector Trajectory Tracking (hi_moe-d2j)
# =============================================================================

@dataclass
class TrajectoryPoint:
    """A point in the solution trajectory."""
    timestamp: str
    context_id: str
    embedding: list[float]
    label: str  # "start", "intermediate", "success", "failure"
    metadata: dict = field(default_factory=dict)


class TrajectoryTracker:
    """Tracks solution paths through embedding space.

    Converts task objectives and states to embeddings, then tracks
    how the solution evolves through the space.

    Usage:
        tracker = TrajectoryTracker()

        # Track solution path
        tracker.add_point("task-1", "start", "Implement sorting algorithm")
        tracker.add_point("task-1", "intermediate", "Using quicksort approach")
        tracker.add_point("task-1", "success", "Completed with tests passing")

        # Analyze trajectory
        analysis = tracker.analyze("task-1")
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.trajectories: dict[str, list[TrajectoryPoint]] = {}
        self._embedding_cache: dict[str, list[float]] = {}

    def add_point(
        self,
        context_id: str,
        label: str,
        description: str,
        metadata: dict | None = None,
    ) -> TrajectoryPoint:
        """Add a point to the trajectory."""
        if context_id not in self.trajectories:
            self.trajectories[context_id] = []

        embedding = self._text_to_embedding(description)

        point = TrajectoryPoint(
            timestamp=datetime.now().isoformat(),
            context_id=context_id,
            embedding=embedding,
            label=label,
            metadata=metadata or {},
        )

        self.trajectories[context_id].append(point)

        logger.debug(f"[TrajectoryTracker] {context_id}: added {label} point")
        return point

    def _text_to_embedding(self, text: str) -> list[float]:
        """Convert text to embedding (simplified hash-based)."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        # Hash to seed
        h = hashlib.md5(text.encode()).hexdigest()
        seed = int(h[:8], 16)

        # Generate pseudo-random vector (thread-safe RNG)
        rng = np.random.default_rng(seed)
        embedding = list(rng.standard_normal(self.embedding_dim) * 0.1)

        self._embedding_cache[text] = embedding
        return embedding

    def get_trajectory(self, context_id: str) -> list[TrajectoryPoint]:
        """Get trajectory for a context."""
        return self.trajectories.get(context_id, [])

    def analyze(self, context_id: str) -> dict:
        """Analyze a trajectory."""
        trajectory = self.trajectories.get(context_id, [])

        if len(trajectory) < 2:
            return {
                "context_id": context_id,
                "points": len(trajectory),
                "analysis": "Insufficient points for analysis",
            }

        # Compute distances between consecutive points
        distances = []
        for i in range(1, len(trajectory)):
            prev_emb = np.array(trajectory[i-1].embedding)
            curr_emb = np.array(trajectory[i].embedding)
            dist = float(np.linalg.norm(curr_emb - prev_emb))
            distances.append(dist)

        # Compute total path length
        total_distance = sum(distances)

        # Compute direct distance (start to end)
        start_emb = np.array(trajectory[0].embedding)
        end_emb = np.array(trajectory[-1].embedding)
        direct_distance = float(np.linalg.norm(end_emb - start_emb))

        # Efficiency = direct / total (1.0 = straight line)
        efficiency = direct_distance / total_distance if total_distance > 0 else 1.0

        # Check for backtracking
        backtracking_count = 0
        for i in range(2, len(trajectory)):
            prev_dir = np.array(trajectory[i-1].embedding) - np.array(trajectory[i-2].embedding)
            curr_dir = np.array(trajectory[i].embedding) - np.array(trajectory[i-1].embedding)
            dot = np.dot(prev_dir, curr_dir)
            if dot < 0:  # Moving backwards
                backtracking_count += 1

        return {
            "context_id": context_id,
            "points": len(trajectory),
            "total_distance": total_distance,
            "direct_distance": direct_distance,
            "efficiency": efficiency,
            "avg_step_size": total_distance / len(distances) if distances else 0,
            "backtracking_events": backtracking_count,
            "labels": [p.label for p in trajectory],
            "interpretation": self._interpret_trajectory(efficiency, backtracking_count),
        }

    def _interpret_trajectory(self, efficiency: float, backtracking: int) -> str:
        """Interpret trajectory characteristics."""
        if efficiency > 0.8 and backtracking == 0:
            return "Direct, efficient solution path"
        elif efficiency > 0.6:
            return "Reasonably direct with minor exploration"
        elif backtracking > 2:
            return "Significant backtracking - struggled to find direction"
        else:
            return "Exploratory path with moderate efficiency"

    def compare_trajectories(
        self,
        context_id_a: str,
        context_id_b: str,
    ) -> dict:
        """Compare two trajectories."""
        analysis_a = self.analyze(context_id_a)
        analysis_b = self.analyze(context_id_b)

        return {
            "context_a": context_id_a,
            "context_b": context_id_b,
            "efficiency_diff": analysis_a.get("efficiency", 0) - analysis_b.get("efficiency", 0),
            "distance_diff": analysis_a.get("total_distance", 0) - analysis_b.get("total_distance", 0),
            "backtracking_diff": analysis_a.get("backtracking_events", 0) - analysis_b.get("backtracking_events", 0),
            "more_efficient": context_id_a if analysis_a.get("efficiency", 0) > analysis_b.get("efficiency", 0) else context_id_b,
        }

    def get_summary(self) -> dict:
        """Get summary of all trajectories."""
        if not self.trajectories:
            return {"trajectories": 0, "total_points": 0}

        analyses = [self.analyze(ctx) for ctx in self.trajectories.keys()]

        return {
            "trajectories": len(self.trajectories),
            "total_points": sum(a["points"] for a in analyses),
            "avg_efficiency": sum(a.get("efficiency", 0) for a in analyses) / len(analyses),
            "avg_backtracking": sum(a.get("backtracking_events", 0) for a in analyses) / len(analyses),
        }


# =============================================================================
# Integrated Progress Monitor
# =============================================================================

class ProgressMonitor:
    """Integrated progress monitor combining all tracking features.

    Brings together:
    - Valence tracking (momentum/emotional state)
    - Surprise detection (unexpected outcomes)
    - Trajectory tracking (solution paths)

    Usage:
        monitor = ProgressMonitor()

        # Set up task
        monitor.start_task("task-1", objective="Implement sorting", difficulty="medium")

        # Record progress
        monitor.record_step("task-1", "intermediate", "Trying quicksort")
        monitor.record_outcome("task-1", success=True, time_ms=5000)

        # Get insights
        insights = monitor.get_insights("task-1")
    """

    def __init__(self):
        self.valence = ValenceTracker()
        self.surprise = SurpriseDetector()
        self.trajectory = TrajectoryTracker()

        # Task metadata
        self.tasks: dict[str, dict] = {}

    def start_task(
        self,
        task_id: str,
        objective: str,
        difficulty: str = "medium",
        expected_success_prob: float | None = None,
        expected_time_ms: int | None = None,
    ) -> None:
        """Start tracking a task."""
        self.tasks[task_id] = {
            "objective": objective,
            "difficulty": difficulty,
            "started_at": datetime.now().isoformat(),
        }

        # Set expectation
        self.surprise.set_expectation(
            task_id,
            success_prob=expected_success_prob,
            expected_time_ms=expected_time_ms,
            difficulty=difficulty,
        )

        # Add start point to trajectory
        self.trajectory.add_point(task_id, "start", objective)

    def record_step(
        self,
        task_id: str,
        label: str,
        description: str,
        metadata: dict | None = None,
    ) -> None:
        """Record an intermediate step."""
        self.trajectory.add_point(task_id, label, description, metadata)

    def record_outcome(
        self,
        task_id: str,
        success: bool,
        time_ms: int | None = None,
        specialist: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Record task outcome and get analysis."""
        task = self.tasks.get(task_id, {})
        difficulty = task.get("difficulty", "medium")

        # Update valence
        valence_state = self.valence.record(task_id, success, difficulty)

        # Check for surprise
        surprise_event = self.surprise.record_outcome(
            task_id, success, time_ms, specialist
        )

        # Add final point to trajectory
        label = "success" if success else "failure"
        desc = description or f"Task {'completed' if success else 'failed'}"
        self.trajectory.add_point(task_id, label, desc)

        return {
            "valence": valence_state.get_signal(),
            "surprise": {
                "detected": surprise_event is not None,
                "category": surprise_event.category if surprise_event else None,
                "score": surprise_event.surprise_score if surprise_event else 0,
            },
            "trajectory": self.trajectory.analyze(task_id),
        }

    def get_insights(self, task_id: str) -> dict:
        """Get comprehensive insights for a task."""
        return {
            "task_id": task_id,
            "task_meta": self.tasks.get(task_id, {}),
            "valence": self.valence.get_valence(task_id).get_signal(),
            "trajectory": self.trajectory.analyze(task_id),
            "should_escalate": self.valence.should_escalate(task_id),
            "should_celebrate": self.valence.should_celebrate(task_id),
        }

    def get_global_insights(self) -> dict:
        """Get global insights across all tasks."""
        return {
            "global_valence": self.valence.get_global_valence().get_signal(),
            "surprise_analysis": self.surprise.analyze_patterns(),
            "trajectory_summary": self.trajectory.get_summary(),
            "total_tasks": len(self.tasks),
        }

    def get_report(self) -> str:
        """Generate human-readable report."""
        global_insights = self.get_global_insights()

        lines = [
            "=" * 60,
            "PROGRESS MONITOR REPORT",
            "=" * 60,
            "",
            f"Total tasks tracked: {global_insights['total_tasks']}",
            "",
            "## Global Valence",
            f"  Value: {global_insights['global_valence']['value']:.2f}",
            f"  Trend: {global_insights['global_valence']['trend']}",
            f"  Interpretation: {global_insights['global_valence']['interpretation']}",
            "",
            "## Surprise Analysis",
            f"  Total surprises: {global_insights['surprise_analysis'].get('total_surprises', 0)}",
            f"  Positive: {global_insights['surprise_analysis'].get('positive_surprises', 0)}",
            f"  Negative: {global_insights['surprise_analysis'].get('negative_surprises', 0)}",
            "",
            "## Trajectory Summary",
            f"  Trajectories: {global_insights['trajectory_summary'].get('trajectories', 0)}",
            f"  Avg efficiency: {global_insights['trajectory_summary'].get('avg_efficiency', 0):.1%}",
            f"  Avg backtracking: {global_insights['trajectory_summary'].get('avg_backtracking', 0):.1f}",
            "=" * 60,
        ]

        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo
    monitor = ProgressMonitor()

    # Simulate a few tasks
    tasks = [
        ("task-1", "Implement binary search", "easy", True),
        ("task-2", "Solve dynamic programming problem", "hard", False),
        ("task-3", "Fix off-by-one error", "medium", True),
        ("task-4", "Optimize algorithm complexity", "hard", True),  # Positive surprise
    ]

    for task_id, objective, difficulty, success in tasks:
        monitor.start_task(task_id, objective, difficulty)
        monitor.record_step(task_id, "intermediate", f"Working on {objective}")
        result = monitor.record_outcome(task_id, success, time_ms=5000)

        print(f"\n{task_id}: {'success' if success else 'failure'}")
        if result["surprise"]["detected"]:
            print(f"  SURPRISE: {result['surprise']['category']} (score={result['surprise']['score']:.2f})")
        print(f"  Valence: {result['valence']['value']:.2f} ({result['valence']['trend']})")

    print("\n" + monitor.get_report())
