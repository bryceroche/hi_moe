"""Meta-learning and continuous learning framework.

Implements:
- hi_moe-cg2: Harness learning loop - the system learns coordination strategies
- hi_moe-0n5: Continuous learning - smooth state transitions, incremental updates

The harness itself can adapt its coordination strategies based on
outcomes, not just the individual specialists.

Key concepts:
- Learning signal: What worked and what didn't
- Strategy adaptation: Modifying routing, prompts, retry logic
- Smooth transitions: Gradual changes vs discrete jumps
- Online updates: Learning during operation, not just offline
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class LearningSignal:
    """A signal about what worked or didn't work."""
    task_id: str
    timestamp: str
    signal_type: str  # "success", "failure", "improvement", "regression"
    source_tier: str
    metric_name: str
    metric_value: float
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "type": self.signal_type,
            "source": self.source_tier,
            "metric": self.metric_name,
            "value": self.metric_value,
        }


@dataclass
class StrategyUpdate:
    """A proposed update to a coordination strategy."""
    strategy_name: str
    parameter: str
    old_value: Any
    new_value: Any
    confidence: float
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class LearningState:
    """Current state of the learning system."""
    # Moving averages for smooth transitions (hi_moe-0n5)
    tier_success_rates: dict[str, float] = field(default_factory=dict)
    specialist_preferences: dict[str, float] = field(default_factory=dict)
    strategy_weights: dict[str, float] = field(default_factory=dict)

    # Learning parameters
    learning_rate: float = 0.1  # How fast to adapt
    momentum: float = 0.9      # Smoothing factor for transitions

    # State history for continuity
    history_window: int = 100
    signal_history: list[LearningSignal] = field(default_factory=list)


class ExponentialMovingAverage:
    """Smooth value updates using EMA (hi_moe-0n5).

    Prevents discrete jumps by smoothly transitioning between values.
    """

    def __init__(self, alpha: float = 0.1, initial: float = 0.5):
        """
        Args:
            alpha: Smoothing factor (0-1). Lower = smoother transitions.
            initial: Initial value
        """
        self.alpha = alpha
        self.value = initial
        self.n_updates = 0

    def update(self, new_value: float) -> float:
        """Update with new value, return smoothed result."""
        if self.n_updates == 0:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        self.n_updates += 1
        return self.value

    def get(self) -> float:
        return self.value


class MetaLearner:
    """Meta-learning system for harness adaptation (hi_moe-cg2).

    Learns coordination strategies from outcomes:
    - Which routing strategies work best
    - When to use which prompt variants
    - Optimal retry configurations
    - Tier-specific adaptations

    Usage:
        learner = MetaLearner()

        # Record outcome
        learner.record_signal(LearningSignal(...))

        # Get strategy recommendations
        strategy = learner.recommend_strategy(task_context)

        # Apply gradual updates
        updates = learner.compute_updates()
        learner.apply_updates(updates)
    """

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

        # Smooth state tracking (hi_moe-0n5)
        self.tier_performance: dict[str, ExponentialMovingAverage] = {}
        self.specialist_performance: dict[str, ExponentialMovingAverage] = {}
        self.strategy_performance: dict[str, ExponentialMovingAverage] = {}

        # Signal accumulation
        self.signals: list[LearningSignal] = []
        self.pending_updates: list[StrategyUpdate] = []

        # Coordination strategies that can be adapted
        self.strategies = {
            "routing": {
                "learned_threshold": 0.6,    # Confidence threshold for learned router
                "heuristic_weight": 0.5,     # Weight for heuristic vs learned
            },
            "retry": {
                "max_fleet_retries": 2,
                "max_dispatcher_retries": 1,
                "escalate_on_failure": True,
            },
            "prompts": {
                "architect_style": "balanced",  # terse, detailed, aggressive, conservative
                "fleet_verbosity": "minimal",
            },
        }

        # Initialize default performances
        for tier in ["monitor", "architect", "dispatcher", "fleet"]:
            self.tier_performance[tier] = ExponentialMovingAverage(learning_rate)

    def record_signal(self, signal: LearningSignal) -> None:
        """Record a learning signal from task execution."""
        self.signals.append(signal)

        # Update relevant performance trackers
        if signal.source_tier in self.tier_performance:
            success_value = 1.0 if signal.signal_type == "success" else 0.0
            self.tier_performance[signal.source_tier].update(success_value)

        # Track specialist performance
        specialist = signal.context.get("specialist")
        if specialist:
            if specialist not in self.specialist_performance:
                self.specialist_performance[specialist] = ExponentialMovingAverage(self.learning_rate)
            success_value = 1.0 if signal.signal_type == "success" else 0.0
            self.specialist_performance[specialist].update(success_value)

        # Track strategy performance
        strategy = signal.context.get("strategy")
        if strategy:
            if strategy not in self.strategy_performance:
                self.strategy_performance[strategy] = ExponentialMovingAverage(self.learning_rate)
            success_value = 1.0 if signal.signal_type == "success" else 0.0
            self.strategy_performance[strategy].update(success_value)

        logger.debug(f"[MetaLearner] Recorded signal: {signal.signal_type} from {signal.source_tier}")

    def compute_updates(self) -> list[StrategyUpdate]:
        """Compute strategy updates based on accumulated signals.

        Uses smooth transitions (hi_moe-0n5) to avoid discrete jumps.
        """
        updates = []

        # Analyze tier performance for retry adjustments
        for tier, perf in self.tier_performance.items():
            if perf.n_updates >= 10:  # Enough data
                success_rate = perf.get()

                # If tier is failing often, increase retries (smooth adjustment)
                if tier == "fleet" and success_rate < 0.7:
                    current = self.strategies["retry"]["max_fleet_retries"]
                    # Smooth increase
                    new_value = current + self.learning_rate * (3 - current) * (0.7 - success_rate)
                    new_value = round(min(max(new_value, 1), 4))  # Clamp 1-4

                    if new_value != current:
                        updates.append(StrategyUpdate(
                            strategy_name="retry",
                            parameter="max_fleet_retries",
                            old_value=current,
                            new_value=new_value,
                            confidence=min(perf.n_updates / 50, 1.0),
                            reasoning=f"Fleet success rate {success_rate:.1%} below target",
                        ))

        # Analyze routing strategy effectiveness
        if self.strategy_performance:
            best_strategy = max(
                self.strategy_performance.items(),
                key=lambda x: x[1].get() if x[1].n_updates >= 5 else 0
            )
            if best_strategy[1].n_updates >= 5:
                best_name, best_perf = best_strategy

                # Adjust heuristic weight based on which strategy works better
                if "learned" in best_name and best_perf.get() > 0.7:
                    current = self.strategies["routing"]["heuristic_weight"]
                    # Decrease heuristic weight (favor learned)
                    new_value = current - self.learning_rate * 0.1
                    new_value = max(new_value, 0.2)

                    if abs(new_value - current) > 0.01:
                        updates.append(StrategyUpdate(
                            strategy_name="routing",
                            parameter="heuristic_weight",
                            old_value=current,
                            new_value=round(new_value, 2),
                            confidence=best_perf.get(),
                            reasoning=f"Learned routing performing well ({best_perf.get():.1%})",
                        ))

        # Analyze prompt style effectiveness
        architect_signals = [s for s in self.signals[-50:]
                           if s.source_tier == "architect"]
        if len(architect_signals) >= 10:
            success_rate = sum(1 for s in architect_signals
                              if s.signal_type == "success") / len(architect_signals)

            if success_rate < 0.6:
                current = self.strategies["prompts"]["architect_style"]
                # Try a different style
                styles = ["terse", "detailed", "aggressive", "conservative"]
                current_idx = styles.index(current) if current in styles else 0
                new_style = styles[(current_idx + 1) % len(styles)]

                updates.append(StrategyUpdate(
                    strategy_name="prompts",
                    parameter="architect_style",
                    old_value=current,
                    new_value=new_style,
                    confidence=0.5,  # Exploratory
                    reasoning=f"Architect success rate low ({success_rate:.1%}), trying {new_style}",
                ))

        self.pending_updates = updates
        return updates

    def apply_updates(self, updates: list[StrategyUpdate] | None = None) -> None:
        """Apply strategy updates with smooth transitions."""
        updates = updates or self.pending_updates

        for update in updates:
            if update.strategy_name in self.strategies:
                strategy = self.strategies[update.strategy_name]
                if update.parameter in strategy:
                    strategy[update.parameter] = update.new_value
                    logger.info(
                        f"[MetaLearner] Updated {update.strategy_name}.{update.parameter}: "
                        f"{update.old_value} -> {update.new_value} "
                        f"(confidence: {update.confidence:.1%})"
                    )

        self.pending_updates = []

    def recommend_strategy(self, task_context: dict) -> dict:
        """Recommend coordination strategy for a task.

        Returns strategy parameters tuned by learning.
        """
        recommendation = {
            "routing": dict(self.strategies["routing"]),
            "retry": dict(self.strategies["retry"]),
            "prompts": dict(self.strategies["prompts"]),
        }

        # Adjust based on task characteristics
        objective = task_context.get("objective", "").lower()

        # If task looks complex, increase patience
        if any(kw in objective for kw in ["complex", "multi-step", "difficult"]):
            recommendation["retry"]["max_fleet_retries"] = min(
                recommendation["retry"]["max_fleet_retries"] + 1, 4
            )

        # If we have good data on specialist for this type, boost confidence
        for specialist, perf in self.specialist_performance.items():
            if specialist in objective and perf.n_updates >= 5:
                if perf.get() > 0.8:
                    recommendation["routing"]["learned_threshold"] = max(
                        recommendation["routing"]["learned_threshold"] - 0.1, 0.4
                    )

        return recommendation

    def get_state(self) -> dict:
        """Get current learning state."""
        return {
            "tier_performance": {
                tier: {"value": perf.get(), "samples": perf.n_updates}
                for tier, perf in self.tier_performance.items()
            },
            "specialist_performance": {
                spec: {"value": perf.get(), "samples": perf.n_updates}
                for spec, perf in self.specialist_performance.items()
            },
            "strategy_performance": {
                strat: {"value": perf.get(), "samples": perf.n_updates}
                for strat, perf in self.strategy_performance.items()
            },
            "current_strategies": self.strategies,
            "total_signals": len(self.signals),
        }

    def get_report(self) -> str:
        """Generate learning status report."""
        lines = [
            "=" * 60,
            "META-LEARNING STATUS",
            "=" * 60,
            "",
            f"Total signals: {len(self.signals)}",
            f"Learning rate: {self.learning_rate}",
            "",
            "## Tier Performance (EMA)",
        ]

        for tier, perf in self.tier_performance.items():
            if perf.n_updates > 0:
                lines.append(f"  {tier}: {perf.get():.1%} ({perf.n_updates} samples)")

        lines.append("\n## Specialist Performance")
        for spec, perf in self.specialist_performance.items():
            if perf.n_updates > 0:
                lines.append(f"  {spec}: {perf.get():.1%} ({perf.n_updates} samples)")

        lines.append("\n## Current Strategies")
        for strategy_name, params in self.strategies.items():
            lines.append(f"  {strategy_name}:")
            for param, value in params.items():
                lines.append(f"    {param}: {value}")

        if self.pending_updates:
            lines.append("\n## Pending Updates")
            for update in self.pending_updates:
                lines.append(
                    f"  {update.strategy_name}.{update.parameter}: "
                    f"{update.old_value} -> {update.new_value}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save learning state."""
        data = {
            "strategies": self.strategies,
            "tier_performance": {
                tier: {"value": perf.get(), "n_updates": perf.n_updates, "alpha": perf.alpha}
                for tier, perf in self.tier_performance.items()
            },
            "specialist_performance": {
                spec: {"value": perf.get(), "n_updates": perf.n_updates, "alpha": perf.alpha}
                for spec, perf in self.specialist_performance.items()
            },
            "signals_count": len(self.signals),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[MetaLearner] Saved state to {path}")


# Continuous learning coordinator
class ContinuousLearner:
    """Coordinates continuous learning across the system (hi_moe-0n5).

    Ensures smooth state transitions and incremental updates rather
    than discrete jumps.
    """

    def __init__(self, meta_learner: MetaLearner):
        self.meta = meta_learner
        self.update_interval = 10  # Updates after N signals
        self.signals_since_update = 0

    def on_task_complete(
        self,
        task_id: str,
        tier: str,
        success: bool,
        specialist: str | None = None,
        strategy: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        """Called when a task completes at any tier."""
        signal = LearningSignal(
            task_id=task_id,
            timestamp=datetime.now().isoformat(),
            signal_type="success" if success else "failure",
            source_tier=tier,
            metric_name="completion",
            metric_value=1.0 if success else 0.0,
            context={
                "specialist": specialist,
                "strategy": strategy,
                **(metrics or {}),
            },
        )

        self.meta.record_signal(signal)
        self.signals_since_update += 1

        # Periodically compute and apply updates
        if self.signals_since_update >= self.update_interval:
            updates = self.meta.compute_updates()
            if updates:
                logger.info(f"[ContinuousLearner] Applying {len(updates)} strategy updates")
                self.meta.apply_updates(updates)
            self.signals_since_update = 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    learner = MetaLearner(learning_rate=0.1)
    continuous = ContinuousLearner(learner)

    # Simulate task completions
    tasks = [
        ("task-1", "fleet", True, "python", "python_direct"),
        ("task-2", "fleet", True, "python", "learned"),
        ("task-3", "fleet", False, "python", "learned"),
        ("task-4", "fleet", True, "math", "math_first"),
        ("task-5", "fleet", True, "python", "learned"),
        ("task-6", "architect", True, None, None),
        ("task-7", "architect", False, None, None),
        ("task-8", "fleet", True, "python", "learned"),
        ("task-9", "fleet", True, "python", "learned"),
        ("task-10", "dispatcher", True, None, "structured_plan"),
        ("task-11", "fleet", True, "algorithms", "learned"),
        ("task-12", "fleet", False, "debugging", "heuristic"),
    ]

    for task_id, tier, success, specialist, strategy in tasks:
        continuous.on_task_complete(task_id, tier, success, specialist, strategy)

    print(learner.get_report())

    # Get recommendation for new task
    print("\n--- Recommendation for complex task ---")
    rec = learner.recommend_strategy({"objective": "implement complex algorithm"})
    print(json.dumps(rec, indent=2))
