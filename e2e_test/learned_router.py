"""Learned routing model for specialist selection (hi_moe-bh3).

Improves upon heuristic routing by learning from task outcomes.
Uses a lightweight feature-based model that can be updated online.

Architecture:
- Feature extraction from task text (keywords, patterns)
- Per-specialist success probability model
- Online learning from outcomes
- Fallback to heuristic when confidence is low
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .dispatcher_schema import VALID_SPECIALISTS

logger = logging.getLogger(__name__)

# Feature keywords by category
FEATURE_KEYWORDS = {
    "algorithm": ["algorithm", "optimal", "complexity", "time", "space", "o(n)", "o(1)"],
    "math": ["math", "calculate", "formula", "equation", "proof", "theorem"],
    "data_structure": ["array", "hash", "tree", "graph", "list", "stack", "queue", "heap"],
    "string": ["string", "substring", "parse", "regex", "pattern", "match"],
    "search": ["search", "find", "lookup", "binary", "linear"],
    "sort": ["sort", "order", "rank", "compare"],
    "dynamic": ["dynamic", "memoization", "dp", "cache", "subproblem"],
    "implementation": ["implement", "write", "create", "build", "code", "function"],
    "debug": ["debug", "fix", "error", "bug", "issue", "broken"],
    "refactor": ["refactor", "clean", "improve", "optimize", "simplify"],
}


@dataclass
class RoutingFeatures:
    """Features extracted from a task for routing decisions."""

    task_text: str
    keyword_counts: dict[str, int] = field(default_factory=dict)
    text_length: int = 0
    has_code: bool = False
    question_type: str = "implementation"  # implementation, analysis, debug

    @classmethod
    def from_task(cls, objective: str, context: dict | None = None) -> "RoutingFeatures":
        """Extract features from task text."""
        text = objective.lower()
        if context:
            plan = context.get("plan", "")
            if plan:
                text = f"{text} {plan.lower()}"

        # Count keyword matches
        keyword_counts = {}
        for category, keywords in FEATURE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > 0:
                keyword_counts[category] = count

        # Detect code blocks
        has_code = "```" in text or "def " in text or "class " in text

        # Classify question type
        if any(kw in text for kw in ["debug", "fix", "error", "bug"]):
            question_type = "debug"
        elif any(kw in text for kw in ["analyze", "explain", "complexity", "proof"]):
            question_type = "analysis"
        else:
            question_type = "implementation"

        return cls(
            task_text=text[:500],  # Truncate for storage
            keyword_counts=keyword_counts,
            text_length=len(text),
            has_code=has_code,
            question_type=question_type,
        )

    def to_vector(self) -> dict[str, float]:
        """Convert features to numeric vector for model."""
        vec = {
            "text_length_norm": min(self.text_length / 1000, 1.0),
            "has_code": 1.0 if self.has_code else 0.0,
            "type_implementation": 1.0 if self.question_type == "implementation" else 0.0,
            "type_analysis": 1.0 if self.question_type == "analysis" else 0.0,
            "type_debug": 1.0 if self.question_type == "debug" else 0.0,
        }
        # Add keyword category counts
        for category in FEATURE_KEYWORDS:
            vec[f"kw_{category}"] = float(self.keyword_counts.get(category, 0))
        return vec


@dataclass
class RoutingRecord:
    """Record of a routing decision and its outcome."""

    timestamp: str
    task_id: str
    features: RoutingFeatures
    specialist: str
    success: bool
    execution_time_ms: float = 0
    error: str | None = None


class SpecialistModel:
    """Simple logistic regression model for one specialist.

    Predicts P(success | features) for this specialist.
    Uses online gradient descent for learning.
    """

    def __init__(self, name: str, learning_rate: float = 0.1):
        self.name = name
        self.learning_rate = learning_rate
        # Weights for each feature (initialized to small random values)
        self.weights: dict[str, float] = defaultdict(float)
        self.bias: float = 0.0
        # Training stats
        self.n_samples: int = 0
        self.n_successes: int = 0

    def predict(self, features: dict[str, float]) -> float:
        """Predict probability of success given features."""
        logit = self.bias
        for feat, val in features.items():
            logit += self.weights[feat] * val
        # Sigmoid
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, logit))))

    def update(self, features: dict[str, float], success: bool) -> None:
        """Update model with new outcome (online learning)."""
        self.n_samples += 1
        if success:
            self.n_successes += 1

        # Gradient descent step
        pred = self.predict(features)
        target = 1.0 if success else 0.0
        error = target - pred

        # Update weights
        self.bias += self.learning_rate * error
        for feat, val in features.items():
            self.weights[feat] += self.learning_rate * error * val

    @property
    def success_rate(self) -> float:
        """Empirical success rate."""
        if self.n_samples == 0:
            return 0.5
        return self.n_successes / self.n_samples

    def to_dict(self) -> dict:
        """Serialize model state."""
        return {
            "name": self.name,
            "weights": dict(self.weights),
            "bias": self.bias,
            "n_samples": self.n_samples,
            "n_successes": self.n_successes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpecialistModel":
        """Deserialize model state."""
        model = cls(data["name"])
        model.weights = defaultdict(float, data.get("weights", {}))
        model.bias = data.get("bias", 0.0)
        model.n_samples = data.get("n_samples", 0)
        model.n_successes = data.get("n_successes", 0)
        return model


class LearnedRouter:
    """Learned routing model that improves from outcomes (hi_moe-bh3).

    Maintains per-specialist success prediction models and updates
    them online as tasks are completed.
    """

    def __init__(
        self,
        specialists: list[str] | None = None,
        confidence_threshold: float = 0.6,
        min_samples: int = 5,
    ):
        """Initialize learned router.

        Args:
            specialists: List of specialist names (uses VALID_SPECIALISTS if None)
            confidence_threshold: Min confidence to override heuristic (0-1)
            min_samples: Min samples before using learned predictions
        """
        self.specialists = specialists or list(VALID_SPECIALISTS)
        self.confidence_threshold = confidence_threshold
        self.min_samples = min_samples

        # Per-specialist models
        self.models: dict[str, SpecialistModel] = {
            name: SpecialistModel(name) for name in self.specialists
        }

        # Training history
        self.history: list[RoutingRecord] = []

    def predict(
        self,
        objective: str,
        context: dict | None = None,
        exclude: list[str] | None = None,
    ) -> tuple[str | None, dict[str, float], str]:
        """Predict best specialist for a task.

        Args:
            objective: Task objective text
            context: Optional task context
            exclude: Specialists to exclude (already tried)

        Returns:
            Tuple of (specialist_or_None, confidence_scores, reasoning)
            Returns None if confidence is too low (fall back to heuristic)
        """
        exclude = exclude or []
        features = RoutingFeatures.from_task(objective, context)
        feature_vec = features.to_vector()

        # Get predictions from all models
        scores: dict[str, float] = {}
        for name, model in self.models.items():
            if name in exclude:
                continue
            scores[name] = model.predict(feature_vec)

        if not scores:
            return None, {}, "All specialists excluded"

        # Find best specialist
        best_specialist = max(scores, key=lambda k: scores[k])
        best_score = scores[best_specialist]

        # Check if we have enough samples and confidence
        model = self.models[best_specialist]
        has_enough_samples = model.n_samples >= self.min_samples
        has_confidence = best_score >= self.confidence_threshold

        if not has_enough_samples:
            reasoning = f"Insufficient samples ({model.n_samples}/{self.min_samples})"
            return None, scores, reasoning

        if not has_confidence:
            reasoning = f"Low confidence ({best_score:.2%} < {self.confidence_threshold:.0%})"
            return None, scores, reasoning

        # Build reasoning
        top_features = sorted(
            [(f, feature_vec[f] * model.weights.get(f, 0))
             for f in feature_vec if feature_vec[f] > 0],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:3]
        feature_reasons = ", ".join(f"{f}={v:.2f}" for f, v in top_features if v != 0)
        reasoning = f"Learned: {best_specialist} ({best_score:.1%} confidence, {feature_reasons})"

        return best_specialist, scores, reasoning

    def record_outcome(
        self,
        task_id: str,
        objective: str,
        context: dict | None,
        specialist: str,
        success: bool,
        execution_time_ms: float = 0,
        error: str | None = None,
    ) -> None:
        """Record routing outcome and update models.

        Args:
            task_id: Unique task identifier
            objective: Task objective text
            context: Task context (optional)
            specialist: Specialist that was used
            success: Whether the task succeeded
            execution_time_ms: Execution time
            error: Error message if failed
        """
        features = RoutingFeatures.from_task(objective, context)

        # Record to history
        record = RoutingRecord(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            features=features,
            specialist=specialist,
            success=success,
            execution_time_ms=execution_time_ms,
            error=error,
        )
        self.history.append(record)

        # Update model
        if specialist in self.models:
            feature_vec = features.to_vector()
            self.models[specialist].update(feature_vec, success)
            logger.info(
                f"[LearnedRouter] Updated {specialist}: "
                f"{self.models[specialist].n_samples} samples, "
                f"{self.models[specialist].success_rate:.1%} success rate"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        return {
            "total_samples": sum(m.n_samples for m in self.models.values()),
            "specialists": {
                name: {
                    "samples": model.n_samples,
                    "success_rate": f"{model.success_rate:.1%}",
                    "top_weights": sorted(
                        [(k, v) for k, v in model.weights.items() if abs(v) > 0.1],
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:5],
                }
                for name, model in self.models.items()
            },
        }

    def save(self, path: Path) -> None:
        """Save router state to file."""
        data = {
            "confidence_threshold": self.confidence_threshold,
            "min_samples": self.min_samples,
            "models": {name: model.to_dict() for name, model in self.models.items()},
            "history_count": len(self.history),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[LearnedRouter] Saved state to {path}")

    @classmethod
    def load(cls, path: Path) -> "LearnedRouter":
        """Load router state from file."""
        with open(path) as f:
            data = json.load(f)

        router = cls(
            confidence_threshold=data.get("confidence_threshold", 0.6),
            min_samples=data.get("min_samples", 5),
        )

        # Load models
        for name, model_data in data.get("models", {}).items():
            if name in router.models:
                router.models[name] = SpecialistModel.from_dict(model_data)

        logger.info(f"[LearnedRouter] Loaded state from {path}")
        return router


class HybridRouter:
    """Combined learned + heuristic router (hi_moe-bh3).

    Uses learned predictions when confident, falls back to heuristics.
    """

    def __init__(
        self,
        learned_router: LearnedRouter | None = None,
        heuristic_fn: Any = None,
    ):
        """Initialize hybrid router.

        Args:
            learned_router: Learned routing model (created if None)
            heuristic_fn: Fallback heuristic function (task -> specialist)
        """
        self.learned = learned_router or LearnedRouter()
        self.heuristic_fn = heuristic_fn

    def route(
        self,
        objective: str,
        context: dict | None = None,
        exclude: list[str] | None = None,
    ) -> tuple[str, str, dict]:
        """Route task to specialist.

        Args:
            objective: Task objective
            context: Optional context
            exclude: Specialists to exclude

        Returns:
            Tuple of (specialist, routing_method, metadata)
        """
        exclude = exclude or []

        # Try learned router first
        specialist, scores, reasoning = self.learned.predict(objective, context, exclude)

        if specialist is not None:
            return specialist, "learned", {
                "reasoning": reasoning,
                "scores": scores,
            }

        # Fall back to heuristic
        if self.heuristic_fn:
            specialist = self.heuristic_fn(objective, context, exclude)
            return specialist, "heuristic", {
                "reasoning": f"Fallback: {reasoning}",
                "scores": scores,
            }

        # Default fallback
        for s in ["python", "general"]:
            if s not in exclude:
                return s, "default", {"reasoning": "No heuristic, using default"}

        return "general", "default", {"reasoning": "All options exhausted"}

    def record_outcome(
        self,
        task_id: str,
        objective: str,
        context: dict | None,
        specialist: str,
        success: bool,
        **kwargs,
    ) -> None:
        """Record outcome to update learned router."""
        self.learned.record_outcome(
            task_id=task_id,
            objective=objective,
            context=context,
            specialist=specialist,
            success=success,
            **kwargs,
        )


# Integration helper for RoutingDispatcher
def create_hybrid_select_specialist(
    router: HybridRouter,
) -> callable:
    """Create a _select_specialist replacement for RoutingDispatcher.

    Returns a function compatible with RoutingDispatcher._select_specialist.
    """
    def select_specialist(
        task: Any,
        exclude: list[str] | None = None,
    ) -> tuple[str, str, list[str]]:
        """Select specialist using hybrid routing.

        Returns:
            Tuple of (specialist, routing_strategy, routing_signals)
        """
        objective = task.objective
        context = task.context if hasattr(task, "context") else None

        specialist, method, metadata = router.route(objective, context, exclude)

        # Convert to dispatcher format
        routing_strategy = "learned_first" if method == "learned" else "heuristic_fallback"
        routing_signals = [f"method:{method}", f"reasoning:{metadata.get('reasoning', '')}"]

        if "scores" in metadata:
            for name, score in metadata["scores"].items():
                routing_signals.append(f"score:{name}={score:.2f}")

        return specialist, routing_strategy, routing_signals

    return select_specialist


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)

    router = LearnedRouter()

    # Simulate some training data
    training_data = [
        ("Implement two sum using hash map", "python", True),
        ("Analyze time complexity of algorithm", "math", True),
        ("Write a function to sort array", "python", True),
        ("Debug this recursive function", "python", False),
        ("Prove the algorithm is optimal", "math", True),
        ("Implement binary search", "python", True),
        ("Calculate the mathematical formula", "math", True),
        ("Fix the bug in this code", "python", True),
    ]

    # Train
    for i, (objective, specialist, success) in enumerate(training_data):
        router.record_outcome(
            task_id=f"train-{i}",
            objective=objective,
            context=None,
            specialist=specialist,
            success=success,
        )

    # Test predictions
    print("\n--- Predictions ---")
    test_cases = [
        "Implement a stack data structure",
        "Prove that the algorithm runs in O(n log n)",
        "Debug the memory leak in the function",
    ]

    for test in test_cases:
        specialist, scores, reasoning = router.predict(test)
        print(f"\nTask: {test}")
        print(f"  Prediction: {specialist}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Scores: {scores}")

    print("\n--- Stats ---")
    print(json.dumps(router.get_stats(), indent=2))
