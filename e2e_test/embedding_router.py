"""Embedding-based routing for specialist selection (hi_moe-awf).

Uses semantic embeddings to route tasks to specialists based on
similarity rather than keyword matching. This captures meaning
and handles novel phrasings better than heuristic routing.

Architecture:
- Pre-compute embeddings for specialist capability descriptions
- Embed incoming task objectives
- Route based on cosine similarity
- Online learning: accumulate prototype embeddings from successes
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dispatcher_schema import VALID_SPECIALISTS

logger = logging.getLogger(__name__)

# Specialist capability descriptions for embedding
# These describe the types of problems each specialist handles
SPECIALIST_DESCRIPTIONS = {
    "python": [
        # Data structure problems
        "given an array of integers find two numbers that add up to target",
        "implement a hash map to store and lookup values efficiently",
        "write a function using stack to match parentheses brackets",
        "merge overlapping intervals in a sorted array",
        "implement binary search to find element in sorted array",
        # String problems
        "find the longest palindromic substring in a string",
        "implement string matching and pattern recognition",
        # General implementation
        "implement a data structure with push pop and peek operations",
        "write Python code to solve the programming problem",
        "create efficient implementation using arrays and dictionaries",
    ],
    "math": [
        # Complexity analysis
        "analyze time complexity and prove algorithm is optimal",
        "prove the algorithm runs in O(n log n) time",
        "derive mathematical bounds for the solution",
        # Mathematical problems
        "calculate using mathematical formulas and equations",
        "apply number theory combinatorics or probability",
        "solve using dynamic programming recurrence relation",
        "prove correctness of the algorithm mathematically",
    ],
    "general": [
        "general problem solving and task completion",
        "handle miscellaneous tasks and questions",
        "provide explanations and documentation",
        "assist with various coding and analysis tasks",
    ],
    # Extended specialists (optional, for future use)
    "algorithms": [
        "design efficient algorithm with optimal complexity",
        "implement dynamic programming or greedy solution",
        "apply graph algorithms like BFS DFS shortest path",
        "use divide and conquer or binary search approach",
        "optimize time and space complexity",
        "solve using sliding window or two pointer technique",
    ],
    "debugging": [
        "find and fix bugs in existing code",
        "debug runtime errors and exceptions",
        "identify logical errors in implementation",
        "trace execution flow to find issues",
    ],
}


@dataclass
class EmbeddingCache:
    """Cache for embeddings to avoid recomputation."""

    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    model_name: str = ""

    def get(self, text: str) -> np.ndarray | None:
        """Get cached embedding if available."""
        return self.embeddings.get(text)

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""
        self.embeddings[text] = embedding

    def clear(self) -> None:
        """Clear the cache."""
        self.embeddings.clear()


@dataclass
class RoutingOutcome:
    """Record of a routing decision and outcome for learning."""

    timestamp: str
    task_id: str
    objective: str
    specialist: str
    success: bool
    similarity_score: float
    embedding: np.ndarray | None = None


class EmbeddingRouter:
    """Embedding-based routing for specialist selection (hi_moe-awf).

    Uses sentence embeddings to match task objectives to specialist
    capabilities based on semantic similarity.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.25,
        specialists: list[str] | None = None,
        cache_embeddings: bool = True,
    ):
        """Initialize embedding router.

        Args:
            model_name: Sentence-transformers model name
            confidence_threshold: Min similarity to route (0-1)
            specialists: List of specialist names to use
            cache_embeddings: Whether to cache embeddings
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.specialists = specialists or list(VALID_SPECIALISTS)
        self.cache_embeddings = cache_embeddings

        # Lazy-load model to avoid import cost at module level
        self._model = None
        self._embedding_dim = None

        # Pre-computed specialist embeddings (centroid of description embeddings)
        self._specialist_embeddings: dict[str, np.ndarray] = {}

        # Prototype embeddings from successful tasks (online learning)
        self._prototypes: dict[str, list[np.ndarray]] = {
            s: [] for s in self.specialists
        }
        self._max_prototypes = 20  # Max per specialist

        # Embedding cache
        self._cache = EmbeddingCache(model_name=model_name) if cache_embeddings else None

        # Outcome history for analysis
        self.history: list[RoutingOutcome] = []

    def _load_model(self) -> None:
        """Lazy-load the sentence transformer model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[EmbeddingRouter] Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"[EmbeddingRouter] Model loaded, dim={self._embedding_dim}")

            # Pre-compute specialist embeddings
            self._compute_specialist_embeddings()

        except ImportError:
            logger.warning(
                "[EmbeddingRouter] sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def _compute_specialist_embeddings(self) -> None:
        """Compute centroid embeddings for each specialist."""
        if self._model is None:
            return

        for specialist in self.specialists:
            descriptions = SPECIALIST_DESCRIPTIONS.get(specialist, [])
            if not descriptions:
                # Use specialist name as fallback
                descriptions = [f"{specialist} specialist for coding tasks"]

            # Embed all descriptions
            embeddings = self._model.encode(descriptions, convert_to_numpy=True)

            # Compute centroid (mean embedding)
            centroid = np.mean(embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize

            self._specialist_embeddings[specialist] = centroid
            logger.debug(f"[EmbeddingRouter] Computed embedding for {specialist}")

    def _embed(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        self._load_model()

        # Check cache
        if self._cache:
            cached = self._cache.get(text)
            if cached is not None:
                return cached

        # Compute embedding
        embedding = self._model.encode(text, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize

        # Cache result
        if self._cache:
            self._cache.set(text, embedding)

        return embedding

    def _compute_similarity(
        self, task_embedding: np.ndarray, specialist: str
    ) -> float:
        """Compute similarity between task and specialist.

        Uses weighted combination of:
        - Similarity to specialist description centroid
        - Similarity to successful task prototypes (if any)
        """
        base_embedding = self._specialist_embeddings.get(specialist)
        if base_embedding is None:
            return 0.0

        # Base similarity to specialist description
        base_sim = float(np.dot(task_embedding, base_embedding))

        # Prototype similarity (from successful tasks)
        prototypes = self._prototypes.get(specialist, [])
        if prototypes:
            proto_sims = [float(np.dot(task_embedding, p)) for p in prototypes]
            proto_sim = max(proto_sims)  # Best match among prototypes

            # Blend: 60% base, 40% prototype (if prototypes exist)
            similarity = 0.6 * base_sim + 0.4 * proto_sim
        else:
            similarity = base_sim

        return similarity

    def predict(
        self,
        objective: str,
        context: dict | None = None,
        exclude: list[str] | None = None,
    ) -> tuple[str | None, dict[str, float], str]:
        """Predict best specialist for a task.

        Args:
            objective: Task objective text
            context: Optional task context (plan, etc.)
            exclude: Specialists to exclude

        Returns:
            Tuple of (specialist_or_None, similarity_scores, reasoning)
            Returns None if confidence is too low
        """
        exclude = exclude or []

        # Build full text for embedding
        text = objective
        if context and context.get("plan"):
            text = f"{objective}\n{context['plan']}"

        # Get task embedding
        try:
            task_embedding = self._embed(text)
        except Exception as e:
            logger.error(f"[EmbeddingRouter] Embedding failed: {e}")
            return None, {}, f"Embedding error: {e}"

        # Compute similarities
        scores: dict[str, float] = {}
        for specialist in self.specialists:
            if specialist in exclude:
                continue
            scores[specialist] = self._compute_similarity(task_embedding, specialist)

        if not scores:
            return None, {}, "All specialists excluded"

        # Find best specialist
        best_specialist = max(scores, key=lambda k: scores[k])
        best_score = scores[best_specialist]

        # Check confidence threshold
        if best_score < self.confidence_threshold:
            reasoning = f"Low similarity ({best_score:.2f} < {self.confidence_threshold})"
            return None, scores, reasoning

        # Build reasoning
        score_summary = ", ".join(
            f"{s}={scores[s]:.2f}" for s in sorted(scores, key=lambda k: -scores[k])[:3]
        )
        n_prototypes = len(self._prototypes.get(best_specialist, []))
        reasoning = f"Embedding: {best_specialist} (sim={best_score:.2f}, protos={n_prototypes}); {score_summary}"

        return best_specialist, scores, reasoning

    def predict_weighted(
        self,
        objective: str,
        context: dict | None = None,
        exclude: list[str] | None = None,
        min_weight: float = 0.1,
    ) -> tuple[dict[str, float], str]:
        """Predict blend weights for all specialists (hi_moe-zrn).

        Returns normalized similarity scores as blend weights.
        Weights sum to 1.0 and can be used for:
        - LoRA weight blending: merged = w1*python + w2*math
        - Probabilistic routing: P(python) = w1

        Args:
            objective: Task objective text
            context: Optional task context
            exclude: Specialists to exclude
            min_weight: Minimum weight to include (filters noise)

        Returns:
            Tuple of (weights_dict, reasoning)
            weights_dict maps specialist -> weight (0-1), sums to 1.0
        """
        exclude = exclude or []

        # Get raw prediction and scores
        _, scores, base_reasoning = self.predict(objective, context, exclude)

        if not scores:
            return {}, "No specialists available"

        # Filter out very low scores (noise)
        filtered = {s: score for s, score in scores.items() if score >= min_weight}
        if not filtered:
            # If all filtered out, keep the best one
            best = max(scores, key=lambda k: scores[k])
            filtered = {best: scores[best]}

        # Normalize to sum to 1.0
        total = sum(filtered.values())
        if total <= 0:
            # Uniform fallback
            n = len(filtered)
            weights = {s: 1.0 / n for s in filtered}
        else:
            weights = {s: score / total for s, score in filtered.items()}

        # Sort by weight descending
        weights = dict(sorted(weights.items(), key=lambda x: -x[1]))

        # Build reasoning
        weight_strs = [f"{s}={w:.0%}" for s, w in weights.items()]
        reasoning = f"Blend weights: {', '.join(weight_strs)}"

        return weights, reasoning

    def select_probabilistic(
        self,
        objective: str,
        context: dict | None = None,
        exclude: list[str] | None = None,
    ) -> tuple[str, dict[str, float], str]:
        """Select specialist probabilistically based on similarity weights (hi_moe-zrn).

        Instead of always picking the highest-scoring specialist,
        samples proportionally to similarity scores. Over many requests,
        this achieves an effective "blend" across specialists.

        Args:
            objective: Task objective text
            context: Optional task context
            exclude: Specialists to exclude

        Returns:
            Tuple of (selected_specialist, weights, reasoning)
        """
        weights, base_reasoning = self.predict_weighted(objective, context, exclude)

        if not weights:
            return "python", {}, "No weights, defaulting to python"

        # Probabilistic selection based on weights
        specialists = list(weights.keys())
        probs = list(weights.values())

        # Use numpy for weighted random choice
        selected = np.random.choice(specialists, p=probs)

        reasoning = f"Probabilistic selection: {selected} (from {base_reasoning})"

        return selected, weights, reasoning

    def recommend_blending(
        self,
        objective: str,
        context: dict | None = None,
        blend_threshold: float = 0.3,
    ) -> tuple[bool, dict[str, float], str]:
        """Check if blending is recommended for a task (hi_moe-zrn).

        Blending is recommended when multiple specialists have similar
        scores, indicating the task spans multiple domains.

        Args:
            objective: Task objective text
            context: Optional task context
            blend_threshold: If second-best is within this of best, blend

        Returns:
            Tuple of (should_blend, weights, reasoning)
        """
        weights, _ = self.predict_weighted(objective, context)

        if len(weights) < 2:
            return False, weights, "Only one specialist available"

        # Check if top two are close
        sorted_weights = sorted(weights.values(), reverse=True)
        best = sorted_weights[0]
        second = sorted_weights[1]

        # If second-best is at least (1 - threshold) of best, recommend blending
        # e.g., if best=0.55 and second=0.35, ratio=0.64 > 0.7 (threshold=0.3)
        ratio = second / best if best > 0 else 0
        should_blend = ratio >= (1 - blend_threshold)

        if should_blend:
            reasoning = f"Blending recommended: top scores are close ({best:.0%} vs {second:.0%})"
        else:
            best_specialist = max(weights, key=lambda k: weights[k])
            reasoning = f"Single specialist preferred: {best_specialist} dominates ({best:.0%} vs {second:.0%})"

        return should_blend, weights, reasoning

    def record_outcome(
        self,
        task_id: str,
        objective: str,
        context: dict | None,
        specialist: str,
        success: bool,
        similarity_score: float = 0.0,
    ) -> None:
        """Record routing outcome and update prototypes.

        On success, adds task embedding to specialist's prototypes
        for improved future routing (online learning).
        """
        # Build text and get embedding
        text = objective
        if context and context.get("plan"):
            text = f"{objective}\n{context['plan']}"

        try:
            embedding = self._embed(text)
        except Exception:
            embedding = None

        # Record to history
        record = RoutingOutcome(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            objective=objective[:200],
            specialist=specialist,
            success=success,
            similarity_score=similarity_score,
            embedding=embedding,
        )
        self.history.append(record)

        # Update prototypes on success
        if success and embedding is not None and specialist in self._prototypes:
            self._prototypes[specialist].append(embedding)

            # Trim to max prototypes (keep most recent)
            if len(self._prototypes[specialist]) > self._max_prototypes:
                self._prototypes[specialist] = self._prototypes[specialist][-self._max_prototypes:]

            logger.info(
                f"[EmbeddingRouter] Added prototype for {specialist} "
                f"(total: {len(self._prototypes[specialist])})"
            )

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics."""
        stats = {
            "model": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "specialists": {},
            "total_outcomes": len(self.history),
        }

        for specialist in self.specialists:
            outcomes = [h for h in self.history if h.specialist == specialist]
            successes = sum(1 for h in outcomes if h.success)
            stats["specialists"][specialist] = {
                "prototype_count": len(self._prototypes.get(specialist, [])),
                "total_routed": len(outcomes),
                "success_rate": f"{successes/len(outcomes):.1%}" if outcomes else "N/A",
            }

        return stats

    def save(self, path: Path) -> None:
        """Save router state (prototypes and history)."""
        # Convert numpy arrays to lists for JSON
        prototypes_json = {}
        for specialist, protos in self._prototypes.items():
            prototypes_json[specialist] = [p.tolist() for p in protos]

        data = {
            "model_name": self.model_name,
            "confidence_threshold": self.confidence_threshold,
            "specialists": self.specialists,
            "prototypes": prototypes_json,
            "history_count": len(self.history),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[EmbeddingRouter] Saved state to {path}")

    @classmethod
    def load(cls, path: Path) -> "EmbeddingRouter":
        """Load router state from file."""
        with open(path) as f:
            data = json.load(f)

        router = cls(
            model_name=data.get("model_name", "all-MiniLM-L6-v2"),
            confidence_threshold=data.get("confidence_threshold", 0.35),
            specialists=data.get("specialists"),
        )

        # Load prototypes (will be converted to numpy after model loads)
        prototypes_json = data.get("prototypes", {})
        for specialist, protos in prototypes_json.items():
            if specialist in router._prototypes:
                router._prototypes[specialist] = [np.array(p) for p in protos]

        logger.info(f"[EmbeddingRouter] Loaded state from {path}")
        return router


class HybridEmbeddingRouter:
    """Combined embedding + heuristic router.

    Uses embedding predictions when confident, falls back to heuristics.
    Can be used as drop-in replacement for LearnedRouter in tiers.py.
    """

    def __init__(
        self,
        embedding_router: EmbeddingRouter | None = None,
        heuristic_fn: Any = None,
        enable_embeddings: bool = True,
    ):
        """Initialize hybrid router.

        Args:
            embedding_router: Embedding router (created if None and enabled)
            heuristic_fn: Fallback heuristic function
            enable_embeddings: Whether to use embeddings (can be disabled)
        """
        self.enable_embeddings = enable_embeddings
        self.embedding_router = embedding_router
        self.heuristic_fn = heuristic_fn

        # Lazy-init embedding router only if enabled
        if enable_embeddings and embedding_router is None:
            try:
                self.embedding_router = EmbeddingRouter()
            except ImportError:
                logger.warning(
                    "[HybridEmbeddingRouter] Could not load embedding router, "
                    "falling back to heuristic only"
                )
                self.enable_embeddings = False

    def predict(
        self,
        objective: str,
        context: dict | None = None,
        exclude: list[str] | None = None,
    ) -> tuple[str | None, dict[str, float], str]:
        """Predict best specialist, falling back to heuristic if needed."""
        exclude = exclude or []

        # Try embedding router first
        if self.enable_embeddings and self.embedding_router:
            try:
                specialist, scores, reasoning = self.embedding_router.predict(
                    objective, context, exclude
                )
                if specialist is not None:
                    return specialist, scores, f"embedding:{reasoning}"
            except Exception as e:
                logger.warning(f"[HybridEmbeddingRouter] Embedding failed: {e}")

        # Fall back to heuristic
        if self.heuristic_fn:
            specialist = self.heuristic_fn(objective, context, exclude)
            return specialist, {}, "heuristic:fallback"

        # Default fallback
        for s in ["python", "general"]:
            if s not in exclude:
                return s, {}, "default:no_heuristic"

        return "general", {}, "default:exhausted"

    def record_outcome(
        self,
        task_id: str,
        objective: str,
        context: dict | None,
        specialist: str,
        success: bool,
        **kwargs,
    ) -> None:
        """Record outcome to update embedding router."""
        if self.embedding_router:
            self.embedding_router.record_outcome(
                task_id=task_id,
                objective=objective,
                context=context,
                specialist=specialist,
                success=success,
            )


# Integration helper for RoutingDispatcher
def create_embedding_select_specialist(
    router: HybridEmbeddingRouter,
) -> callable:
    """Create a _select_specialist replacement for RoutingDispatcher.

    Returns a function compatible with RoutingDispatcher._select_specialist.
    """
    def select_specialist(
        task: Any,
        exclude: list[str] | None = None,
    ) -> tuple[str, str, list[str]]:
        """Select specialist using embedding routing.

        Returns:
            Tuple of (specialist, routing_strategy, routing_signals)
        """
        objective = task.objective
        context = task.context if hasattr(task, "context") else None

        specialist, scores, reasoning = router.predict(objective, context, exclude)

        if specialist is None:
            # Router declined, return default
            specialist = "python"
            for s in ["python", "general"]:
                if s not in (exclude or []):
                    specialist = s
                    break

        # Determine strategy from reasoning
        if reasoning.startswith("embedding:"):
            routing_strategy = "embedding"
        else:
            routing_strategy = "heuristic_fallback"

        routing_signals = [f"method:{routing_strategy}", f"reasoning:{reasoning}"]
        for name, score in scores.items():
            routing_signals.append(f"score:{name}={score:.2f}")

        return specialist, routing_strategy, routing_signals

    return select_specialist


if __name__ == "__main__":
    # Demo
    logging.basicConfig(level=logging.INFO)

    print("Initializing EmbeddingRouter...")
    router = EmbeddingRouter()

    # Test predictions
    test_cases = [
        "Given an array of integers, return indices of two numbers that add up to target",
        "Prove that the algorithm runs in O(n log n) time complexity",
        "Implement a stack data structure with push, pop, and peek operations",
        "Find the longest palindromic substring in a given string",
        "Calculate the mathematical formula for Fibonacci numbers",
        # Cross-domain tasks (good candidates for blending)
        "Write a Python function to calculate prime factorization using Sieve of Eratosthenes",
        "Implement dynamic programming to compute optimal matrix chain multiplication",
    ]

    print("\n--- Standard Predictions ---")
    for test in test_cases:
        specialist, scores, reasoning = router.predict(test)
        print(f"\nTask: {test[:60]}...")
        print(f"  Prediction: {specialist}")
        print(f"  Reasoning: {reasoning}")

    print("\n\n--- Weighted Routing (hi_moe-zrn) ---")
    for test in test_cases:
        weights, reasoning = router.predict_weighted(test)
        should_blend, _, blend_reason = router.recommend_blending(test)
        print(f"\nTask: {test[:60]}...")
        print(f"  Weights: {weights}")
        print(f"  Blend? {should_blend} - {blend_reason}")

    print("\n\n--- Probabilistic Selection (5 samples each) ---")
    for test in test_cases[-2:]:  # Just the cross-domain ones
        print(f"\nTask: {test[:60]}...")
        selections = []
        for _ in range(5):
            specialist, weights, _ = router.select_probabilistic(test)
            selections.append(specialist)
        print(f"  Weights: {weights}")
        print(f"  5 samples: {selections}")

    # Simulate some outcomes for prototype learning
    print("\n\n--- Recording outcomes ---")
    router.record_outcome("t1", "implement two sum", None, "python", True)
    router.record_outcome("t2", "prove correctness", None, "math", True)
    router.record_outcome("t3", "write binary search", None, "python", True)

    print("\n--- Stats ---")
    print(json.dumps(router.get_stats(), indent=2))
