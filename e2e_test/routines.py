"""Routines and skills - learned patterns for reuse.

Implements:
- hi_moe-kzt: Introspective state object (system state awareness)
- hi_moe-881: Strategy versioning & routine saving
- hi_moe-a1p: Routine compression into skills

Key concepts:
- Routine: A successful routing pattern + specialist combination
- Skill: An abstracted routine that applies to a class of problems
- State: Current system configuration and capabilities
"""
from __future__ import annotations

import hashlib
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
class RoutingPattern:
    """A snapshot of a routing decision."""
    task_keywords: list[str]
    specialist: str
    strategy: str  # "math_first", "python_direct", "learned", etc.
    routing_signals: list[str]


@dataclass
class Routine:
    """A successful approach to a problem class (hi_moe-881).

    Captures the routing pattern, specialist, and context that led
    to success. Can be replayed on similar problems.
    """
    id: str
    name: str
    pattern: RoutingPattern
    problem_signature: str  # Hash of problem characteristics
    success_count: int = 1
    failure_count: int = 0
    avg_execution_time_ms: float = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0

    @property
    def is_reliable(self) -> bool:
        """Routine is reliable if success rate > 80% with enough samples."""
        return self.success_count >= 3 and self.success_rate >= 0.8

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "pattern": {
                "keywords": self.pattern.task_keywords,
                "specialist": self.pattern.specialist,
                "strategy": self.pattern.strategy,
            },
            "signature": self.problem_signature,
            "success_rate": f"{self.success_rate:.1%}",
            "usage_count": self.success_count + self.failure_count,
            "version": self.version,
        }


@dataclass
class Skill:
    """An abstracted routine that applies to a problem class (hi_moe-a1p).

    Skills are compressed from multiple similar routines. They represent
    learned capabilities that can be applied broadly.
    """
    id: str
    name: str
    description: str
    problem_class: str  # "array_manipulation", "graph_traversal", etc.
    required_specialists: list[str]
    typical_strategy: str
    trigger_keywords: list[str]
    source_routines: list[str]  # IDs of routines this was learned from
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def matches(self, objective: str) -> float:
        """Score how well this skill matches an objective (0-1)."""
        objective_lower = objective.lower()
        matches = sum(1 for kw in self.trigger_keywords if kw in objective_lower)
        return min(matches / max(len(self.trigger_keywords), 1), 1.0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "problem_class": self.problem_class,
            "specialists": self.required_specialists,
            "trigger_keywords": self.trigger_keywords,
            "confidence": f"{self.confidence:.1%}",
            "source_count": len(self.source_routines),
        }


@dataclass
class IntrospectiveState:
    """System state for Architect reasoning (hi_moe-kzt).

    Captures what the system knows about itself:
    - Loaded specialists and their capabilities
    - Current bottlenecks and performance
    - Available routines and skills
    """
    timestamp: str
    loaded_specialists: list[str]
    specialist_success_rates: dict[str, float]
    bottlenecks: list[str]
    available_routines: int
    available_skills: int
    routing_accuracy: float
    system_health: str

    def to_prompt_context(self) -> str:
        """Generate context for Architect prompts."""
        lines = [
            "## System State",
            f"- Health: {self.system_health}",
            f"- Specialists: {', '.join(self.loaded_specialists)}",
            f"- Routing accuracy: {self.routing_accuracy:.1%}",
            f"- Available routines: {self.available_routines}",
            f"- Available skills: {self.available_skills}",
        ]

        if self.bottlenecks:
            lines.append(f"- Bottlenecks: {', '.join(self.bottlenecks[:3])}")

        # Add specialist performance
        lines.append("\n### Specialist Performance")
        for spec, rate in sorted(self.specialist_success_rates.items(), key=lambda x: -x[1]):
            lines.append(f"- {spec}: {rate:.0%}")

        return "\n".join(lines)


class RoutineManager:
    """Manages routines and skills (hi_moe-881, hi_moe-a1p).

    Usage:
        manager = RoutineManager()

        # Record successful execution
        manager.record_success(task_id, objective, specialist, strategy, time_ms)

        # Find matching routine for new task
        routine = manager.find_routine(objective)

        # Compress routines into skills
        manager.compress_to_skills()

        # Get state for Architect
        state = manager.get_introspective_state()
    """

    def __init__(self, storage_path: Path | None = None):
        self.routines: dict[str, Routine] = {}
        self.skills: dict[str, Skill] = {}
        self.storage_path = storage_path

        # Metrics for introspection
        self.total_routings: int = 0
        self.successful_routings: int = 0
        self.specialist_stats: dict[str, dict] = defaultdict(
            lambda: {"success": 0, "failure": 0}
        )

        if storage_path and storage_path.exists():
            self._load()

    def record_success(
        self,
        task_id: str,
        objective: str,
        specialist: str,
        strategy: str,
        execution_time_ms: float,
        routing_signals: list[str] | None = None,
    ) -> Routine:
        """Record a successful task execution, creating or updating routine."""
        self.total_routings += 1
        self.successful_routings += 1
        self.specialist_stats[specialist]["success"] += 1

        # Extract keywords from objective
        keywords = self._extract_keywords(objective)
        signature = self._compute_signature(keywords)

        # Check if routine exists
        if signature in self.routines:
            routine = self.routines[signature]
            routine.success_count += 1
            routine.avg_execution_time_ms = (
                (routine.avg_execution_time_ms * (routine.success_count - 1) + execution_time_ms)
                / routine.success_count
            )
            routine.last_used = datetime.now().isoformat()
            routine.version += 1
            logger.info(f"[Routines] Updated routine '{routine.name}' (v{routine.version})")
        else:
            # Create new routine
            routine = Routine(
                id=f"routine-{len(self.routines) + 1}",
                name=self._generate_routine_name(keywords),
                pattern=RoutingPattern(
                    task_keywords=keywords,
                    specialist=specialist,
                    strategy=strategy,
                    routing_signals=routing_signals or [],
                ),
                problem_signature=signature,
                avg_execution_time_ms=execution_time_ms,
            )
            self.routines[signature] = routine
            logger.info(f"[Routines] Created routine '{routine.name}'")

        return routine

    def record_failure(
        self,
        task_id: str,
        objective: str,
        specialist: str,
    ) -> None:
        """Record a failed execution."""
        self.total_routings += 1
        self.specialist_stats[specialist]["failure"] += 1

        keywords = self._extract_keywords(objective)
        signature = self._compute_signature(keywords)

        if signature in self.routines:
            self.routines[signature].failure_count += 1
            logger.debug(f"[Routines] Recorded failure for signature {signature[:8]}")

    def find_routine(self, objective: str) -> Routine | None:
        """Find a matching routine for an objective."""
        keywords = self._extract_keywords(objective)
        signature = self._compute_signature(keywords)

        # Exact match
        if signature in self.routines:
            routine = self.routines[signature]
            if routine.is_reliable:
                return routine

        # Fuzzy match by keyword overlap
        best_match = None
        best_score = 0.5  # Minimum threshold

        for routine in self.routines.values():
            if not routine.is_reliable:
                continue

            score = self._keyword_overlap(keywords, routine.pattern.task_keywords)
            if score > best_score:
                best_score = score
                best_match = routine

        return best_match

    def find_skill(self, objective: str) -> Skill | None:
        """Find a matching skill for an objective."""
        best_match = None
        best_score = 0.3  # Minimum threshold

        for skill in self.skills.values():
            score = skill.matches(objective)
            if score > best_score:
                best_score = score
                best_match = skill

        return best_match

    def compress_to_skills(self, min_routines: int = 3) -> list[Skill]:
        """Compress similar routines into skills (hi_moe-a1p).

        Looks for patterns across multiple routines and abstracts them.
        """
        new_skills = []

        # Group routines by specialist
        by_specialist = defaultdict(list)
        for routine in self.routines.values():
            if routine.is_reliable:
                by_specialist[routine.pattern.specialist].append(routine)

        # Look for patterns within each specialist
        for specialist, routines in by_specialist.items():
            if len(routines) < min_routines:
                continue

            # Find common keywords
            all_keywords = [set(r.pattern.task_keywords) for r in routines]
            common = set.intersection(*all_keywords) if all_keywords else set()

            if len(common) >= 2:
                skill_id = f"skill-{len(self.skills) + 1}"
                skill = Skill(
                    id=skill_id,
                    name=self._generate_skill_name(common),
                    description=f"Skill learned from {len(routines)} routines",
                    problem_class=self._infer_problem_class(common),
                    required_specialists=[specialist],
                    typical_strategy=routines[0].pattern.strategy,
                    trigger_keywords=list(common),
                    source_routines=[r.id for r in routines],
                    confidence=sum(r.success_rate for r in routines) / len(routines),
                )
                self.skills[skill_id] = skill
                new_skills.append(skill)
                logger.info(f"[Routines] Compressed {len(routines)} routines into skill '{skill.name}'")

        return new_skills

    def get_introspective_state(self) -> IntrospectiveState:
        """Get current system state for Architect reasoning (hi_moe-kzt)."""
        # Calculate specialist success rates
        success_rates = {}
        for spec, stats in self.specialist_stats.items():
            total = stats["success"] + stats["failure"]
            success_rates[spec] = stats["success"] / total if total > 0 else 0.5

        # Identify bottlenecks
        bottlenecks = []
        for spec, rate in success_rates.items():
            if rate < 0.7:
                bottlenecks.append(f"{spec} ({rate:.0%} success)")

        # Routing accuracy
        routing_accuracy = (
            self.successful_routings / self.total_routings
            if self.total_routings > 0 else 0.5
        )

        # System health
        if routing_accuracy >= 0.85:
            health = "healthy"
        elif routing_accuracy >= 0.7:
            health = "degraded"
        else:
            health = "unhealthy"

        return IntrospectiveState(
            timestamp=datetime.now().isoformat(),
            loaded_specialists=list(self.specialist_stats.keys()),
            specialist_success_rates=success_rates,
            bottlenecks=bottlenecks,
            available_routines=len([r for r in self.routines.values() if r.is_reliable]),
            available_skills=len(self.skills),
            routing_accuracy=routing_accuracy,
            system_health=health,
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from text."""
        # Remove common words
        stopwords = {"a", "an", "the", "is", "are", "to", "for", "and", "or", "in", "on", "at"}

        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Keep most relevant (up to 10)
        return keywords[:10]

    def _compute_signature(self, keywords: list[str]) -> str:
        """Compute signature hash from keywords."""
        canonical = " ".join(sorted(keywords))
        return hashlib.md5(canonical.encode()).hexdigest()[:12]

    def _keyword_overlap(self, kw1: list[str], kw2: list[str]) -> float:
        """Compute overlap score between keyword lists."""
        set1, set2 = set(kw1), set(kw2)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union

    def _generate_routine_name(self, keywords: list[str]) -> str:
        """Generate human-readable routine name."""
        if not keywords:
            return "unnamed_routine"
        return "_".join(keywords[:3])

    def _generate_skill_name(self, keywords: set[str]) -> str:
        """Generate human-readable skill name."""
        if not keywords:
            return "unnamed_skill"
        return "_".join(sorted(keywords)[:2]) + "_skill"

    def _infer_problem_class(self, keywords: set[str]) -> str:
        """Infer problem class from keywords."""
        class_hints = {
            "array_manipulation": {"array", "list", "sort", "search"},
            "graph_traversal": {"graph", "tree", "node", "path"},
            "dynamic_programming": {"dp", "memoization", "subsequence"},
            "string_processing": {"string", "substring", "pattern"},
            "math_analysis": {"math", "calculate", "formula"},
        }

        for pclass, hints in class_hints.items():
            if keywords & hints:
                return pclass

        return "general"

    def get_stats(self) -> dict:
        """Get routine/skill statistics."""
        reliable = [r for r in self.routines.values() if r.is_reliable]
        return {
            "total_routines": len(self.routines),
            "reliable_routines": len(reliable),
            "total_skills": len(self.skills),
            "total_routings": self.total_routings,
            "routing_accuracy": f"{self.successful_routings / max(self.total_routings, 1):.1%}",
        }

    def save(self) -> None:
        """Save state to storage."""
        if not self.storage_path:
            return

        data = {
            "routines": {sig: r.to_dict() for sig, r in self.routines.items()},
            "skills": {sid: s.to_dict() for sid, s in self.skills.items()},
            "stats": self.get_stats(),
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[Routines] Saved to {self.storage_path}")

    def _load(self) -> None:
        """Load state from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)
            logger.info(f"[Routines] Loaded from {self.storage_path}")
        except Exception as e:
            logger.warning(f"[Routines] Failed to load: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = RoutineManager()

    # Simulate successful executions
    tasks = [
        ("Implement two sum using hash map", "python", "python_direct"),
        ("Write a function to find two numbers that add to target", "python", "python_direct"),
        ("Solve the two sum problem efficiently", "python", "learned"),
        ("Prove the algorithm runs in O(n)", "math", "math_first"),
        ("Analyze time complexity of the solution", "math", "math_first"),
        ("Show that the approach is optimal", "math", "math_first"),
        ("Sort an array in ascending order", "python", "python_direct"),
        ("Implement quicksort algorithm", "python", "python_direct"),
    ]

    for i, (objective, specialist, strategy) in enumerate(tasks):
        manager.record_success(f"task-{i}", objective, specialist, strategy, 2000 + i * 100)

    # Simulate some failures
    manager.record_failure("task-99", "Debug the memory leak", "debugging")
    manager.record_failure("task-100", "Fix the bug in recursion", "debugging")

    # Find routine for new task
    routine = manager.find_routine("Find two elements that sum to a value")
    if routine:
        print(f"\nFound routine: {routine.name} (specialist: {routine.pattern.specialist})")

    # Compress to skills
    print("\n--- Compressing routines to skills ---")
    new_skills = manager.compress_to_skills(min_routines=2)
    for skill in new_skills:
        print(f"  Created: {skill.name} ({skill.problem_class})")

    # Get introspective state
    print("\n--- Introspective State ---")
    state = manager.get_introspective_state()
    print(state.to_prompt_context())

    # Stats
    print("\n--- Stats ---")
    print(json.dumps(manager.get_stats(), indent=2))
