"""Handoff protocol training data generator (hi_moe-9km).

Generates training examples that demonstrate proper handoff behavior
between tiers. Prevents mode collapse during specialist LoRA training
by including explicit examples of:

1. When to accept a task (within specialist scope)
2. When to escalate (outside specialist scope)
3. Proper handoff format and context preservation
4. Tier boundary respect

Data quality matters more than training technique.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

# Specialist types for training
SPECIALIST_TYPES = ["python", "math", "algorithms", "debugging", "refactoring"]

# Task categories by specialist domain
DOMAIN_TASKS = {
    "python": [
        "Implement a function to reverse a string",
        "Write a class for a linked list",
        "Create a decorator for timing functions",
        "Implement a context manager for file handling",
        "Write unit tests for the user authentication module",
    ],
    "math": [
        "Prove that the algorithm is O(n log n)",
        "Calculate the expected value of the random variable",
        "Derive the closed-form solution for the recurrence",
        "Analyze the convergence of the iterative method",
        "Prove the correctness of the greedy approach",
    ],
    "algorithms": [
        "Design an efficient algorithm for finding the k-th largest element",
        "Implement dynamic programming solution for longest common subsequence",
        "Create a graph algorithm to detect cycles",
        "Optimize the sorting algorithm for nearly-sorted data",
        "Design a data structure for range queries",
    ],
    "debugging": [
        "Find the bug causing the infinite loop",
        "Debug the memory leak in the cache implementation",
        "Identify why the race condition occurs",
        "Fix the off-by-one error in the binary search",
        "Trace the null pointer exception source",
    ],
    "refactoring": [
        "Refactor the monolithic function into smaller units",
        "Extract the common logic into a base class",
        "Improve the readability of the nested conditionals",
        "Apply the strategy pattern to the payment processing",
        "Remove code duplication in the validation logic",
    ],
}

# Cross-domain tasks that require escalation
ESCALATION_TASKS = {
    "python": [
        ("Prove that my implementation is correct", "math"),
        ("Is this algorithm optimal?", "math"),
        ("Can you analyze the time complexity?", "algorithms"),
        ("Why isn't my code working?", "debugging"),
    ],
    "math": [
        ("Implement the algorithm we derived", "python"),
        ("Write the code for this formula", "python"),
        ("Debug the numerical instability", "debugging"),
    ],
    "algorithms": [
        ("Write the Python implementation", "python"),
        ("Prove the correctness formally", "math"),
        ("Fix the edge case handling", "debugging"),
    ],
    "debugging": [
        ("Now implement the fix", "python"),
        ("Is the algorithm approach correct?", "math"),
        ("Refactor to prevent future bugs", "refactoring"),
    ],
    "refactoring": [
        ("Add the new feature to the refactored code", "python"),
        ("Verify the refactoring preserves correctness", "debugging"),
        ("Analyze if this improves performance", "algorithms"),
    ],
}


@dataclass
class HandoffExample:
    """A training example demonstrating handoff behavior."""

    specialist: str
    task: str
    action: Literal["accept", "escalate"]
    response: str
    escalate_to: str | None = None
    context_preserved: dict = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class TrainingDataset:
    """Collection of training examples for a specialist."""

    specialist: str
    examples: list[HandoffExample] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_jsonl(self) -> str:
        """Convert to JSONL format for training."""
        lines = []
        for ex in self.examples:
            entry = {
                "messages": [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": self._format_task(ex.task)},
                    {"role": "assistant", "content": ex.response},
                ],
                "metadata": {
                    "action": ex.action,
                    "escalate_to": ex.escalate_to,
                    "reasoning": ex.reasoning,
                },
            }
            lines.append(json.dumps(entry))
        return "\n".join(lines)

    def _get_system_prompt(self) -> str:
        """Get system prompt for this specialist."""
        prompts = {
            "python": """You are a Python programming specialist. Your role is to write clean, efficient Python code.

IMPORTANT: You must respect tier boundaries:
- ACCEPT tasks that involve Python implementation, testing, or code writing
- ESCALATE tasks that require mathematical proofs, algorithm analysis, or debugging existing code
- When escalating, preserve all context and explain why escalation is needed""",

            "math": """You are a mathematical reasoning specialist. Your role is to provide formal proofs and mathematical analysis.

IMPORTANT: You must respect tier boundaries:
- ACCEPT tasks that involve proofs, derivations, or mathematical analysis
- ESCALATE tasks that require code implementation or debugging
- When escalating, preserve all mathematical context""",

            "algorithms": """You are an algorithms specialist. Your role is to design efficient algorithms and analyze complexity.

IMPORTANT: You must respect tier boundaries:
- ACCEPT tasks that involve algorithm design, complexity analysis, or data structure selection
- ESCALATE tasks that require code implementation or formal proofs
- When escalating, include the algorithm design for implementation""",

            "debugging": """You are a debugging specialist. Your role is to identify and diagnose bugs.

IMPORTANT: You must respect tier boundaries:
- ACCEPT tasks that involve finding bugs, tracing errors, or diagnosing issues
- ESCALATE tasks that require implementing fixes or refactoring
- When escalating, include the root cause analysis""",

            "refactoring": """You are a refactoring specialist. Your role is to improve code structure and quality.

IMPORTANT: You must respect tier boundaries:
- ACCEPT tasks that involve restructuring, pattern application, or code organization
- ESCALATE tasks that require new feature implementation or bug fixes
- When escalating, include the refactoring plan""",
        }
        return prompts.get(self.specialist, prompts["python"])

    def _format_task(self, task: str) -> str:
        """Format task as user message."""
        return f"Task: {task}\n\nPlease process this task according to your specialist role."


def generate_accept_example(specialist: str, task: str) -> HandoffExample:
    """Generate an example where the specialist accepts the task."""
    responses = {
        "python": f"""I'll implement this Python solution.

```python
# Implementation for: {task}
def solution():
    # [Code implementation here]
    pass
```

This solution handles the requirements by [explanation].""",

        "math": f"""I'll provide a formal analysis.

**Theorem**: [Statement]

**Proof**:
1. [Step 1]
2. [Step 2]
3. Therefore, [conclusion]

This proves the required property for: {task}""",

        "algorithms": f"""I'll design an algorithm for this.

**Algorithm Design**:
- Approach: [Description]
- Time Complexity: O(n)
- Space Complexity: O(1)

**Pseudocode**:
```
function solve(input):
    [steps]
    return result
```

This algorithm efficiently handles: {task}""",

        "debugging": f"""I'll analyze and identify the bug.

**Investigation**:
1. Traced execution from [start point]
2. Found issue at [location]

**Root Cause**: [Description]

**Evidence**: [Stack trace or reproduction]

The bug causing "{task}" is [explanation].""",

        "refactoring": f"""I'll improve the code structure.

**Refactoring Plan**:
1. [Step 1]
2. [Step 2]

**Before/After**:
- Before: [Old pattern]
- After: [New pattern]

This refactoring addresses: {task}""",
    }

    return HandoffExample(
        specialist=specialist,
        task=task,
        action="accept",
        response=responses.get(specialist, responses["python"]),
        reasoning=f"Task is within {specialist} specialist scope",
    )


def generate_escalate_example(
    specialist: str,
    task: str,
    escalate_to: str,
) -> HandoffExample:
    """Generate an example where the specialist escalates the task."""
    context = {
        "original_task": task,
        "work_completed": f"Analyzed from {specialist} perspective",
        "escalation_reason": f"Requires {escalate_to} specialist capabilities",
    }

    response = f"""This task requires escalation to the {escalate_to} specialist.

**ESCALATE TO**: {escalate_to}

**Reason**: This task involves {escalate_to} expertise that is outside my scope as a {specialist} specialist.

**Context to Preserve**:
- Original request: {task}
- My analysis: [Brief analysis from {specialist} perspective]
- What's needed: [Specific {escalate_to} capabilities required]

I've prepared the handoff with all relevant context for the {escalate_to} specialist to continue."""

    return HandoffExample(
        specialist=specialist,
        task=task,
        action="escalate",
        response=response,
        escalate_to=escalate_to,
        context_preserved=context,
        reasoning=f"Task requires {escalate_to} capabilities, not {specialist}",
    )


def generate_training_dataset(
    specialist: str,
    n_accept: int = 10,
    n_escalate: int = 10,
) -> TrainingDataset:
    """Generate a balanced training dataset for a specialist.

    Args:
        specialist: Specialist type to generate data for
        n_accept: Number of accept examples
        n_escalate: Number of escalate examples

    Returns:
        TrainingDataset with balanced examples
    """
    examples = []

    # Generate accept examples (tasks within scope)
    domain_tasks = DOMAIN_TASKS.get(specialist, [])
    for _ in range(n_accept):
        task = random.choice(domain_tasks) if domain_tasks else "Implement a solution"
        # Add variation
        variations = [
            task,
            f"Please {task.lower()}",
            f"I need you to {task.lower()}",
            f"Can you {task.lower()}?",
        ]
        examples.append(generate_accept_example(specialist, random.choice(variations)))

    # Generate escalate examples (tasks outside scope)
    escalation_tasks = ESCALATION_TASKS.get(specialist, [])
    for _ in range(n_escalate):
        if escalation_tasks:
            task, target = random.choice(escalation_tasks)
        else:
            task, target = "Unknown task type", "general"
        # Add variation
        variations = [
            task,
            f"Also, {task.lower()}",
            f"Next, {task.lower()}",
            f"Finally, {task.lower()}",
        ]
        examples.append(generate_escalate_example(specialist, random.choice(variations), target))

    # Shuffle to mix accept and escalate
    random.shuffle(examples)

    return TrainingDataset(specialist=specialist, examples=examples)


def generate_all_datasets(
    output_dir: Path,
    n_accept: int = 20,
    n_escalate: int = 20,
) -> dict[str, Path]:
    """Generate training datasets for all specialists.

    Args:
        output_dir: Directory to save datasets
        n_accept: Accept examples per specialist
        n_escalate: Escalate examples per specialist

    Returns:
        Dict mapping specialist to output path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for specialist in SPECIALIST_TYPES:
        dataset = generate_training_dataset(specialist, n_accept, n_escalate)
        path = output_dir / f"{specialist}_handoff_training.jsonl"
        with open(path, "w") as f:
            f.write(dataset.to_jsonl())
        paths[specialist] = path
        print(f"Generated {len(dataset.examples)} examples for {specialist}: {path}")

    # Also generate a combined dataset
    combined = []
    for specialist in SPECIALIST_TYPES:
        dataset = generate_training_dataset(specialist, n_accept // 2, n_escalate // 2)
        combined.extend(dataset.examples)

    random.shuffle(combined)
    combined_dataset = TrainingDataset(specialist="combined", examples=combined)
    combined_path = output_dir / "combined_handoff_training.jsonl"
    with open(combined_path, "w") as f:
        f.write(combined_dataset.to_jsonl())
    paths["combined"] = combined_path
    print(f"Generated {len(combined)} combined examples: {combined_path}")

    return paths


def validate_dataset(path: Path) -> dict:
    """Validate a training dataset for quality.

    Returns stats about accept/escalate balance and coverage.
    """
    with open(path) as f:
        lines = f.readlines()

    stats = {
        "total": len(lines),
        "accept": 0,
        "escalate": 0,
        "escalate_targets": {},
    }

    for line in lines:
        data = json.loads(line)
        meta = data.get("metadata", {})
        action = meta.get("action", "unknown")
        if action == "accept":
            stats["accept"] += 1
        elif action == "escalate":
            stats["escalate"] += 1
            target = meta.get("escalate_to", "unknown")
            stats["escalate_targets"][target] = stats["escalate_targets"].get(target, 0) + 1

    stats["balance_ratio"] = (
        stats["accept"] / stats["escalate"] if stats["escalate"] > 0 else float("inf")
    )

    return stats


if __name__ == "__main__":
    import sys

    output_dir = Path("data/handoff_training")

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    print(f"Generating handoff training data to {output_dir}")
    paths = generate_all_datasets(output_dir)

    print("\n--- Validation ---")
    for specialist, path in paths.items():
        stats = validate_dataset(path)
        print(f"\n{specialist}:")
        print(f"  Total: {stats['total']}")
        print(f"  Accept: {stats['accept']}, Escalate: {stats['escalate']}")
        print(f"  Balance: {stats['balance_ratio']:.2f}")
        if stats["escalate_targets"]:
            print(f"  Escalate targets: {stats['escalate_targets']}")
