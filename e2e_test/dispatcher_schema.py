"""Dispatcher structured output schema and parsing.

Implements issue hi_moe-4dy: Dispatcher structured output via prompt enforcement.

Schema: {"steps": [{"description": "string", "specialist": "python|math|general"}]}
Strategy: Prompt engineering + validation + single retry
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

# Valid specialist types for v0.1
SpecialistType = Literal["python", "math", "general"]
VALID_SPECIALISTS = {"python", "math", "general"}


@dataclass
class Step:
    """A single step in the dispatcher's execution plan."""

    description: str
    specialist: SpecialistType

    def __post_init__(self):
        if not self.description or not self.description.strip():
            raise ValueError("Step description cannot be empty")
        if self.specialist not in VALID_SPECIALISTS:
            raise ValueError(
                f"Invalid specialist '{self.specialist}'. "
                f"Must be one of: {', '.join(VALID_SPECIALISTS)}"
            )


@dataclass
class DispatcherPlan:
    """Structured output from the Dispatcher tier.

    Represents a linear sequence of steps to execute.
    v0.1: Sequential execution only (no DAG support yet).
    """

    steps: list[Step] = field(default_factory=list)

    def __post_init__(self):
        if not self.steps:
            raise ValueError("Plan must have at least one step")

    @classmethod
    def from_dict(cls, data: dict) -> "DispatcherPlan":
        """Parse from dictionary, validating structure."""
        if "steps" not in data:
            raise ValueError("Missing 'steps' key in plan")

        steps_data = data["steps"]
        if not isinstance(steps_data, list):
            raise ValueError("'steps' must be a list")

        steps = []
        for i, step_data in enumerate(steps_data):
            if not isinstance(step_data, dict):
                raise ValueError(f"Step {i} must be an object")

            if "description" not in step_data:
                raise ValueError(f"Step {i} missing 'description'")
            if "specialist" not in step_data:
                raise ValueError(f"Step {i} missing 'specialist'")

            steps.append(
                Step(
                    description=step_data["description"],
                    specialist=step_data["specialist"],
                )
            )

        return cls(steps=steps)


# Prompt template for structured output (hi_moe-eet: optimized for token efficiency)
# Original: 640 chars -> Optimized: 280 chars (~56% reduction)
DISPATCHER_SYSTEM_PROMPT = """Task dispatcher. Output JSON only:
{"steps": [{"description": "action", "specialist": "python|math|general"}]}
Specialists: python=code, math=analysis, general=other. 1-3 steps max. No markdown."""

DISPATCHER_USER_PROMPT = """Task: {objective}
{context_section}
JSON: {{"steps": [...]}}"""


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response, handling common formatting issues.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If no valid JSON found
    """
    # Strip whitespace
    text = response.strip()

    # Handle QwQ model's <think>...</think> reasoning traces
    # The model outputs reasoning in think tags before the actual JSON response
    think_pattern = r"<think>.*?</think>"
    text = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code blocks
    patterns = [
        r"```json\s*(.*?)\s*```",  # ```json ... ```
        r"```\s*(.*?)\s*```",  # ``` ... ```
        r"\{.*\}",  # Raw JSON object
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if "```" in pattern else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")


def parse_dispatcher_response(response: str) -> DispatcherPlan:
    """Parse and validate dispatcher response into DispatcherPlan.

    Args:
        response: Raw LLM response text

    Returns:
        Validated DispatcherPlan

    Raises:
        ValueError: If parsing or validation fails
    """
    data = extract_json_from_response(response)
    return DispatcherPlan.from_dict(data)


async def get_dispatcher_plan(
    llm_client,
    objective: str,
    context: dict | None = None,
    max_retries: int = 1,
) -> DispatcherPlan:
    """Get structured plan from dispatcher with retry on parse failure.

    Args:
        llm_client: LLM client with generate() method
        objective: Task objective to break down
        context: Optional context dict
        max_retries: Number of retries on parse failure (default: 1)

    Returns:
        Validated DispatcherPlan

    Raises:
        ValueError: If parsing fails after all retries
    """
    # Build context section
    context_section = ""
    if context:
        plan = context.get("plan", "")
        if plan:
            context_section = f"\nPlan context:\n{plan}\n"

    user_prompt = DISPATCHER_USER_PROMPT.format(
        objective=objective,
        context_section=context_section,
    )

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = await llm_client.generate(
                [
                    {"role": "system", "content": DISPATCHER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temp for structured output
                max_tokens=512,
            )

            plan = parse_dispatcher_response(response)
            logger.info(
                f"[Dispatcher] Parsed plan with {len(plan.steps)} steps "
                f"(attempt {attempt + 1})"
            )
            return plan

        except ValueError as e:
            last_error = e
            logger.warning(
                f"[Dispatcher] Parse failed (attempt {attempt + 1}): {e}"
            )
            if attempt < max_retries:
                # Add error context to retry prompt
                user_prompt = (
                    f"{DISPATCHER_USER_PROMPT.format(objective=objective, context_section=context_section)}\n\n"
                    f"IMPORTANT: Your previous response was invalid JSON. "
                    f"Error: {e}\n"
                    f"Output ONLY valid JSON with no extra text."
                )

    raise ValueError(f"Failed to parse dispatcher output after {max_retries + 1} attempts: {last_error}")
