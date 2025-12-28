"""7B Dispatcher prompt optimized for sub-100ms routing (hi_moe-jhf).

This prompt is designed for Qwen2.5-7B-Instruct to act as a fast routing
classifier. Key optimizations:
- JSON-constrained output for deterministic parsing
- Classification-style prompt (not generative)
- Few-shot examples for edge cases
- Minimal output tokens

Use with vLLM guided decoding (outlines) to enforce schema.
"""

from typing import Literal

# Valid specialists for routing
SPECIALISTS = Literal["python", "math", "algorithms", "debugging", "refactoring"]

# JSON output schema for vLLM guided decoding
ROUTING_SCHEMA = {
    "type": "object",
    "properties": {
        "specialist": {
            "type": "string",
            "enum": ["python", "math", "algorithms", "debugging", "refactoring"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        }
    },
    "required": ["specialist", "confidence"],
    "additionalProperties": False
}

# System prompt - classification focused, no reasoning output
SYSTEM_PROMPT = """You are a fast routing classifier. Given a coding problem, output JSON selecting the best specialist.

Specialists:
- python: General code implementation, functions, classes
- math: Mathematical reasoning, proofs, formulas, number theory
- algorithms: Sorting, searching, graph algorithms, dynamic programming, optimization
- debugging: Fix bugs, errors, exceptions in existing code
- refactoring: Clean up, improve, optimize existing working code

Output ONLY valid JSON: {"specialist": "...", "confidence": 0.0-1.0}"""

# Few-shot examples for edge cases (Math vs Algorithms boundary)
FEW_SHOT_EXAMPLES = [
    # Clear python
    {
        "problem": "Write a function that reverses a string",
        "output": {"specialist": "python", "confidence": 0.95}
    },
    # Clear math
    {
        "problem": "Prove that the sum of first n odd numbers equals n squared",
        "output": {"specialist": "math", "confidence": 0.95}
    },
    # Clear algorithms
    {
        "problem": "Implement Dijkstra's shortest path algorithm",
        "output": {"specialist": "algorithms", "confidence": 0.95}
    },
    # Edge case: Math vs Algorithms (dynamic programming)
    {
        "problem": "Find the minimum number of coins to make change for amount N",
        "output": {"specialist": "algorithms", "confidence": 0.80}
    },
    # Edge case: Math vs Algorithms (number theory + optimization)
    {
        "problem": "Find the nth Fibonacci number efficiently",
        "output": {"specialist": "math", "confidence": 0.70}
    },
    # Edge case: Python vs Algorithms (implementation focus)
    {
        "problem": "Given an array, return indices of two numbers that add to target",
        "output": {"specialist": "python", "confidence": 0.75}
    },
    # Clear debugging
    {
        "problem": "This code throws IndexError, fix it: def get_last(arr): return arr[len(arr)]",
        "output": {"specialist": "debugging", "confidence": 0.95}
    },
    # Clear refactoring
    {
        "problem": "This working code is too slow, optimize it without changing behavior",
        "output": {"specialist": "refactoring", "confidence": 0.90}
    },
]


def build_prompt(problem: str, include_few_shot: bool = True) -> list[dict]:
    """Build the prompt for 7B dispatcher.

    Args:
        problem: The problem description to route
        include_few_shot: Whether to include few-shot examples (adds ~200 tokens)

    Returns:
        List of messages for chat completion
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if include_few_shot:
        # Add few-shot examples as user/assistant pairs
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example["problem"]})
            messages.append({
                "role": "assistant",
                "content": f'{{"specialist": "{example["output"]["specialist"]}", "confidence": {example["output"]["confidence"]}}}'
            })

    # Add the actual problem
    messages.append({"role": "user", "content": problem})

    return messages


def build_minimal_prompt(problem: str) -> list[dict]:
    """Build minimal prompt without few-shot for maximum speed.

    Use when latency is critical and problem is straightforward.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem}
    ]


# vLLM guided decoding config
VLLM_GUIDED_CONFIG = {
    "guided_json": ROUTING_SCHEMA,
    "guided_decoding_backend": "outlines"  # or "lm-format-enforcer"
}


# Token budget estimates
TOKEN_ESTIMATES = {
    "system_prompt": 80,
    "few_shot_examples": 200,  # 8 examples * ~25 tokens each
    "max_output": 30,  # {"specialist": "algorithms", "confidence": 0.80}
    "typical_problem": 50,

    # Total with few-shot: ~360 tokens input + 30 output
    # Total minimal: ~130 tokens input + 30 output
}
