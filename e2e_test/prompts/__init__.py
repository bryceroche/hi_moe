"""Optimized prompts for hi-moe tiers (hi_moe-jhf)."""

from .dispatcher_7b import (
    SYSTEM_PROMPT as DISPATCHER_7B_SYSTEM,
    ROUTING_SCHEMA,
    FEW_SHOT_EXAMPLES,
    VLLM_GUIDED_CONFIG,
    build_prompt,
    build_minimal_prompt,
)

__all__ = [
    "DISPATCHER_7B_SYSTEM",
    "ROUTING_SCHEMA",
    "FEW_SHOT_EXAMPLES",
    "VLLM_GUIDED_CONFIG",
    "build_prompt",
    "build_minimal_prompt",
]
