# hi_moe: Mixture of Experts Harness

## Vision

A harness that enables a mixture of experts to cooperate on long-horizon tasks, solving the context drift problem through structured coordination and persistent state.

## Problem Statement

Current LLM architectures lose coherence on extended tasks. Context windows fill, earlier decisions get forgotten, and the system drifts from its original goals. We need a way to maintain strategic alignment while delegating specialized work.

## Architecture Overview

### Four-Tier Hierarchy

```
┌─────────────────────────────────────────┐
│         PROGRESS MONITOR                │  ← Tier 4: Meta-evaluation
│   Tracks progress, detects surprise,    │     "Are we getting somewhere?"
│   provides learning signal              │
├─────────────────────────────────────────┤
│         ABSTRACT ARCHITECT              │  ← Tier 3: Strategic planning
│   Sets goals, maintains big picture,    │     "What are we trying to do?"
│   reads outcomes from below             │
├─────────────────────────────────────────┤
│         ROUTING DISPATCHER              │  ← Tier 2: Task decomposition
│   Breaks tasks into graphs,             │     "Who should do this?"
│   assigns to specialists                │
├─────────────────────────────────────────┤
│         SPECIALIZED FLEET               │  ← Tier 1: Domain execution
│   LoRA adapters: Python, CUDA,          │     "How do I solve this specific thing?"
│   Math, etc.                            │
└─────────────────────────────────────────┘
```

### Key Technical Decisions

- **Base Model**: Qwen QwQ-32B (frozen, shared across all tiers)
- **Specialization**: LoRA adapters for domain-specific execution
- **Hot-Swap**: S-LoRA + vLLM unified paging (12+ concurrent specialists)
- **State Management**: Beads - shared state object all tiers can read
- **Routing**: Hybrid approach (hardcoded rules + learned Routing LoRA)

### Communication Pattern

- **Down**: Structured delegation with clear task specifications
- **Up**: Outcome reporting (not just success/failure, but what actually happened)
- **Across**: Beads state object provides shared context

## Core Innovations

### 1. Progress as Top-Level Concern
The system tracks whether it's making progress, not just completing tasks. Spinning without advancement gets detected and triggers strategy revision.

### 2. Surprise-Driven Learning
- Outcomes within confidence bounds → compress and forget
- Surprising outcomes → flag for analysis, update strategies

### 3. Routine Caching
Successful patterns get versioned and saved. Similar problems retrieve proven approaches rather than rediscovering them.

### 4. Self-Reflection
The system can model itself: which specialists are loaded, routing accuracy, bottlenecks. The Architect reasons about system capabilities when planning.

## Validation Strategy

**First stress test**: Competitive programming problems
- Objectively measurable outcomes
- Requires genuine tier coordination
- Clear signal on whether the architecture works

## Open Questions

1. **Fixed vs. Adaptive**: Is this a static orchestration system, or does the harness itself learn?
2. **Soft vs. Hard Routing**: Single specialist assignment vs. weighted contributions?
3. **Online Learning**: Can LoRA weights update incrementally from successful executions?

## Project Status

See `bd list --status=open` for current issues and `bd ready` for available work.
