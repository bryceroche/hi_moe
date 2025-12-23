# Abstract Architect Specification

> The strategic tier - sets goals and maintains the big picture.

## Related Specifications

- **[Handoff Protocol](./handoff_protocol.md)**: Message, DelegationPayload, OutcomePayload types
- **[Routing Dispatcher](./routing_dispatcher.md)**: Receives delegations from Architect
- **[LoRA Infrastructure](./lora_infrastructure.md)**: Base model configuration

## Required Imports

```python
from __future__ import annotations
import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from openai import AsyncOpenAI

# From other specs
from handoff_protocol import (
    Message, TierLevel, DelegationPayload, OutcomePayload,
    OutcomeStatus, CancellationPayload, Artifact
)
```

## Overview

The Abstract Architect is the strategic tier of the hi_moe hierarchy. It receives high-level tasks, develops strategic plans, and delegates work to the Routing Dispatcher. The Architect maintains the "big picture" - understanding what we're trying to achieve and why, while leaving tactical decisions (how to decompose, which specialist) to lower tiers.

```
External Task / Progress Monitor
       │
       │ Problem statement, constraints
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ABSTRACT ARCHITECT                        │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Strategy   │    │   Context    │    │   Outcome    │  │
│  │   Planner    │───▶│   Manager    │───▶│   Handler    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                 Strategic Context                     │  │
│  │   • Current objective           • Progress summary    │  │
│  │   • Constraints                 • Strategy revision   │  │
│  │   • Success criteria            • Failure analysis    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
       │
       │ DelegationPayload (task_type="decompose_and_solve")
       ▼
Routing Dispatcher
```

## Architect Interface

### Input: External Task

```python
@dataclass
class ArchitectTask:
    """What the Architect receives from external source or Progress Monitor."""
    task_id: str                    # Unique identifier
    problem_statement: str          # What needs to be solved
    constraints: list[str]          # Hard requirements
    success_criteria: list[str]     # How to know we succeeded
    context: dict[str, Any]         # Additional context (test cases, examples)
    priority: int                   # 0=critical, 4=backlog
    timeout_ms: int                 # Total time budget for the task
    metadata: dict[str, Any] = field(default_factory=dict)
```

### Output: Architect Result

```python
@dataclass
class ArchitectResult:
    """What the Architect returns after processing a task."""
    task_id: str
    status: OutcomeStatus
    summary: str                    # What happened overall
    solution: str | None            # Final solution if successful
    artifacts: list[Artifact]       # All outputs
    confidence: float               # Overall confidence
    execution_time_ms: int
    strategy_revisions: int         # How many times we revised approach
    surprise_flag: bool
    surprise_reason: str | None
    error_info: ErrorInfo | None

@dataclass
class ErrorInfo:
    """Error details for failed tasks."""
    code: str
    message: str
    recoverable: bool
    attempted_strategies: list[str]
```

## Strategy Planner

### Planner Design

The Strategy Planner analyzes tasks and develops high-level approaches.

```python
class StrategyPlanner:
    """Develops strategic approaches for tasks."""

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        beads: BeadsClient,
        max_strategies: int = 3,
    ):
        self.client = llm_client
        self.beads = beads
        self.max_strategies = max_strategies

    async def plan(self, task: ArchitectTask) -> StrategicPlan:
        """Develop a strategic plan for the task."""
        # Check for cached routines first
        cached = await self._check_routine_cache(task)
        if cached:
            return cached

        # Generate strategic options
        response = await self.client.chat.completions.create(
            model="base",  # Frozen QwQ-32B
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": self._build_planning_prompt(task)},
            ],
            temperature=0.3,
        )

        plan = self._parse_plan(response.choices[0].message.content)

        # Validate plan is actionable
        self._validate_plan(plan, task)

        return plan

    async def _check_routine_cache(self, task: ArchitectTask) -> StrategicPlan | None:
        """Check if we have a proven routine for similar tasks."""
        # Hash task characteristics for lookup
        task_signature = self.compute_signature(task)
        cached = await self.beads.get(f"history/routines/{task_signature}")

        if cached and cached.get("success_rate", 0) >= 0.8:
            return StrategicPlan.from_cached(cached)
        return None

    def compute_signature(self, task: ArchitectTask) -> str:
        """Compute a signature for routine matching (public for caching)."""
        # Extract key features for similarity matching
        features = {
            "keywords": self._extract_keywords(task.problem_statement),
            "constraint_types": [c.split()[0].lower() for c in task.constraints if c],
        }
        return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()[:12]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract significant keywords from problem statement."""
        # Remove common words, keep technical terms
        stopwords = {"a", "an", "the", "is", "are", "be", "to", "of", "and", "or", "in", "on", "for", "with", "that", "this", "it"}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        # Return top 10 most distinctive
        return sorted(set(keywords))[:10]

    def _build_planning_prompt(self, task: ArchitectTask) -> str:
        """Build the planning prompt."""
        return PLANNING_PROMPT.format(
            problem=task.problem_statement,
            constraints="\n".join(f"- {c}" for c in task.constraints),
            success_criteria="\n".join(f"- {c}" for c in task.success_criteria),
            context=json.dumps(task.context, indent=2) if task.context else "(none)",
        )

    def _parse_plan(self, content: str) -> StrategicPlan:
        """Parse LLM response into StrategicPlan."""
        # Extract JSON from markdown code block if present
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.index("```") + 3
            end = content.index("```", start)
            content = content[start:end].strip()

        data = json.loads(content)
        return StrategicPlan(
            analysis=data.get("analysis", ""),
            approach=data.get("approach", ""),
            delegation_type=data.get("delegation_type", "decompose_and_solve"),
            key_challenges=data.get("key_challenges", []),
            success_indicators=data.get("success_indicators", []),
            fallback_strategies=data.get("fallback_strategies", []),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            timeout_allocation=data.get("timeout_allocation", {}),
        )

    def _validate_plan(self, plan: StrategicPlan, task: ArchitectTask) -> None:
        """Validate plan is actionable."""
        if not plan.analysis:
            raise ValueError("Plan missing analysis")
        if not plan.approach:
            raise ValueError("Plan missing approach")
        if plan.delegation_type not in ("decompose_and_solve", "direct_solve", "analyze_only"):
            raise ValueError(f"Invalid delegation_type: {plan.delegation_type}")
```

### Strategic Plan Structure

```python
@dataclass
class StrategicPlan:
    """High-level approach for solving a task."""
    analysis: str                   # Understanding of the problem
    approach: str                   # Chosen strategy description
    delegation_type: str            # "decompose_and_solve", "direct_solve", etc.
    key_challenges: list[str]       # Anticipated difficulties
    success_indicators: list[str]   # How to know each phase succeeded
    fallback_strategies: list[str]  # Alternative approaches if primary fails
    estimated_complexity: str       # "low", "medium", "high"
    timeout_allocation: dict[str, int]  # Time budget per phase

    @classmethod
    def from_cached(cls, data: dict) -> StrategicPlan:
        """Reconstruct from cached routine."""
        return cls(
            analysis=data["analysis"],
            approach=data["approach"],
            delegation_type=data["delegation_type"],
            key_challenges=data.get("key_challenges", []),
            success_indicators=data.get("success_indicators", []),
            fallback_strategies=data.get("fallback_strategies", []),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            timeout_allocation=data.get("timeout_allocation", {}),
        )
```

### System Prompts

```python
ARCHITECT_SYSTEM_PROMPT = """You are the Abstract Architect, the strategic planning tier of a hierarchical AI system.

Your role:
1. Understand the BIG PICTURE of what needs to be accomplished
2. Develop a strategic approach (not tactical implementation details)
3. Identify key challenges and success criteria
4. Prepare fallback strategies if the primary approach fails

You delegate to the Routing Dispatcher, which handles:
- Breaking tasks into subtasks
- Assigning work to specialists
- Tactical execution details

DO NOT:
- Write code or implementation details
- Decide which specialist should handle what
- Micromanage the decomposition

DO:
- Clearly articulate what success looks like
- Identify constraints that must be respected
- Anticipate what could go wrong
- Provide clear delegation instructions

Output in the exact JSON format specified."""

PLANNING_PROMPT = """Develop a strategic plan for this task:

## Problem Statement
{problem}

## Constraints
{constraints}

## Success Criteria
{success_criteria}

## Additional Context
{context}

## Output Format
Respond with a JSON object:
```json
{{
  "analysis": "Your understanding of what this task requires (2-3 sentences)",
  "approach": "Your strategic approach (1-2 sentences)",
  "delegation_type": "decompose_and_solve|direct_solve|analyze_only",
  "key_challenges": ["challenge 1", "challenge 2"],
  "success_indicators": ["indicator 1", "indicator 2"],
  "fallback_strategies": ["if primary fails, try X", "if that fails, try Y"],
  "estimated_complexity": "low|medium|high"
}}
```"""
```

## Context Manager

### Managing Strategic Context

```python
class ContextManager:
    """Manages strategic context in Beads."""

    def __init__(self, beads: BeadsClient):
        self.beads = beads

    async def initialize_task_context(
        self,
        task: ArchitectTask,
        plan: StrategicPlan,
    ) -> str:
        """Set up task context in Beads. Returns context key."""
        context_key = f"task/{task.task_id}"

        await self.beads.set(f"{context_key}/objective", {
            "problem_statement": task.problem_statement,
            "constraints": task.constraints,
            "success_criteria": task.success_criteria,
            "priority": task.priority,
            "created_at": datetime.now().isoformat(),
        })

        await self.beads.set(f"{context_key}/strategy", {
            "analysis": plan.analysis,
            "approach": plan.approach,
            "delegation_type": plan.delegation_type,
            "key_challenges": plan.key_challenges,
            "fallback_strategies": plan.fallback_strategies,
        })

        await self.beads.set(f"{context_key}/progress", {
            "status": "in_progress",
            "current_phase": "delegation",
            "revisions": 0,
            "started_at": datetime.now().isoformat(),
        })

        return context_key

    async def update_progress(
        self,
        context_key: str,
        phase: str,
        summary: str,
        confidence: float,
    ):
        """Update progress tracking."""
        progress = await self.beads.get(f"{context_key}/progress") or {}
        progress.update({
            "current_phase": phase,
            "last_update": datetime.now().isoformat(),
            "latest_summary": summary,
            "confidence": confidence,
        })
        await self.beads.set(f"{context_key}/progress", progress)

    async def record_revision(
        self,
        context_key: str,
        reason: str,
        new_strategy: str,
    ):
        """Record a strategy revision."""
        progress = await self.beads.get(f"{context_key}/progress") or {}
        revisions = progress.get("revisions", 0) + 1
        progress["revisions"] = revisions
        await self.beads.set(f"{context_key}/progress", progress)

        # Log the revision
        await self.beads.append(f"{context_key}/revision-history", {
            "revision_number": revisions,
            "reason": reason,
            "new_strategy": new_strategy,
            "timestamp": datetime.now().isoformat(),
        })

    async def finalize_context(
        self,
        context_key: str,
        status: OutcomeStatus,
        summary: str,
    ):
        """Mark task as complete and update context."""
        progress = await self.beads.get(f"{context_key}/progress") or {}
        progress.update({
            "status": status.value,
            "current_phase": "complete",
            "final_summary": summary,
            "completed_at": datetime.now().isoformat(),
        })
        await self.beads.set(f"{context_key}/progress", progress)
```

## Delegation Handler

### Building Delegations

```python
class DelegationHandler:
    """Handles delegation to Routing Dispatcher."""

    def __init__(
        self,
        dispatcher_client: DispatcherClient,
        context_manager: ContextManager,
    ):
        self.dispatcher = dispatcher_client
        self.context = context_manager

    async def delegate(
        self,
        task: ArchitectTask,
        plan: StrategicPlan,
        context_key: str,
        remaining_timeout_ms: int | None = None,
    ) -> OutcomePayload:
        """Delegate task to Routing Dispatcher and await result."""
        # Build message (matches handoff_protocol.md schema)
        message = Message(
            version="1.0.0",
            id=f"arch-{task.task_id}-{int(time.time())}",
            timestamp=datetime.now(),
            source_tier=TierLevel.ABSTRACT_ARCHITECT,
            target_tier=TierLevel.ROUTING_DISPATCHER,
            correlation_id=task.task_id,
            payload=None,  # Payload attached separately
        )

        # Build delegation payload
        payload = DelegationPayload(
            task_type=plan.delegation_type,
            objective=task.problem_statement,
            constraints=task.constraints,
            context_refs=[f"beads:{context_key}/objective"],
            priority=task.priority,
            timeout_ms=self._calculate_timeout(task, plan, remaining_timeout_ms),
            deadline_hint=None,
            specialist_hint=None,
            task_data={
                "success_criteria": task.success_criteria,
                "key_challenges": plan.key_challenges,
                "task_context": task.context,
            },
        )

        # Update context
        await self.context.update_progress(
            context_key,
            phase="awaiting_dispatcher",
            summary="Delegated to Routing Dispatcher",
            confidence=0.5,
        )

        # Send delegation and await outcome
        try:
            outcome = await self.dispatcher.delegate(message, payload)
            return outcome
        except asyncio.TimeoutError:
            return OutcomePayload(
                status=OutcomeStatus.TIMEOUT,
                summary="Dispatcher did not respond in time",
                result_refs=[],
                confidence=0.0,
                execution_time_ms=task.timeout_ms,
                resources_used={},
                surprise_flag=True,
                surprise_reason="Dispatcher timeout",
                error_type="TIMEOUT",
                error_detail=f"No response after {task.timeout_ms}ms",
                recoverable=True,
                artifacts=[],
            )

    def _calculate_timeout(
        self,
        task: ArchitectTask,
        plan: StrategicPlan,
        remaining_ms: int | None = None,
    ) -> int:
        """Calculate timeout for dispatcher delegation."""
        # Use remaining time if tracking across revisions, else use task timeout
        base_timeout = remaining_ms if remaining_ms is not None else task.timeout_ms

        # Reserve overhead for Architect (more for complex tasks that may need revisions)
        overhead_factors = {"low": 0.05, "medium": 0.10, "high": 0.20}
        overhead = overhead_factors.get(plan.estimated_complexity, 0.10)

        available = int(base_timeout * (1 - overhead))

        # Ensure minimum timeout
        return max(available, 5000)  # At least 5 seconds
```

## Outcome Handler

### Processing Dispatcher Results

```python
class OutcomeHandler:
    """Processes outcomes from Routing Dispatcher."""

    def __init__(
        self,
        context_manager: ContextManager,
        planner: StrategyPlanner,
        max_revisions: int = 3,
    ):
        self.context = context_manager
        self.planner = planner
        self.max_revisions = max_revisions

    async def handle_outcome(
        self,
        task: ArchitectTask,
        plan: StrategicPlan,
        outcome: OutcomePayload,
        context_key: str,
        revision_count: int,
    ) -> tuple[ArchitectResult | None, StrategicPlan | None]:
        """
        Process outcome and decide next action.
        Returns (result, None) if done, or (None, new_plan) if retrying.
        """
        # Update context with outcome
        await self.context.update_progress(
            context_key,
            phase="evaluating_outcome",
            summary=outcome.summary,
            confidence=outcome.confidence,
        )

        # Success path
        if outcome.status == OutcomeStatus.SUCCESS:
            return await self._handle_success(task, outcome, context_key, revision_count), None

        # Partial success - evaluate if acceptable
        if outcome.status == OutcomeStatus.PARTIAL:
            return await self._handle_partial(
                task, plan, outcome, context_key, revision_count
            )

        # Failure path - attempt recovery
        if outcome.status in (OutcomeStatus.FAILED, OutcomeStatus.TIMEOUT):
            return await self._handle_failure(
                task, plan, outcome, context_key, revision_count
            )

        # Needs clarification - we can't clarify, so fail
        if outcome.status == OutcomeStatus.NEEDS_CLARIFICATION:
            return self._build_clarification_result(task, outcome, context_key), None

        # Unknown status
        return self._build_error_result(
            task, f"Unknown outcome status: {outcome.status}", context_key
        ), None

    async def _handle_success(
        self,
        task: ArchitectTask,
        outcome: OutcomePayload,
        context_key: str,
        revision_count: int,
    ) -> ArchitectResult:
        """Process successful outcome."""
        await self.context.finalize_context(
            context_key, OutcomeStatus.SUCCESS, outcome.summary
        )

        # Extract solution from artifacts
        solution = self._extract_solution(outcome.artifacts)

        # Cache successful routine if confidence is high
        if outcome.confidence >= 0.8:
            await self._cache_routine(task, context_key)

        return ArchitectResult(
            task_id=task.task_id,
            status=OutcomeStatus.SUCCESS,
            summary=outcome.summary,
            solution=solution,
            artifacts=outcome.artifacts,
            confidence=outcome.confidence,
            execution_time_ms=outcome.execution_time_ms,
            strategy_revisions=revision_count,
            surprise_flag=outcome.surprise_flag,
            surprise_reason=outcome.surprise_reason,
            error_info=None,
        )

    async def _handle_partial(
        self,
        task: ArchitectTask,
        plan: StrategicPlan,
        outcome: OutcomePayload,
        context_key: str,
        revision_count: int,
    ) -> tuple[ArchitectResult | None, StrategicPlan | None]:
        """Handle partial success - decide whether to accept or retry."""
        # If confidence is reasonable, accept partial result
        if outcome.confidence >= 0.6:
            await self.context.finalize_context(
                context_key, OutcomeStatus.PARTIAL, outcome.summary
            )
            return ArchitectResult(
                task_id=task.task_id,
                status=OutcomeStatus.PARTIAL,
                summary=outcome.summary,
                solution=self._extract_solution(outcome.artifacts),
                artifacts=outcome.artifacts,
                confidence=outcome.confidence,
                execution_time_ms=outcome.execution_time_ms,
                strategy_revisions=revision_count,
                surprise_flag=outcome.surprise_flag,
                surprise_reason=outcome.surprise_reason,
                error_info=None,
            ), None

        # Low confidence partial - try to improve if we have revisions left
        if revision_count < self.max_revisions and len(plan.fallback_strategies) > 0:
            new_plan = await self._revise_strategy(
                task, plan, outcome, context_key, revision_count
            )
            return None, new_plan

        # Accept partial result
        await self.context.finalize_context(
            context_key, OutcomeStatus.PARTIAL, outcome.summary
        )
        return ArchitectResult(
            task_id=task.task_id,
            status=OutcomeStatus.PARTIAL,
            summary=f"Partial result after {revision_count} revisions: {outcome.summary}",
            solution=self._extract_solution(outcome.artifacts),
            artifacts=outcome.artifacts,
            confidence=outcome.confidence,
            execution_time_ms=outcome.execution_time_ms,
            strategy_revisions=revision_count,
            surprise_flag=True,
            surprise_reason="Could not achieve full success",
            error_info=None,
        ), None

    async def _handle_failure(
        self,
        task: ArchitectTask,
        plan: StrategicPlan,
        outcome: OutcomePayload,
        context_key: str,
        revision_count: int,
    ) -> tuple[ArchitectResult | None, StrategicPlan | None]:
        """Handle failure - attempt strategy revision."""
        if revision_count < self.max_revisions and len(plan.fallback_strategies) > 0:
            new_plan = await self._revise_strategy(
                task, plan, outcome, context_key, revision_count
            )
            return None, new_plan

        # No more revisions - return failure
        await self.context.finalize_context(
            context_key, OutcomeStatus.FAILED, outcome.summary
        )
        return ArchitectResult(
            task_id=task.task_id,
            status=OutcomeStatus.FAILED,
            summary=f"Failed after {revision_count} strategy revisions",
            solution=None,
            artifacts=outcome.artifacts,
            confidence=0.0,
            execution_time_ms=outcome.execution_time_ms,
            strategy_revisions=revision_count,
            surprise_flag=True,
            surprise_reason=outcome.error_detail or "Unknown failure",
            error_info=ErrorInfo(
                code=outcome.error_type or "UNKNOWN",
                message=outcome.error_detail or "Task failed",
                recoverable=False,
                attempted_strategies=[s for s in plan.fallback_strategies[:revision_count]],
            ),
        ), None

    async def _revise_strategy(
        self,
        task: ArchitectTask,
        plan: StrategicPlan,
        outcome: OutcomePayload,
        context_key: str,
        revision_count: int,
    ) -> StrategicPlan:
        """Revise strategy based on failure."""
        # Pick next fallback strategy
        fallback_index = min(revision_count, len(plan.fallback_strategies) - 1)
        fallback = plan.fallback_strategies[fallback_index]

        await self.context.record_revision(
            context_key,
            reason=f"Previous attempt failed: {outcome.summary}",
            new_strategy=fallback,
        )

        # Generate new plan using fallback
        return StrategicPlan(
            analysis=f"Revision {revision_count + 1}: {plan.analysis}",
            approach=fallback,
            delegation_type=plan.delegation_type,
            key_challenges=plan.key_challenges + [outcome.error_detail or "Previous failure"],
            success_indicators=plan.success_indicators,
            fallback_strategies=plan.fallback_strategies[fallback_index + 1:],
            estimated_complexity="high",  # Revisions are harder
            timeout_allocation=plan.timeout_allocation,
        )

    def _extract_solution(self, artifacts: list) -> str | None:
        """Extract solution from artifacts."""
        for artifact in artifacts:
            if artifact.artifact_type == "code":
                return artifact.inline_content or f"See: {artifact.content_ref}"
            if artifact.artifact_type == "solution":
                return artifact.inline_content or f"See: {artifact.content_ref}"
        return None

    def _build_clarification_result(
        self,
        task: ArchitectTask,
        outcome: OutcomePayload,
        context_key: str,
    ) -> ArchitectResult:
        """Build result for NEEDS_CLARIFICATION status."""
        return ArchitectResult(
            task_id=task.task_id,
            status=OutcomeStatus.FAILED,
            summary=f"Task requires clarification: {outcome.error_detail}",
            solution=None,
            artifacts=outcome.artifacts,
            confidence=0.0,
            execution_time_ms=outcome.execution_time_ms,
            strategy_revisions=0,
            surprise_flag=True,
            surprise_reason="Ambiguous task requirements",
            error_info=ErrorInfo(
                code="NEEDS_CLARIFICATION",
                message=outcome.error_detail or "Task requirements are ambiguous",
                recoverable=True,
                attempted_strategies=[],
            ),
        )

    def _build_error_result(
        self,
        task: ArchitectTask,
        error_message: str,
        context_key: str,
    ) -> ArchitectResult:
        """Build result for unexpected errors."""
        return ArchitectResult(
            task_id=task.task_id,
            status=OutcomeStatus.FAILED,
            summary=f"Unexpected error: {error_message}",
            solution=None,
            artifacts=[],
            confidence=0.0,
            execution_time_ms=0,
            strategy_revisions=0,
            surprise_flag=True,
            surprise_reason=error_message,
            error_info=ErrorInfo(
                code="UNEXPECTED_ERROR",
                message=error_message,
                recoverable=False,
                attempted_strategies=[],
            ),
        )

    async def _cache_routine(self, task: ArchitectTask, context_key: str):
        """Cache successful approach as a routine."""
        signature = self.planner.compute_signature(task)
        strategy = await self.context.beads.get(f"{context_key}/strategy")

        if strategy:
            await self.context.beads.set(f"history/routines/{signature}", {
                **strategy,
                "success_rate": 1.0,
                "usage_count": 1,
                "cached_at": datetime.now().isoformat(),
            })
```

## Main Architect Class

```python
class AbstractArchitect:
    """The strategic tier - orchestrates task execution."""

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        dispatcher_client: DispatcherClient,
        beads: BeadsClient,
        max_revisions: int = 3,
    ):
        self.beads = beads
        self.planner = StrategyPlanner(llm_client, beads)
        self.context = ContextManager(beads)
        self.delegation = DelegationHandler(dispatcher_client, self.context)
        self.outcome_handler = OutcomeHandler(self.context, self.planner, max_revisions)

    async def execute(self, task: ArchitectTask) -> ArchitectResult:
        """Execute a task through strategic planning and delegation."""
        start_time = time.monotonic()
        revision_count = 0
        remaining_timeout_ms = task.timeout_ms

        # Phase 1: Strategic Planning
        plan = await self.planner.plan(task)

        # Phase 2: Initialize Context
        context_key = await self.context.initialize_task_context(task, plan)

        # Phase 3: Execution Loop (with revision support)
        while True:
            # Calculate remaining time
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            remaining_timeout_ms = task.timeout_ms - elapsed_ms

            if remaining_timeout_ms <= 0:
                # Out of time
                return ArchitectResult(
                    task_id=task.task_id,
                    status=OutcomeStatus.TIMEOUT,
                    summary=f"Task timed out after {revision_count} revisions",
                    solution=None,
                    artifacts=[],
                    confidence=0.0,
                    execution_time_ms=elapsed_ms,
                    strategy_revisions=revision_count,
                    surprise_flag=True,
                    surprise_reason="Ran out of time",
                    error_info=ErrorInfo(
                        code="TIMEOUT",
                        message=f"Exceeded {task.timeout_ms}ms budget",
                        recoverable=False,
                        attempted_strategies=[],
                    ),
                )

            # Delegate to Dispatcher with remaining timeout
            outcome = await self.delegation.delegate(
                task, plan, context_key, remaining_timeout_ms
            )

            # Handle outcome
            result, new_plan = await self.outcome_handler.handle_outcome(
                task, plan, outcome, context_key, revision_count
            )

            if result is not None:
                # We're done
                result.execution_time_ms = int((time.monotonic() - start_time) * 1000)
                return result

            if new_plan is not None:
                # Retry with revised strategy
                plan = new_plan
                revision_count += 1
                continue

            # Shouldn't reach here
            break

        # Fallback error
        return ArchitectResult(
            task_id=task.task_id,
            status=OutcomeStatus.FAILED,
            summary="Unexpected execution path",
            solution=None,
            artifacts=[],
            confidence=0.0,
            execution_time_ms=int((time.monotonic() - start_time) * 1000),
            strategy_revisions=revision_count,
            surprise_flag=True,
            surprise_reason="Execution loop exited unexpectedly",
            error_info=ErrorInfo(
                code="INTERNAL_ERROR",
                message="Architect execution loop exited without result",
                recoverable=False,
                attempted_strategies=[],
            ),
        )

    async def cancel(self, task_id: str, reason: str) -> bool:
        """Cancel an in-progress task."""
        context_key = f"task/{task_id}"

        # Check if task exists and is in progress
        progress = await self.beads.get(f"{context_key}/progress")
        if not progress or progress.get("status") != "in_progress":
            return False

        # Send cancellation to Dispatcher
        await self.delegation.dispatcher.cancel(task_id, reason, cascade=True)

        # Update context
        await self.context.finalize_context(
            context_key, OutcomeStatus.CANCELLED, f"Cancelled: {reason}"
        )

        return True
```

## Dispatcher Client Interface

```python
class DispatcherClient:
    """Client interface for communicating with Routing Dispatcher."""

    async def delegate(
        self,
        message: Message,
        payload: DelegationPayload,
    ) -> OutcomePayload:
        """Send delegation and await outcome."""
        raise NotImplementedError

    async def cancel(
        self,
        correlation_id: str,
        reason: str,
        cascade: bool,
    ) -> bool:
        """Cancel in-flight work."""
        raise NotImplementedError
```

## Beads Integration

### State Namespaces

```python
# Architect writes to task/* namespace
ARCHITECT_BEADS_KEYS = {
    "objective": "task/{task_id}/objective",
    "strategy": "task/{task_id}/strategy",
    "progress": "task/{task_id}/progress",
    "revisions": "task/{task_id}/revision-history",
}

# Architect reads from these namespaces (written by other tiers)
ARCHITECT_READS = {
    "routing": "routing/*",          # Dispatcher's task graphs
    "execution": "execution/*",      # Fleet's outputs
    "system": "system/*",           # System state from Progress Monitor
    "history": "history/*",         # Cached routines and context
}
```

## Configuration

```yaml
# architect_config.yaml
planning:
  model: "base"              # Frozen QwQ-32B
  temperature: 0.3           # Slightly creative for strategy
  max_strategies: 3          # Number of fallback strategies to generate

execution:
  max_revisions: 3           # Maximum strategy revision attempts
  timeout_reserve: 0.1       # Reserve 10% of timeout for overhead
  partial_acceptance_threshold: 0.6  # Accept partial if confidence >= this

caching:
  routine_cache_threshold: 0.8  # Cache routines with success >= this
  signature_algorithm: "md5"    # For routine matching
```

## Usage Example

```python
async def main():
    # Setup
    llm_client = AsyncOpenAI(base_url="http://localhost:8000/v1")
    dispatcher = DispatcherClient(...)  # Implementation TBD
    beads = BeadsClient()

    # Create Architect
    architect = AbstractArchitect(
        llm_client=llm_client,
        dispatcher_client=dispatcher,
        beads=beads,
        max_revisions=3,
    )

    # Define task
    task = ArchitectTask(
        task_id="task-001",
        problem_statement="""
            Implement a function that finds the longest palindromic substring
            in a given string. The function should handle edge cases and be
            efficient for strings up to 1000 characters.
        """,
        constraints=[
            "Time complexity should be O(n²) or better",
            "Space complexity should be O(1) if possible",
            "Handle empty strings and single characters",
        ],
        success_criteria=[
            "All provided test cases pass",
            "Edge cases handled correctly",
            "Solution is readable and well-documented",
        ],
        context={
            "test_cases": [
                {"input": "babad", "expected": "bab"},
                {"input": "cbbd", "expected": "bb"},
                {"input": "", "expected": ""},
            ],
            "language": "python",
        },
        priority=1,
        timeout_ms=120000,
    )

    # Execute
    result = await architect.execute(task)

    print(f"Status: {result.status}")
    print(f"Summary: {result.summary}")
    print(f"Confidence: {result.confidence}")
    print(f"Strategy Revisions: {result.strategy_revisions}")

    if result.solution:
        print(f"\nSolution:\n{result.solution}")
```

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_strategy_planning():
    planner = StrategyPlanner(mock_client, mock_beads)
    task = ArchitectTask(
        task_id="test-1",
        problem_statement="Implement binary search",
        constraints=["O(log n) time"],
        success_criteria=["All tests pass"],
        context={},
        priority=1,
        timeout_ms=60000,
    )

    plan = await planner.plan(task)

    assert plan.analysis
    assert plan.approach
    assert plan.delegation_type in ["decompose_and_solve", "direct_solve", "analyze_only"]

@pytest.mark.asyncio
async def test_outcome_handling_success():
    handler = OutcomeHandler(mock_context, mock_planner)
    outcome = OutcomePayload(
        status=OutcomeStatus.SUCCESS,
        summary="Task completed successfully",
        result_refs=["beads:result/1"],
        confidence=0.95,
        execution_time_ms=5000,
        resources_used={},
        surprise_flag=False,
        surprise_reason=None,
        error_type=None,
        error_detail=None,
        recoverable=None,
        artifacts=[],
    )

    result, new_plan = await handler.handle_outcome(
        mock_task, mock_plan, outcome, "task/test-1", 0
    )

    assert result is not None
    assert result.status == OutcomeStatus.SUCCESS
    assert new_plan is None

@pytest.mark.asyncio
async def test_strategy_revision_on_failure():
    handler = OutcomeHandler(mock_context, mock_planner, max_revisions=3)
    plan = StrategicPlan(
        analysis="Test",
        approach="Primary approach",
        delegation_type="decompose_and_solve",
        key_challenges=[],
        success_indicators=[],
        fallback_strategies=["Try approach B", "Try approach C"],
        estimated_complexity="medium",
        timeout_allocation={},
    )
    outcome = OutcomePayload(
        status=OutcomeStatus.FAILED,
        summary="Primary approach failed",
        result_refs=[],
        confidence=0.0,
        execution_time_ms=5000,
        resources_used={},
        surprise_flag=True,
        surprise_reason="Unexpected failure",
        error_type="EXECUTION_ERROR",
        error_detail="Could not complete",
        recoverable=True,
        artifacts=[],
    )

    result, new_plan = await handler.handle_outcome(
        mock_task, plan, outcome, "task/test-1", 0
    )

    assert result is None  # Should retry
    assert new_plan is not None
    assert "Try approach B" in new_plan.approach
```

### Integration Tests

```python
@pytest.mark.integration
async def test_architect_end_to_end():
    architect = AbstractArchitect(...)
    task = ArchitectTask(
        task_id="integration-test-1",
        problem_statement="Add two numbers",
        constraints=[],
        success_criteria=["Returns correct sum"],
        context={"a": 2, "b": 3, "expected": 5},
        priority=1,
        timeout_ms=30000,
    )

    result = await architect.execute(task)

    assert result.status in [OutcomeStatus.SUCCESS, OutcomeStatus.PARTIAL]
    assert result.execution_time_ms > 0
```
