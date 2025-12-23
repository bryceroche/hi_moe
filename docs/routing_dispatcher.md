# Routing Dispatcher Specification

> The tactical tier - decomposes tasks and routes to specialists.

## Related Specifications

- **[Handoff Protocol](./handoff_protocol.md)**: Message, OutcomeStatus, DelegationPayload types
- **[Specialized Fleet](./specialized_fleet.md)**: FleetClient, SpecialistExecutor integration
- **[LoRA Infrastructure](./lora_infrastructure.md)**: AdapterRegistry, adapter management

## Required Imports

```python
from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from openai import AsyncOpenAI

# From other specs
from handoff_protocol import Message, OutcomeStatus
from lora_infrastructure import AdapterRegistry
from specialized_fleet import FleetClient
```

## Overview

The Routing Dispatcher is the middle tier of the hi_moe hierarchy. It receives high-level objectives from the Abstract Architect, breaks them into executable subtasks, routes each to appropriate specialists, and aggregates results.

```
Abstract Architect
       │
       │ DelegationPayload (task_type="decompose_and_solve")
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROUTING DISPATCHER                        │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    Task      │    │   Routing    │    │    Graph     │  │
│  │ Decomposer   │───▶│   Engine     │───▶│  Executor    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │           │
│         ▼                   ▼                   ▼           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Task Graph (DAG)                   │  │
│  │   ┌───┐     ┌───┐     ┌───┐                          │  │
│  │   │ A │────▶│ B │────▶│ D │                          │  │
│  │   └───┘     └───┘     └───┘                          │  │
│  │     │                   ▲                             │  │
│  │     │       ┌───┐       │                             │  │
│  │     └──────▶│ C │───────┘                             │  │
│  │             └───┘                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
       │
       │ Multiple DelegationPayloads to Fleet
       ▼
Specialized Fleet
```

## Dispatcher Interface

### Input: From Architect

```python
@dataclass
class DispatcherTask:
    """What the Dispatcher receives from Architect."""
    message: Message                    # Full message envelope
    task_type: str                      # "decompose_and_solve", "route_single", etc.
    objective: str                      # High-level goal
    constraints: list[str]              # Boundaries from Architect
    context_refs: list[str]             # Beads keys for context
    priority: int                       # Inherited priority
    timeout_ms: int                     # Total time budget
```

### Output: To Architect

```python
@dataclass
class DispatcherOutcome:
    """What the Dispatcher returns to Architect."""
    status: OutcomeStatus               # Aggregated status
    summary: str                        # What happened overall
    subtask_outcomes: list[SubtaskSummary]  # Per-subtask results
    artifacts: list[Artifact]           # Combined outputs
    confidence: float                   # Aggregated confidence
    execution_time_ms: int              # Total time
    resources_used: AggregatedMetrics   # Combined resource usage
    surprise_flag: bool
    surprise_reason: str | None
    error_info: ErrorInfo | None
```

```python
@dataclass
class SubtaskSummary:
    """Summary of one subtask execution."""
    subtask_id: str
    description: str
    specialist: str
    status: OutcomeStatus
    confidence: float
    execution_time_ms: int

@dataclass
class SubtaskOutcome:
    """Full outcome from executing a subtask (internal use)."""
    subtask_id: str
    description: str
    specialist: str
    status: OutcomeStatus
    confidence: float
    execution_time_ms: int
    artifacts: list[Artifact]
    error_info: ErrorInfo | None = None

@dataclass
class Artifact:
    """Output artifact from a subtask."""
    type: str                       # "code", "analysis", "test_result"
    content: str
    metadata: dict[str, Any]

@dataclass
class ErrorInfo:
    """Error details for failed subtasks."""
    code: str
    message: str
    subtask_id: str | None = None
    recoverable: bool = False

@dataclass
class AggregatedMetrics:
    """Combined resource usage across subtasks."""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    specialists_used: list[str]

@dataclass
class FleetDelegation:
    """Delegation payload sent to Fleet for subtask execution."""
    message: Message
    task_type: str
    objective: str
    constraints: list[str]
    context: dict[str, Any]
    specialist_hint: str
    timeout_ms: int
```

> **Note**: This spec extends `OutcomeStatus` with `TIMEOUT` and `SKIPPED` values.
> Update handoff_protocol.md to add these, or map TIMEOUT→FAILED, SKIPPED→CANCELLED.

## Task Decomposition

### Decomposer Design

The Task Decomposer analyzes objectives and breaks them into subtasks.

```python
@dataclass
class DecompositionPlan:
    """Parsed output from LLM decomposition."""
    analysis: str
    subtasks: list[dict]
    execution_order: str  # "parallel", "sequential", "mixed"

class TaskDecomposer:
    """Breaks high-level objectives into subtask DAGs."""

    def __init__(
        self,
        llm_client: AsyncOpenAI,
        beads: BeadsClient,
        max_subtasks: int = 10,
    ):
        self.client = llm_client
        self.beads = beads
        self.max_subtasks = max_subtasks

    async def decompose(self, task: DispatcherTask) -> TaskGraph:
        """Decompose objective into subtask graph."""
        # Use LLM to analyze and decompose
        decomposition = await self._analyze_objective(task)

        # Enforce max subtasks limit
        if len(decomposition.subtasks) > self.max_subtasks:
            decomposition.subtasks = decomposition.subtasks[:self.max_subtasks]

        # Build graph from decomposition
        graph = self._build_graph(decomposition, task.timeout_ms)

        # Validate graph (no cycles, all deps exist)
        self._validate_graph(graph)

        return graph

    async def _analyze_objective(self, task: DispatcherTask) -> DecompositionPlan:
        """Use LLM to plan decomposition."""
        prompt = DECOMPOSITION_PROMPT.format(
            objective=task.objective,
            constraints="\n".join(f"- {c}" for c in task.constraints),
            context=await self._resolve_context(task.context_refs),
        )

        response = await self.client.chat.completions.create(
            model="base",  # Use base model for decomposition
            messages=[
                {"role": "system", "content": DECOMPOSER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        return self._parse_decomposition(response.choices[0].message.content)

    async def _resolve_context(self, context_refs: list[str]) -> str:
        """Resolve context references from Beads."""
        context_parts = []
        for ref in context_refs:
            if ref.startswith("beads:"):
                key = ref[6:]  # Strip "beads:" prefix
                data = await self.beads.get(key)
                if data:
                    context_parts.append(f"[{key}]: {json.dumps(data)}")
        return "\n".join(context_parts) if context_parts else "(no context)"

    def _parse_decomposition(self, content: str) -> DecompositionPlan:
        """Parse LLM response into DecompositionPlan."""
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
        return DecompositionPlan(
            analysis=data.get("analysis", ""),
            subtasks=data.get("subtasks", []),
            execution_order=data.get("execution_order", "mixed"),
        )

    def _build_graph(self, plan: DecompositionPlan, total_timeout_ms: int) -> TaskGraph:
        """Build TaskGraph from decomposition plan."""
        subtasks: dict[str, Subtask] = {}
        all_ids = {s["id"] for s in plan.subtasks}

        for s in plan.subtasks:
            # Filter out invalid dependencies
            valid_deps = [d for d in s.get("depends_on", []) if d in all_ids]

            subtasks[s["id"]] = Subtask(
                id=s["id"],
                description=s["description"],
                task_type=s.get("task_type", "execute_code"),
                domain_hints=s.get("domain_hints", []),
                depends_on=valid_deps,
                estimated_complexity=s.get("estimated_complexity", "medium"),
                timeout_ms=0,  # Allocated later
            )

        # Find roots (no dependencies) and leaves (nothing depends on them)
        root_ids = [id for id, s in subtasks.items() if not s.depends_on]
        depended_on = set()
        for s in subtasks.values():
            depended_on.update(s.depends_on)
        leaf_ids = [id for id in subtasks if id not in depended_on]

        return TaskGraph(
            subtasks=subtasks,
            root_ids=root_ids,
            leaf_ids=leaf_ids,
        )

    def _validate_graph(self, graph: TaskGraph) -> None:
        """Validate graph has no cycles and all deps exist."""
        # Check all dependencies exist
        for subtask in graph.subtasks.values():
            for dep_id in subtask.depends_on:
                if dep_id not in graph.subtasks:
                    raise ValueError(f"Subtask {subtask.id} depends on non-existent {dep_id}")

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            for dep_id in graph.subtasks[node_id].depends_on:
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

            rec_stack.remove(node_id)
            return False

        for node_id in graph.subtasks:
            if node_id not in visited:
                if has_cycle(node_id):
                    raise ValueError("Task graph contains a cycle")
```

### Decomposition Prompt

```python
DECOMPOSER_SYSTEM_PROMPT = """You are a task decomposition specialist. Your job is to break down complex programming tasks into smaller, independent subtasks that can be executed by specialists.

Rules:
1. Each subtask should be completable by a single specialist
2. Identify dependencies between subtasks (what must complete before what)
3. Maximize parallelism where possible
4. Keep subtasks focused and atomic
5. Output in the exact JSON format specified"""

DECOMPOSITION_PROMPT = """Break down this objective into subtasks:

## Objective
{objective}

## Constraints
{constraints}

## Context
{context}

## Output Format
Respond with a JSON object:
```json
{{
  "analysis": "Brief analysis of the task (1-2 sentences)",
  "subtasks": [
    {{
      "id": "subtask_1",
      "description": "What this subtask accomplishes",
      "task_type": "execute_code|execute_test|execute_analysis|execute_debug",
      "domain_hints": ["python", "testing"],
      "depends_on": [],
      "estimated_complexity": "low|medium|high"
    }}
  ],
  "execution_order": "parallel|sequential|mixed"
}}
```"""
```

### Task Graph

```python
@dataclass
class Subtask:
    """A single unit of work in the task graph."""
    id: str
    description: str
    task_type: str
    domain_hints: list[str]         # Hints for routing
    depends_on: list[str]           # Subtask IDs this depends on
    estimated_complexity: str       # low, medium, high
    timeout_ms: int                 # Allocated timeout
    specialist: str | None = None   # Assigned after routing
    status: str = "pending"         # pending, running, completed, failed, skipped
    retry_count: int = 0            # Number of retry attempts

@dataclass
class TaskGraph:
    """DAG of subtasks with dependencies."""
    subtasks: dict[str, Subtask]    # id -> Subtask
    root_ids: list[str]             # Subtasks with no dependencies
    leaf_ids: list[str]             # Subtasks nothing depends on

    def get_ready(self) -> list[Subtask]:
        """Get subtasks ready to execute (deps satisfied)."""
        ready = []
        for subtask in self.subtasks.values():
            if subtask.status != "pending":
                continue

            # Check dependency statuses
            all_completed = True
            any_failed = False
            for dep_id in subtask.depends_on:
                dep_status = self.subtasks[dep_id].status
                if dep_status in ("failed", "skipped"):
                    any_failed = True
                    break
                elif dep_status != "completed":
                    all_completed = False

            # If any dependency failed, skip this subtask (cascading failure)
            if any_failed:
                subtask.status = "skipped"
                continue

            if all_completed:
                ready.append(subtask)
        return ready

    def mark_completed(self, subtask_id: str):
        """Mark a subtask as completed."""
        self.subtasks[subtask_id].status = "completed"

    def mark_failed(self, subtask_id: str):
        """Mark a subtask as failed."""
        self.subtasks[subtask_id].status = "failed"

    def is_complete(self) -> bool:
        """Check if all subtasks are done (completed, failed, or skipped)."""
        return all(s.status in ("completed", "failed", "skipped") for s in self.subtasks.values())

    def get_success_rate(self) -> float:
        """Calculate completion success rate."""
        completed = sum(1 for s in self.subtasks.values() if s.status == "completed")
        total = len(self.subtasks)
        return completed / total if total > 0 else 0.0
```

## Routing Engine

### Hybrid Routing Strategy

The Dispatcher uses a two-tier routing approach:

1. **Hardcoded Rules**: Fast, deterministic routing for obvious cases
2. **Routing LoRA**: Learned routing for ambiguous cases

```python
@dataclass
class RoutingRule:
    """A hardcoded routing rule."""
    name: str
    condition: Callable[[Subtask], bool]
    specialist: str
    confidence: float

@dataclass
class RoutingDecision:
    """Record of a routing decision for training."""
    subtask_id: str
    subtask_description: str
    domain_hints: list[str]
    chosen_specialist: str
    routing_method: str  # "hardcoded", "routing_lora", "fallback"
    timestamp: datetime
    # Filled in after execution
    outcome_status: OutcomeStatus | None = None
    outcome_confidence: float | None = None

class RoutingEngine:
    """Routes subtasks to specialists using hybrid approach."""

    def __init__(
        self,
        registry: AdapterRegistry,
        routing_lora: str | None = None,
        llm_client: AsyncOpenAI | None = None,
    ):
        self.registry = registry
        self.routing_lora = routing_lora
        self.client = llm_client
        self.routing_log: list[RoutingDecision] = []

    async def route(self, subtask: Subtask) -> str:
        """Route subtask to appropriate specialist."""
        # Tier 1: Try hardcoded rules first
        specialist = self._route_by_rules(subtask)
        if specialist:
            self._log_decision(subtask, specialist, "hardcoded")
            return specialist

        # Tier 2: Use Routing LoRA for ambiguous cases
        if self.routing_lora and self.client:
            specialist = await self._route_by_lora(subtask)
            if specialist:
                self._log_decision(subtask, specialist, "routing_lora")
                return specialist

        # Fallback: best-effort domain matching
        specialist = self._route_by_domain_match(subtask)
        self._log_decision(subtask, specialist or "base", "fallback")
        return specialist or "base"

    def _log_decision(self, subtask: Subtask, specialist: str, method: str):
        """Log a routing decision for training data."""
        self.routing_log.append(RoutingDecision(
            subtask_id=subtask.id,
            subtask_description=subtask.description,
            domain_hints=subtask.domain_hints,
            chosen_specialist=specialist,
            routing_method=method,
            timestamp=datetime.now(),
        ))

    def update_decision(self, subtask_id: str, outcome: SubtaskOutcome):
        """Update routing decision with execution outcome."""
        for decision in self.routing_log:
            if decision.subtask_id == subtask_id:
                decision.outcome_status = outcome.status
                decision.outcome_confidence = outcome.confidence
                break

    def _route_by_domain_match(self, subtask: Subtask) -> str | None:
        """Fallback: match domains to available specialists."""
        active_adapters = self.registry.get_active_adapters()
        best_match = None
        best_score = 0

        for adapter in active_adapters:
            # Count overlapping domains
            overlap = len(set(subtask.domain_hints) & set(adapter.domains))
            if overlap > best_score:
                best_score = overlap
                best_match = adapter.name

        return best_match
```

### Hardcoded Routing Rules

```python
# Module-level routing rules
ROUTING_RULES: list[RoutingRule] = [
    # File extension rules
    RoutingRule(
        name="python_files",
        condition=lambda s: any(h.endswith(".py") for h in s.domain_hints),
        specialist="python-lora",
        confidence=0.95,
    ),
    RoutingRule(
        name="cuda_files",
        condition=lambda s: any(h in [".cu", ".cuh", "cuda"] for h in s.domain_hints),
        specialist="cuda-lora",
        confidence=0.95,
    ),
    RoutingRule(
        name="web_files",
        condition=lambda s: any(h in [".js", ".ts", ".tsx", ".jsx", "react", "vue"] for h in s.domain_hints),
        specialist="web-lora",
        confidence=0.90,
    ),

    # Task type rules
    RoutingRule(
        name="test_tasks",
        condition=lambda s: s.task_type == "execute_test" and "python" in s.domain_hints,
        specialist="python-lora",
        confidence=0.90,
    ),
    RoutingRule(
        name="math_proofs",
        condition=lambda s: any(h in ["proof", "theorem", "algorithm", "complexity"] for h in s.domain_hints),
        specialist="math-lora",
        confidence=0.85,
    ),

    # Keyword rules
    RoutingRule(
        name="sql_tasks",
        condition=lambda s: any(h in ["sql", "database", "query", "postgres", "mysql"] for h in s.domain_hints),
        specialist="data-lora",
        confidence=0.90,
    ),
    RoutingRule(
        name="docker_tasks",
        condition=lambda s: any(h in ["docker", "kubernetes", "k8s", "ci/cd", "devops"] for h in s.domain_hints),
        specialist="devops-lora",
        confidence=0.85,
    ),
]

# Method on RoutingEngine class
def _route_by_rules(self, subtask: Subtask) -> str | None:
    """Apply hardcoded routing rules."""
    for rule in ROUTING_RULES:
        if rule.condition(subtask):
            # Verify specialist exists and is active
            adapter = self.registry.get_active(rule.specialist)
            if adapter:
                return rule.specialist
    return None
```

### Routing LoRA

For ambiguous cases, use a trained Routing LoRA:

```python
ROUTING_PROMPT = """Given this subtask, select the best specialist.

## Subtask
Description: {description}
Task Type: {task_type}
Domain Hints: {domain_hints}

## Available Specialists
{specialists}

## Output
Respond with ONLY the specialist name, nothing else."""

# Method on RoutingEngine class
async def _route_by_lora(self, subtask: Subtask) -> str | None:
    """Use Routing LoRA for ambiguous routing decisions."""
    specialists = self.registry.get_active_adapters()
    specialist_list = "\n".join(
        f"- {a.name}: {', '.join(a.domains)}"
        for a in specialists
    )

    prompt = ROUTING_PROMPT.format(
        description=subtask.description,
        task_type=subtask.task_type,
        domain_hints=", ".join(subtask.domain_hints),
        specialists=specialist_list,
    )

    response = await self.client.chat.completions.create(
        model=self.routing_lora,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # Deterministic routing
        max_tokens=50,
    )

    chosen = response.choices[0].message.content.strip()

    # Validate choice exists
    if self.registry.get_active(chosen):
        return chosen
    return None
```

### Routing Decision Logging

Log all routing decisions for training data. The `RoutingDecision` dataclass is defined above with the `RoutingEngine`.

```python
# Method on RoutingEngine class
async def export_successful_routes(self, beads: BeadsClient):
    """Export successful hardcoded routes as training data."""
    successful = [
        d for d in self.routing_log
        if d.routing_method == "hardcoded"
        and d.outcome_status == OutcomeStatus.SUCCESS
        and d.outcome_confidence and d.outcome_confidence >= 0.8
    ]

    for decision in successful:
        await beads.append("routing/training-data", {
            "input": {
                "description": decision.subtask_description,
                "domain_hints": decision.domain_hints,
            },
            "output": decision.chosen_specialist,
            "confidence": decision.outcome_confidence,
        })
```

## Graph Executor

### Parallel Execution Engine

```python
class GraphExecutor:
    """Executes task graphs with parallel subtask execution."""

    def __init__(
        self,
        fleet_client: FleetClient,
        routing_engine: RoutingEngine,
        failure_handler: FailureHandler | None = None,
        max_parallel: int = 4,
    ):
        self.fleet = fleet_client
        self.router = routing_engine
        self.failure_handler = failure_handler
        self.max_parallel = max_parallel

    async def execute(
        self,
        graph: TaskGraph,
        parent_message: Message,
        total_timeout_ms: int,
    ) -> list[SubtaskOutcome]:
        """Execute all subtasks in dependency order."""
        outcomes: dict[str, SubtaskOutcome] = {}
        start_time = time.monotonic()

        while not graph.is_complete():
            # Check timeout
            elapsed_ms = (time.monotonic() - start_time) * 1000
            if elapsed_ms > total_timeout_ms:
                # Mark remaining as timed out
                for subtask in graph.subtasks.values():
                    if subtask.status == "pending":
                        outcomes[subtask.id] = self._timeout_outcome(subtask)
                        graph.mark_failed(subtask.id)
                break

            # Get ready subtasks
            ready = graph.get_ready()
            if not ready:
                # Deadlock or all in progress - wait
                await asyncio.sleep(0.1)
                continue

            # Execute ready subtasks in parallel (up to max_parallel)
            batch = ready[:self.max_parallel]
            tasks = [
                self._execute_subtask(subtask, parent_message, total_timeout_ms - elapsed_ms)
                for subtask in batch
            ]

            # Mark as running
            for subtask in batch:
                subtask.status = "running"

            # Await batch completion
            batch_outcomes = await asyncio.gather(*tasks, return_exceptions=True)

            # Process outcomes
            for subtask, outcome in zip(batch, batch_outcomes):
                if isinstance(outcome, Exception):
                    outcomes[subtask.id] = self._exception_outcome(subtask, outcome)
                    graph.mark_failed(subtask.id)
                else:
                    outcomes[subtask.id] = outcome
                    if outcome.status == OutcomeStatus.SUCCESS:
                        graph.mark_completed(subtask.id)
                    else:
                        # Handle failure with configured strategy
                        if self.failure_handler:
                            try:
                                handled = await self.failure_handler.handle_failure(
                                    subtask, outcome, graph, self
                                )
                                if handled is None:
                                    # Subtask will be retried, don't mark failed yet
                                    del outcomes[subtask.id]
                                    continue
                            except GraphExecutionAborted:
                                # fail_fast strategy - abort entire graph
                                raise
                        graph.mark_failed(subtask.id)

        return list(outcomes.values())

    async def _execute_subtask(
        self,
        subtask: Subtask,
        parent_message: Message,
        remaining_timeout_ms: float,
    ) -> SubtaskOutcome:
        """Execute a single subtask via Fleet."""
        # Route to specialist
        specialist = await self.router.route(subtask)
        subtask.specialist = specialist

        # Calculate subtask timeout
        subtask_timeout = min(
            subtask.timeout_ms,
            int(remaining_timeout_ms * 0.9),  # Leave buffer
        )

        # Build delegation
        delegation = FleetDelegation(
            message=self._build_child_message(parent_message, subtask),
            task_type=subtask.task_type,
            objective=subtask.description,
            constraints=[],
            context={},  # Would resolve from parent context
            specialist_hint=specialist,
            timeout_ms=subtask_timeout,
        )

        # Execute via Fleet
        fleet_outcome = await self.fleet.execute(delegation)

        # Update routing log with outcome
        self.router.update_decision(subtask.id, fleet_outcome)

        return SubtaskOutcome(
            subtask_id=subtask.id,
            description=subtask.description,
            specialist=specialist,
            status=fleet_outcome.status,
            confidence=fleet_outcome.confidence,
            execution_time_ms=fleet_outcome.execution_time_ms,
            artifacts=fleet_outcome.artifacts,
        )

    def _build_child_message(self, parent: Message, subtask: Subtask) -> Message:
        """Build child message for subtask execution."""
        return Message(
            id=f"{parent.id}:{subtask.id}",
            parent_id=parent.id,
            source_tier="dispatcher",
            target_tier="fleet",
            timestamp=datetime.now(),
            correlation_id=parent.correlation_id,
            priority=parent.priority,
            version=parent.version,
        )

    def _timeout_outcome(self, subtask: Subtask) -> SubtaskOutcome:
        """Create outcome for timed-out subtask."""
        return SubtaskOutcome(
            subtask_id=subtask.id,
            description=subtask.description,
            specialist=subtask.specialist or "unknown",
            status=OutcomeStatus.TIMEOUT,
            confidence=0.0,
            execution_time_ms=subtask.timeout_ms,
            artifacts=[],
            error_info=ErrorInfo(
                code="TIMEOUT",
                message=f"Subtask {subtask.id} exceeded timeout of {subtask.timeout_ms}ms",
                subtask_id=subtask.id,
                recoverable=True,
            ),
        )

    def _exception_outcome(self, subtask: Subtask, exc: Exception) -> SubtaskOutcome:
        """Create outcome for subtask that raised exception."""
        return SubtaskOutcome(
            subtask_id=subtask.id,
            description=subtask.description,
            specialist=subtask.specialist or "unknown",
            status=OutcomeStatus.FAILED,
            confidence=0.0,
            execution_time_ms=0,
            artifacts=[],
            error_info=ErrorInfo(
                code="EXCEPTION",
                message=str(exc),
                subtask_id=subtask.id,
                recoverable=False,
            ),
        )
```

### Timeout Budget Allocation

```python
def allocate_timeouts(graph: TaskGraph, total_timeout_ms: int) -> None:
    """Allocate timeout budgets to subtasks."""
    # Calculate critical path length
    critical_path = find_critical_path(graph)
    critical_length = len(critical_path)

    # Base timeout per critical path step
    base_per_step = total_timeout_ms / max(critical_length, 1)

    # Allocate based on complexity
    complexity_multipliers = {"low": 0.5, "medium": 1.0, "high": 2.0}

    for subtask in graph.subtasks.values():
        multiplier = complexity_multipliers.get(subtask.estimated_complexity, 1.0)
        subtask.timeout_ms = int(base_per_step * multiplier)

def find_critical_path(graph: TaskGraph) -> list[str]:
    """Find longest path through the DAG (roots to leaves)."""
    # Build reverse adjacency: who depends on this node?
    dependents: dict[str, list[str]] = {id: [] for id in graph.subtasks}
    for id, subtask in graph.subtasks.items():
        for dep_id in subtask.depends_on:
            dependents[dep_id].append(id)

    # Use dynamic programming: longest path FROM each node
    memo: dict[str, list[str]] = {}

    def longest_from(node_id: str) -> list[str]:
        if node_id in memo:
            return memo[node_id]

        children = dependents[node_id]
        if not children:
            # Leaf node
            memo[node_id] = [node_id]
            return memo[node_id]

        # Find longest path through any child
        longest_child_path = []
        for child_id in children:
            child_path = longest_from(child_id)
            if len(child_path) > len(longest_child_path):
                longest_child_path = child_path

        memo[node_id] = [node_id] + longest_child_path
        return memo[node_id]

    # Find longest path starting from any root
    longest_path = []
    for root_id in graph.root_ids:
        path = longest_from(root_id)
        if len(path) > len(longest_path):
            longest_path = path

    return longest_path
```

## Result Aggregation

### Aggregator Design

```python
class ResultAggregator:
    """Aggregates subtask outcomes into dispatcher outcome."""

    def __init__(self, strategy: str = "all_success"):
        self.strategy = strategy

    def aggregate(
        self,
        outcomes: list[SubtaskOutcome],
        graph: TaskGraph,
        total_execution_time_ms: int,
    ) -> DispatcherOutcome:
        """Aggregate subtask outcomes into final result."""
        # Determine overall status
        status = self._aggregate_status(outcomes)

        # Aggregate confidence
        confidence = self._aggregate_confidence(outcomes)

        # Combine artifacts
        artifacts = self._combine_artifacts(outcomes)

        # Build summaries
        summaries = [
            SubtaskSummary(
                subtask_id=o.subtask_id,
                description=o.description,
                specialist=o.specialist,
                status=o.status,
                confidence=o.confidence,
                execution_time_ms=o.execution_time_ms,
            )
            for o in outcomes
        ]

        # Check for surprises
        surprise_flag, surprise_reason = self._check_surprise(outcomes, graph)

        return DispatcherOutcome(
            status=status,
            summary=self._build_summary(outcomes, graph),
            subtask_outcomes=summaries,
            artifacts=artifacts,
            confidence=confidence,
            execution_time_ms=total_execution_time_ms,
            resources_used=self._aggregate_resources(outcomes),
            surprise_flag=surprise_flag,
            surprise_reason=surprise_reason,
            error_info=self._aggregate_errors(outcomes) if status == OutcomeStatus.FAILED else None,
        )

    def _aggregate_status(self, outcomes: list[SubtaskOutcome]) -> OutcomeStatus:
        """Determine overall status based on strategy."""
        if not outcomes:
            return OutcomeStatus.FAILED  # No work done = failure

        statuses = [o.status for o in outcomes]

        if self.strategy == "all_success":
            if all(s == OutcomeStatus.SUCCESS for s in statuses):
                return OutcomeStatus.SUCCESS
            elif any(s == OutcomeStatus.SUCCESS for s in statuses):
                return OutcomeStatus.PARTIAL
            else:
                return OutcomeStatus.FAILED

        elif self.strategy == "any_success":
            if any(s == OutcomeStatus.SUCCESS for s in statuses):
                return OutcomeStatus.SUCCESS
            else:
                return OutcomeStatus.FAILED

        elif self.strategy == "majority":
            success_count = sum(1 for s in statuses if s == OutcomeStatus.SUCCESS)
            if success_count > len(statuses) / 2:
                return OutcomeStatus.SUCCESS
            elif success_count > 0:
                return OutcomeStatus.PARTIAL
            else:
                return OutcomeStatus.FAILED

        return OutcomeStatus.PARTIAL  # Default

    def _aggregate_confidence(self, outcomes: list[SubtaskOutcome]) -> float:
        """Calculate weighted average confidence."""
        if not outcomes:
            return 0.0

        # Weight by execution time (longer tasks = more important)
        total_time = sum(o.execution_time_ms for o in outcomes)
        if total_time == 0:
            return sum(o.confidence for o in outcomes) / len(outcomes)

        weighted_sum = sum(
            o.confidence * o.execution_time_ms
            for o in outcomes
        )
        return weighted_sum / total_time

    def _combine_artifacts(self, outcomes: list[SubtaskOutcome]) -> list[Artifact]:
        """Combine artifacts from all subtasks."""
        combined = []
        for outcome in outcomes:
            for artifact in outcome.artifacts:
                # Tag with subtask origin
                artifact.metadata["subtask_id"] = outcome.subtask_id
                combined.append(artifact)
        return combined

    def _build_summary(self, outcomes: list[SubtaskOutcome], graph: TaskGraph) -> str:
        """Build human-readable summary."""
        total = len(outcomes)
        success = sum(1 for o in outcomes if o.status == OutcomeStatus.SUCCESS)
        failed = sum(1 for o in outcomes if o.status == OutcomeStatus.FAILED)

        return f"{success}/{total} subtasks completed successfully. {failed} failed."

    def _check_surprise(
        self,
        outcomes: list[SubtaskOutcome],
        graph: TaskGraph,
    ) -> tuple[bool, str | None]:
        """Check for surprising aggregate results."""
        # Empty outcomes is surprising
        if not outcomes:
            return True, "No subtasks were executed"

        # All failed is surprising
        if all(o.status == OutcomeStatus.FAILED for o in outcomes):
            return True, "All subtasks failed"

        # Very low aggregate confidence is surprising
        avg_confidence = sum(o.confidence for o in outcomes) / len(outcomes)
        if avg_confidence < 0.3:
            return True, f"Very low average confidence ({avg_confidence:.2f})"

        return False, None

    def _aggregate_resources(self, outcomes: list[SubtaskOutcome]) -> AggregatedMetrics:
        """Combine resource usage from all subtasks."""
        specialists_used = list(set(o.specialist for o in outcomes))

        # Note: actual token counts would come from Fleet outcomes
        # This is a placeholder structure
        return AggregatedMetrics(
            total_tokens=0,  # Would sum from fleet metrics
            prompt_tokens=0,
            completion_tokens=0,
            specialists_used=specialists_used,
        )

    def _aggregate_errors(self, outcomes: list[SubtaskOutcome]) -> ErrorInfo:
        """Aggregate error information from failed subtasks."""
        failed = [o for o in outcomes if o.status == OutcomeStatus.FAILED and o.error_info]

        if not failed:
            return ErrorInfo(
                code="UNKNOWN",
                message="Unknown failure",
                recoverable=False,
            )

        if len(failed) == 1:
            return failed[0].error_info

        # Multiple failures - summarize
        error_messages = [f"{o.subtask_id}: {o.error_info.message}" for o in failed]
        return ErrorInfo(
            code="MULTIPLE_FAILURES",
            message=f"{len(failed)} subtasks failed: " + "; ".join(error_messages[:3]),
            recoverable=any(o.error_info.recoverable for o in failed),
        )
```

## Error Handling

### Subtask Failure Strategies

```python
class GraphExecutionAborted(Exception):
    """Raised when graph execution is aborted due to fail_fast strategy."""
    pass

class FailureHandler:
    """Handles subtask failures with configurable strategies."""

    def __init__(
        self,
        registry: AdapterRegistry,
        strategy: str = "continue",
        max_retries: int = 2,
    ):
        self.registry = registry
        self.strategy = strategy  # "continue", "fail_fast", "retry"
        self.max_retries = max_retries

    async def handle_failure(
        self,
        subtask: Subtask,
        outcome: SubtaskOutcome,
        graph: TaskGraph,
        executor: GraphExecutor,
    ) -> SubtaskOutcome | None:
        """Handle a failed subtask based on strategy."""

        if self.strategy == "fail_fast":
            # Abort entire graph
            raise GraphExecutionAborted(f"Subtask {subtask.id} failed: {outcome.status}")

        elif self.strategy == "retry":
            # Retry with different specialist
            if subtask.retry_count < self.max_retries:
                subtask.retry_count += 1
                alternate = await self._find_alternate_specialist(subtask, outcome)
                if alternate:
                    subtask.specialist = alternate
                    subtask.status = "pending"
                    return None  # Will be re-executed

        # "continue" strategy: mark failed, continue with other subtasks
        return outcome

    async def _find_alternate_specialist(
        self,
        subtask: Subtask,
        failed_outcome: SubtaskOutcome,
    ) -> str | None:
        """Find an alternate specialist for retry."""
        # Get all candidates except the one that failed
        candidates = [
            a for a in self.registry.get_by_domains(subtask.domain_hints)
            if a.name != failed_outcome.specialist
        ]

        if candidates:
            # Pick best by success rate
            return max(candidates, key=lambda a: a.success_rate).name
        return None
```

## Beads Integration

### State Management

```python
# Dispatcher writes to these namespaces
DISPATCHER_BEADS_KEYS = {
    "task_graph": "routing/task-graph/{message_id}",
    "assignments": "routing/assignments/{message_id}",
    "pending": "routing/pending",
    "training_data": "routing/training-data",
}

class DispatcherState:
    """Manages dispatcher state in Beads."""

    def __init__(self, beads: BeadsClient):
        self.beads = beads

    async def save_task_graph(
        self,
        message_id: str,
        graph: TaskGraph,
    ):
        """Save task graph to Beads for visibility."""
        key = f"routing/task-graph/{message_id}"
        await self.beads.set(key, {
            "subtasks": {
                id: {
                    "description": s.description,
                    "task_type": s.task_type,
                    "specialist": s.specialist,
                    "status": s.status,
                    "depends_on": s.depends_on,
                }
                for id, s in graph.subtasks.items()
            },
            "root_ids": graph.root_ids,
            "leaf_ids": graph.leaf_ids,
            "created_at": datetime.now().isoformat(),
        })

    async def update_subtask_status(
        self,
        message_id: str,
        subtask_id: str,
        status: str,
    ):
        """Update a single subtask's status."""
        key = f"routing/task-graph/{message_id}"
        graph_data = await self.beads.get(key)
        if graph_data and subtask_id in graph_data.get("subtasks", {}):
            graph_data["subtasks"][subtask_id]["status"] = status
            await self.beads.set(key, graph_data)
```

## Configuration

### Dispatcher Configuration

```yaml
# dispatcher_config.yaml
decomposition:
  model: "base"  # Use base model for decomposition
  temperature: 0.2
  max_subtasks: 10  # Limit subtask count

routing:
  strategy: "hybrid"  # "hardcoded", "lora", "hybrid"
  routing_lora: "routing-lora"  # Name of routing adapter
  fallback: "base"  # Fallback when no match

execution:
  max_parallel: 4  # Max concurrent subtasks
  failure_strategy: "continue"  # "continue", "fail_fast", "retry"
  max_retries: 2

aggregation:
  strategy: "all_success"  # "all_success", "any_success", "majority"
```

## Usage Example

```python
async def main():
    # Setup
    llm_client = AsyncOpenAI(base_url="http://localhost:8000/v1")
    registry = AdapterRegistry.from_beads()
    fleet_client = FleetClient(...)
    beads = BeadsClient()

    # Create dispatcher components
    decomposer = TaskDecomposer(llm_client, beads, max_subtasks=10)
    router = RoutingEngine(registry, routing_lora="routing-lora", llm_client=llm_client)
    failure_handler = FailureHandler(registry, strategy="continue", max_retries=2)
    executor = GraphExecutor(fleet_client, router, failure_handler, max_parallel=4)
    aggregator = ResultAggregator(strategy="all_success")
    state = DispatcherState(beads)

    # Receive task from Architect
    task = DispatcherTask(
        message=incoming_message,
        task_type="decompose_and_solve",
        objective="Implement a binary search tree with insert, delete, and search operations",
        constraints=["Must handle duplicates", "O(log n) average case"],
        context_refs=["beads:task/requirements"],
        priority=1,
        timeout_ms=120000,
    )

    # Decompose into subtasks
    graph = await decomposer.decompose(task)
    allocate_timeouts(graph, task.timeout_ms)

    # Save graph for visibility
    await state.save_task_graph(task.message.id, graph)

    # Execute graph
    start_time = time.monotonic()
    outcomes = await executor.execute(graph, task.message, task.timeout_ms)
    total_time = int((time.monotonic() - start_time) * 1000)

    # Export successful routes for training
    await router.export_successful_routes(beads)

    # Aggregate results
    result = aggregator.aggregate(outcomes, graph, total_time)

    return result
```

## Testing Strategy

### Unit Tests

```python
@pytest.mark.asyncio
async def test_task_decomposition():
    decomposer = TaskDecomposer(mock_client, mock_beads, max_subtasks=10)
    graph = await decomposer.decompose(sample_task)
    assert len(graph.subtasks) > 0
    assert len(graph.root_ids) > 0  # Has starting points

@pytest.mark.asyncio
async def test_routing_hardcoded():
    router = RoutingEngine(mock_registry)
    subtask = Subtask(
        id="test_1",
        description="Write a Python function",
        task_type="execute_code",
        domain_hints=["python", ".py"],
        depends_on=[],
        estimated_complexity="medium",
        timeout_ms=30000,
    )
    specialist = await router.route(subtask)
    assert specialist == "python-lora"

@pytest.mark.asyncio
async def test_graph_execution_order():
    # Graph: A -> B, A -> C, B + C -> D
    graph = TaskGraph(
        subtasks={
            "A": Subtask(id="A", description="Task A", task_type="execute_code",
                        domain_hints=["python"], depends_on=[],
                        estimated_complexity="low", timeout_ms=10000),
            "B": Subtask(id="B", description="Task B", task_type="execute_code",
                        domain_hints=["python"], depends_on=["A"],
                        estimated_complexity="low", timeout_ms=10000),
            "C": Subtask(id="C", description="Task C", task_type="execute_code",
                        domain_hints=["python"], depends_on=["A"],
                        estimated_complexity="low", timeout_ms=10000),
            "D": Subtask(id="D", description="Task D", task_type="execute_code",
                        domain_hints=["python"], depends_on=["B", "C"],
                        estimated_complexity="low", timeout_ms=10000),
        },
        root_ids=["A"],
        leaf_ids=["D"],
    )
    executor = GraphExecutor(mock_fleet, mock_router, max_parallel=4)
    # Should execute A first, then B and C in parallel, then D
    # Verify execution order via mock call assertions
```

### Integration Tests

```python
@pytest.mark.integration
async def test_dispatcher_end_to_end():
    dispatcher = Dispatcher(...)
    task = DispatcherTask(
        objective="Write a function to reverse a string",
        ...
    )
    outcome = await dispatcher.process(task)
    assert outcome.status in [OutcomeStatus.SUCCESS, OutcomeStatus.PARTIAL]
```
