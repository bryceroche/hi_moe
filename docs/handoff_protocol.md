# Handoff Protocol Specification

> Defines how tiers communicate in the hi_moe architecture.

## Overview

Communication flows in two directions:
- **Delegation (down)**: Higher tiers assign work to lower tiers
- **Reporting (up)**: Lower tiers report outcomes to higher tiers
- **Shared State (lateral)**: All tiers read from Beads state object

## Message Schema

All messages use a common envelope:

```python
@dataclass
class Message:
    version: str                # Protocol version (e.g., "1.0.0")
    id: str                     # UUID for tracking/correlation
    timestamp: datetime         # When message was created
    source_tier: TierLevel      # Who sent this
    target_tier: TierLevel      # Who receives this
    correlation_id: str | None  # Links to originating request
    payload: DelegationPayload | OutcomePayload | CancellationPayload
```

**Version Compatibility**:
- Major version bump = breaking change, receivers MUST reject unknown major versions
- Minor version bump = additive change, receivers SHOULD accept unknown fields
- Current version: `1.0.0`

```python
class TierLevel(Enum):
    PROGRESS_MONITOR = 4    # Meta-evaluation
    ABSTRACT_ARCHITECT = 3  # Strategic planning
    ROUTING_DISPATCHER = 2  # Task decomposition
    SPECIALIZED_FLEET = 1   # Domain execution
```

## Delegation Format (Down)

When a higher tier assigns work to a lower tier:

```python
@dataclass
class DelegationPayload:
    task_type: str              # What kind of work (e.g., "solve", "route", "execute")
    objective: str              # Natural language description of goal
    constraints: list[str]      # Boundaries and requirements
    context_refs: list[str]     # Beads keys for relevant context
    priority: int               # 0=critical, 4=backlog

    # Timeout handling
    timeout_ms: int | None      # Hard timeout; receiver MUST respond before this
    deadline_hint: str | None   # Soft guidance for planning, not enforced

    # For Dispatcher → Fleet routing
    specialist_hint: str | None # Suggested LoRA if known

    # Structured task data (tier-specific)
    task_data: dict             # Flexible payload per task type
```

**Timeout Semantics**:
- If `timeout_ms` is set and exceeded, receiver MUST return `TIMEOUT` status
- Sender starts watchdog timer on delegation; if no response, treats as `TIMEOUT`
- Recommended defaults: Fleet=30000ms, Dispatcher=60000ms, Architect=120000ms

### Example: Architect → Dispatcher

```json
{
  "version": "1.0.0",
  "id": "msg-001",
  "timestamp": "2025-01-15T10:30:00Z",
  "source_tier": "ABSTRACT_ARCHITECT",
  "target_tier": "ROUTING_DISPATCHER",
  "correlation_id": null,
  "payload": {
    "task_type": "decompose_and_solve",
    "objective": "Implement binary search with edge case handling",
    "constraints": [
      "Must handle empty arrays",
      "O(log n) time complexity required"
    ],
    "context_refs": ["beads:problem-statement", "beads:test-cases"],
    "priority": 1,
    "timeout_ms": 60000,
    "deadline_hint": null,
    "specialist_hint": null,
    "task_data": {
      "problem_id": "leetcode-704",
      "language": "python"
    }
  }
}
```

### Example: Dispatcher → Fleet

```json
{
  "version": "1.0.0",
  "id": "msg-002",
  "timestamp": "2025-01-15T10:30:05Z",
  "source_tier": "ROUTING_DISPATCHER",
  "target_tier": "SPECIALIZED_FLEET",
  "correlation_id": "msg-001",
  "payload": {
    "task_type": "execute_code",
    "objective": "Write binary search function",
    "constraints": ["Return -1 if not found"],
    "context_refs": ["beads:problem-statement"],
    "priority": 1,
    "timeout_ms": 30000,
    "deadline_hint": null,
    "specialist_hint": "python-lora",
    "task_data": {
      "subtask_index": 1,
      "subtask_total": 3,
      "input_signature": "def binary_search(nums: List[int], target: int) -> int",
      "test_cases": ["beads:test-cases"]
    }
  }
}
```

## Outcome Reporting Format (Up)

When a lower tier reports results to a higher tier:

```python
@dataclass
class OutcomePayload:
    status: OutcomeStatus       # How did it go?
    summary: str                # Natural language summary
    result_refs: list[str]      # Beads keys for outputs

    # Metrics for learning
    confidence: float           # 0.0-1.0, self-assessed
    execution_time_ms: int      # How long it took
    resources_used: dict        # LoRAs loaded, tokens consumed, etc.

    # For surprising outcomes
    surprise_flag: bool         # Outside expected bounds?
    surprise_reason: str | None # Why it was surprising

    # Error details (only present if status is FAILED, BLOCKED, NEEDS_CLARIFICATION)
    error_type: str | None
    error_detail: str | None
    recoverable: bool | None    # Only set on error statuses

    # What actually happened (not just success/fail)
    artifacts: list[Artifact]   # Concrete outputs
```

```python
class OutcomeStatus(Enum):
    SUCCESS = "success"           # Task completed as requested
    PARTIAL = "partial"           # Some objectives met
    FAILED = "failed"             # Could not complete
    BLOCKED = "blocked"           # Waiting on external dependency
    NEEDS_CLARIFICATION = "needs_clarification"  # Ambiguous request
    TIMEOUT = "timeout"           # Exceeded timeout_ms
    CANCELLED = "cancelled"       # Aborted by higher tier
    THROTTLED = "throttled"       # Receiver overloaded, try later
```

```python
@dataclass
class Artifact:
    artifact_type: str          # "code", "analysis", "test_result", etc.
    content_ref: str | None     # Beads key (use for content > 1KB)
    inline_content: str | None  # Small artifacts < 1KB can be inline
    metadata: dict              # Type-specific metadata
```

**Inline vs Reference Rule**:
- Content < 1KB: use `inline_content`, set `content_ref` to None
- Content >= 1KB: store in Beads, use `content_ref`, set `inline_content` to None
- Never set both; exactly one must be non-null

## Cancellation Format

When a higher tier needs to abort in-flight work:

```python
@dataclass
class CancellationPayload:
    target_message_id: str      # The delegation message to cancel
    reason: str                 # Why we're cancelling
    cascade: bool               # Should receiver cancel its downstream delegations?
```

**Cancellation Semantics**:
- Sender issues cancellation, receiver SHOULD stop work promptly
- Receiver responds with `CANCELLED` status including any partial results
- If `cascade=True`, receiver propagates cancellation to all downstream tasks
- Cancellation is best-effort; work may complete before cancellation arrives

### Example: Architect cancels Dispatcher work

```json
{
  "version": "1.0.0",
  "id": "msg-cancel-001",
  "timestamp": "2025-01-15T10:31:00Z",
  "source_tier": "ABSTRACT_ARCHITECT",
  "target_tier": "ROUTING_DISPATCHER",
  "correlation_id": null,
  "payload": {
    "target_message_id": "msg-001",
    "reason": "Strategy revision - new approach identified",
    "cascade": true
  }
}
```

### Example: Fleet → Dispatcher

```json
{
  "version": "1.0.0",
  "id": "msg-003",
  "timestamp": "2025-01-15T10:30:45Z",
  "source_tier": "SPECIALIZED_FLEET",
  "target_tier": "ROUTING_DISPATCHER",
  "correlation_id": "msg-002",
  "payload": {
    "status": "success",
    "summary": "Implemented binary search with edge case handling. All test cases pass.",
    "result_refs": ["beads:execution/outputs/msg-002"],
    "confidence": 0.92,
    "execution_time_ms": 3200,
    "resources_used": {
      "lora": "python-lora",
      "tokens_in": 450,
      "tokens_out": 280
    },
    "surprise_flag": false,
    "surprise_reason": null,
    "error_type": null,
    "error_detail": null,
    "recoverable": null,
    "artifacts": [
      {
        "artifact_type": "code",
        "content_ref": "beads:execution/outputs/msg-002",
        "inline_content": null,
        "metadata": {
          "language": "python",
          "lines": 15,
          "complexity": "O(log n)"
        }
      },
      {
        "artifact_type": "test_result",
        "content_ref": null,
        "inline_content": "5/5 tests passed",
        "metadata": {
          "passed": 5,
          "failed": 0
        }
      }
    ]
  }
}
```

## Beads State Contract

Beads provides shared state that all tiers can read. Write access is scoped.

### Namespaces

```
beads:task/*        # Current task context (Architect writes, all read)
beads:routing/*     # Routing decisions (Dispatcher writes)
beads:execution/*   # Execution artifacts (Fleet writes)
beads:system/*      # System state (Progress Monitor writes)
beads:history/*     # Compressed historical context (PM writes)
```

### Write Permissions

| Namespace | Monitor | Architect | Dispatcher | Fleet |
|-----------|---------|-----------|------------|-------|
| task/*    | R       | RW        | R          | R     |
| routing/* | R       | R         | RW         | R     |
| execution/*| R      | R         | R          | RW    |
| system/*  | RW      | R         | R          | R     |
| history/* | RW      | R         | R          | R     |

### Key Patterns

```python
# Task context (set by Architect)
beads:task/current-objective     # What we're trying to achieve
beads:task/constraints           # Boundaries
beads:task/progress-summary      # Updated as work progresses

# Routing state (set by Dispatcher)
beads:routing/task-graph         # Decomposed subtasks as DAG
beads:routing/assignments        # Which specialist handles what
beads:routing/pending            # Queue of unassigned work

# Execution artifacts (set by Fleet)
beads:execution/outputs/{id}     # Results from specialist execution
beads:execution/logs/{id}        # Detailed execution logs

# System state (set by Progress Monitor)
beads:system/loaded-loras        # Currently active specialists
beads:system/routing-accuracy    # Recent routing success rate
beads:system/bottlenecks         # Identified performance issues

# History (set by Progress Monitor)
beads:history/context-vector     # Compressed representation of trajectory
beads:history/routines/{name}    # Cached successful patterns
```

## Parallel Execution & Aggregation

The Dispatcher can fan out work to multiple specialists in parallel. This section defines how outcomes are aggregated.

### Fan-Out Pattern

```python
@dataclass
class ParallelDelegation:
    parent_id: str              # Delegation that spawned these
    child_ids: list[str]        # Individual delegation message IDs
    aggregation_strategy: str   # How to combine results
```

### Aggregation Strategies

| Strategy | Behavior | Use When |
|----------|----------|----------|
| `all_success` | PARTIAL if any fail, SUCCESS only if all succeed | Dependent subtasks |
| `any_success` | SUCCESS if any succeed, FAILED only if all fail | Redundant attempts |
| `majority` | SUCCESS if >50% succeed | Voting/consensus |
| `first_success` | Return first SUCCESS, cancel others | Racing strategies |

### Aggregated Outcome

When Dispatcher reports upstream after parallel execution:

```python
@dataclass
class AggregatedOutcome:
    strategy_used: str
    child_outcomes: list[ChildOutcomeSummary]
    aggregated_status: OutcomeStatus
    aggregated_confidence: float  # Weighted average of child confidences
```

```python
@dataclass
class ChildOutcomeSummary:
    message_id: str
    status: OutcomeStatus
    confidence: float
    specialist: str
```

### Partial Success Rules

When using `all_success` strategy with mixed results:

1. **Majority succeeded**: Status = `PARTIAL`, include all successful artifacts
2. **Majority failed**: Status = `PARTIAL`, flag for strategy revision
3. **All failed**: Status = `FAILED`, escalate with all error details

The `PARTIAL` status signals to the Architect: "We made progress but didn't fully complete. Here's what we have."

### Example: Aggregated Outcome

```json
{
  "version": "1.0.0",
  "id": "msg-agg-001",
  "timestamp": "2025-01-15T10:32:00Z",
  "source_tier": "ROUTING_DISPATCHER",
  "target_tier": "ABSTRACT_ARCHITECT",
  "correlation_id": "msg-001",
  "payload": {
    "status": "partial",
    "summary": "2/3 subtasks completed. Test generation failed.",
    "result_refs": ["beads:routing/task-graph"],
    "confidence": 0.78,
    "execution_time_ms": 8500,
    "resources_used": {
      "parallel_tasks": 3,
      "specialists_used": ["python-lora", "test-lora"]
    },
    "surprise_flag": false,
    "surprise_reason": null,
    "error_type": "partial_completion",
    "error_detail": "test-lora failed: timeout after 30000ms",
    "recoverable": true,
    "artifacts": [
      {
        "artifact_type": "aggregation_summary",
        "content_ref": null,
        "inline_content": "{\"strategy\": \"all_success\", \"succeeded\": 2, \"failed\": 1}",
        "metadata": {
          "child_outcomes": [
            {"id": "msg-002", "status": "success", "specialist": "python-lora"},
            {"id": "msg-003", "status": "success", "specialist": "python-lora"},
            {"id": "msg-004", "status": "timeout", "specialist": "test-lora"}
          ]
        }
      }
    ]
  }
}
```

## Error Handling

### Retry Semantics

1. **Transient failures**: Retry up to 3 times with exponential backoff
2. **Specialist unavailable**: Dispatcher re-routes to alternate specialist
3. **Persistent failures**: Escalate to higher tier with `FAILED` status

### Error Escalation Path

```
Fleet failure → Dispatcher (can re-route or retry)
       ↓ (if unrecoverable)
Dispatcher failure → Architect (can revise strategy)
       ↓ (if unrecoverable)
Architect failure → Progress Monitor (can reset or abort)
```

### Clarification Protocol

When a tier receives an ambiguous request:

1. Return `NEEDS_CLARIFICATION` status immediately
2. Include specific questions in `error_detail`
3. Higher tier either clarifies or revises the delegation
4. Do NOT guess - ambiguity should flow up, not down

```json
{
  "status": "needs_clarification",
  "summary": "Ambiguous requirements for edge case handling",
  "error_type": "ambiguous_request",
  "error_detail": "Should empty array return -1 or raise exception?",
  "recoverable": true
}
```

## Message Flow Example

Complete flow for a competitive programming problem:

```
Progress Monitor
    │
    │ ① Delegates: "Solve problem X, track progress"
    ▼
Abstract Architect
    │
    │  Reads: beads:task/problem-statement
    │  Writes: beads:task/current-objective, beads:task/constraints
    │
    │ ② Delegates: "Decompose and solve binary search problem"
    ▼
Routing Dispatcher
    │
    │  Reads: beads:task/*, available specialists
    │  Writes: beads:routing/task-graph, beads:routing/assignments
    │
    │ ③ Delegates: "Execute code task" (to python-lora)
    ▼
Specialized Fleet (python-lora)
    │
    │  Reads: beads:task/*, beads:routing/assignments
    │  Writes: beads:execution/outputs/*, beads:execution/logs/*
    │
    │ ④ Reports: SUCCESS with code artifact
    ▼
Routing Dispatcher
    │
    │  Updates: beads:routing/task-graph (mark complete)
    │
    │ ⑤ Reports: SUCCESS with aggregated results
    ▼
Abstract Architect
    │
    │  Updates: beads:task/progress-summary
    │
    │ ⑥ Reports: SUCCESS with solution
    ▼
Progress Monitor
    │
    │  Updates: beads:system/*, beads:history/*
    │  Evaluates: Was this surprising? Update confidence bounds.
    │
    Done (or trigger next task)
```

## Implementation Notes

### Message Transport

For MVP, use simple function calls between tiers. Later:
- Consider message queue (Redis, RabbitMQ) for async processing
- Add tracing/observability via OpenTelemetry

### Serialization

- Use JSON for debugging visibility
- Consider msgpack for production performance
- All messages must be serializable to Beads

### Idempotency & Deduplication

- All handlers must be idempotent
- Retry same message = same outcome

**Deduplication Responsibility**:
- Each tier is responsible for deduplicating messages it receives
- Store seen `message.id` values in `beads:system/seen-messages/{tier}`
- TTL: 1 hour (messages older than this can be safely forgotten)
- On duplicate: return cached response, do not re-execute

```python
# Pseudocode for dedup check
def handle_message(msg: Message) -> Message:
    cache_key = f"beads:system/seen-messages/{my_tier}/{msg.id}"

    if beads.exists(cache_key):
        return beads.get(cache_key)  # Return cached response

    response = actually_process(msg)
    beads.set(cache_key, response, ttl=3600)
    return response
```
