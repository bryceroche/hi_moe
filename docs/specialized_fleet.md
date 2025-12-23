# Specialized Fleet Specification

> The execution tier - domain specialists that do the actual work.

## Overview

The Specialized Fleet is the bottom tier of the hi_moe hierarchy. It receives task delegations from the Routing Dispatcher and executes them using domain-specific LoRA adapters on a shared base model.

```
Routing Dispatcher
       │
       │ DelegationPayload (task_type="execute_*")
       ▼
┌─────────────────────────────────────────────────────┐
│                 SPECIALIZED FLEET                    │
│                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ python  │ │  cuda   │ │  math   │ │  web    │   │
│  │  lora   │ │  lora   │ │  lora   │ │  lora   │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       └──────────┬┴──────────┬┴───────────┘        │
│                  ▼           ▼                      │
│         ┌────────────────────────┐                  │
│         │   Qwen QwQ-32B (AWQ)   │                  │
│         │      frozen base       │                  │
│         └────────────────────────┘                  │
│                     │                               │
│                     ▼                               │
│              vLLM Server                            │
└─────────────────────────────────────────────────────┘
       │
       │ OutcomePayload (status, artifacts, metrics)
       ▼
Routing Dispatcher
```

## Fleet Interface

### Input: DelegationPayload

The Fleet receives delegations from the Dispatcher:

```python
@dataclass
class FleetDelegation:
    """What the Fleet receives from Dispatcher."""
    message: Message                    # Full message envelope
    task_type: str                      # "execute_code", "execute_analysis", etc.
    objective: str                      # What to accomplish
    constraints: list[str]              # Boundaries
    context: dict                       # Resolved context from Beads refs
    specialist_hint: str                # Which adapter to use
    timeout_ms: int                     # Hard deadline
```

### Output: OutcomePayload

The Fleet returns structured outcomes:

```python
@dataclass
class FleetOutcome:
    """What the Fleet returns to Dispatcher."""
    status: OutcomeStatus               # SUCCESS, PARTIAL, FAILED, etc.
    summary: str                        # What happened (natural language)
    artifacts: list[Artifact]           # Concrete outputs
    confidence: float                   # Self-assessed 0.0-1.0
    execution_time_ms: int              # How long it took
    resources_used: ResourceMetrics     # Token counts, adapter info
    surprise_flag: bool                 # Outside expected bounds?
    surprise_reason: str | None         # Why it was surprising
    error_info: ErrorInfo | None        # If failed
```

## Specialist Executor

### Core Execution Loop

```python
class SpecialistExecutor:
    """Executes tasks using LoRA specialists via vLLM."""

    def __init__(self, vllm_client: AsyncOpenAI, registry: AdapterRegistry):
        self.client = vllm_client
        self.registry = registry

    async def execute(self, delegation: FleetDelegation) -> FleetOutcome:
        start_time = time.monotonic()

        # 1. Resolve specialist
        specialist = self._resolve_specialist(delegation.specialist_hint)

        # 2. Build prompt
        prompt = self._build_prompt(delegation, specialist)

        # 3. Execute with timeout
        try:
            response = await asyncio.wait_for(
                self._call_vllm(specialist, prompt),
                timeout=delegation.timeout_ms / 1000
            )
        except asyncio.TimeoutError:
            return self._timeout_outcome(delegation, start_time)
        except Exception as e:
            return self._error_outcome(delegation, e, start_time)

        # 4. Parse response and build outcome
        return self._build_outcome(delegation, response, specialist, start_time)

    def _resolve_specialist(self, hint: str) -> AdapterInfo:
        """Get adapter info, falling back to base model if not found."""
        if hint:
            adapter = self.registry.get_active(hint)
            if adapter:
                return adapter

        # Fallback: no adapter (base model)
        return AdapterInfo(
            name="base",
            lora_int_id=0,
            path=None,
            rank=0,
            domains=["general"],
            base_model=self.registry.base_model,
            version="0.0.0",
            # ... other fields with defaults
        )

    async def _call_vllm(
        self,
        specialist: AdapterInfo,
        prompt: str
    ) -> ChatCompletion:
        """Call vLLM with the appropriate adapter."""
        if specialist.name == "base":
            model = self.registry.base_model
        else:
            model = specialist.name  # Use --lora-modules name

        # System prompt establishes specialist identity
        system_prompt = self._build_system_prompt(specialist)

        return await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self._get_temperature(specialist),
            max_tokens=4096,
        )

    def _build_system_prompt(self, specialist: AdapterInfo) -> str:
        """Build system prompt for specialist identity."""
        domain = specialist.domains[0] if specialist.domains else "general"
        return f"""You are a {domain} specialist with deep expertise in {', '.join(specialist.domains)}.

Your responses must be:
- Precise and technically correct
- Well-structured following the exact output format requested
- Honest about uncertainty (reflect this in your confidence score)

Always provide your reasoning before your solution."""

    def _get_temperature(self, specialist: AdapterInfo) -> float:
        """Get temperature based on specialist domain."""
        # Creative tasks get higher temperature
        creative_domains = {"docs", "documentation", "writing"}
        if any(d in creative_domains for d in specialist.domains):
            return 0.3
        # Analytical/code tasks get low temperature
        return 0.1

    def _build_outcome(
        self,
        delegation: FleetDelegation,
        response: ChatCompletion,
        specialist: AdapterInfo,
        start_time: float
    ) -> FleetOutcome:
        """Parse response and construct outcome."""
        execution_time_ms = int((time.monotonic() - start_time) * 1000)

        # Extract response text and token usage
        response_text = response.choices[0].message.content
        tokens_used = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

        # Parse response
        parser = ResponseParser()
        parsed = parser.parse_with_fallback(response_text)

        # Build outcome using OutcomeBuilder
        builder = OutcomeBuilder()
        return builder.build(
            delegation=delegation,
            parsed=parsed,
            specialist=specialist,
            execution_time_ms=execution_time_ms,
            tokens_used=tokens_used,
        )
```

### Prompt Building

Each task type has a prompt template. The executor fills in context.

```python
class PromptBuilder:
    """Builds specialist prompts from delegation context."""

    TEMPLATES = {
        "execute_code": CODE_EXECUTION_TEMPLATE,
        "execute_analysis": ANALYSIS_TEMPLATE,
        "execute_test": TEST_EXECUTION_TEMPLATE,
        "execute_debug": DEBUG_TEMPLATE,
        "execute_refactor": REFACTOR_TEMPLATE,
    }

    def build(self, delegation: FleetDelegation, specialist: AdapterInfo) -> str:
        template = self.TEMPLATES.get(delegation.task_type, GENERIC_TEMPLATE)

        return template.format(
            objective=delegation.objective,
            constraints=self._format_constraints(delegation.constraints),
            context=self._format_context(delegation.context),
            specialist_domain=specialist.domains[0] if specialist.domains else "general",
            output_format=OUTPUT_FORMAT_INSTRUCTIONS,
        )

    def _format_constraints(self, constraints: list[str]) -> str:
        if not constraints:
            return "None specified."
        return "\n".join(f"- {c}" for c in constraints)

    def _format_context(self, context: dict) -> str:
        """Format resolved context for the prompt."""
        parts = []
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 1000:
                # Truncate long context with note
                value = value[:1000] + f"\n... [truncated, {len(value)} chars total]"
            parts.append(f"### {key}\n```\n{value}\n```")
        return "\n\n".join(parts)
```

## Prompt Templates

### Code Execution Template

```python
CODE_EXECUTION_TEMPLATE = """You are a {specialist_domain} specialist. Execute the following task precisely.

## Objective
{objective}

## Constraints
{constraints}

## Context
{context}

## Output Format
{output_format}

Begin your response with your reasoning, then provide the solution."""
```

### Output Format Instructions

```python
OUTPUT_FORMAT_INSTRUCTIONS = """Respond in this exact format:

<reasoning>
Brief explanation of your approach (2-3 sentences max)
</reasoning>

<solution>
Your code or answer here
</solution>

<confidence>
A number from 0.0 to 1.0 indicating how confident you are in this solution
</confidence>

<notes>
Any edge cases, assumptions, or concerns (optional)
</notes>"""
```

### Analysis Template

```python
ANALYSIS_TEMPLATE = """You are a {specialist_domain} specialist. Analyze the following and provide insights.

## Objective
{objective}

## Constraints
{constraints}

## Context
{context}

## Output Format
{output_format}

Focus on actionable insights. Be specific and concrete."""
```

### Debug Template

```python
DEBUG_TEMPLATE = """You are a {specialist_domain} debugging specialist. Identify and fix the issue.

## Problem
{objective}

## Constraints
{constraints}

## Code/Context
{context}

## Output Format
{output_format}

First identify the root cause, then provide the fix."""
```

## Response Parsing

### Structured Output Parser

```python
import re
from dataclasses import dataclass

@dataclass
class ParsedResponse:
    reasoning: str
    solution: str
    confidence: float
    notes: str | None
    raw_response: str

class ResponseParser:
    """Parse specialist responses into structured format."""

    PATTERNS = {
        "reasoning": re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL),
        "solution": re.compile(r"<solution>(.*?)</solution>", re.DOTALL),
        "confidence": re.compile(r"<confidence>([\d.]+)</confidence>"),
        "notes": re.compile(r"<notes>(.*?)</notes>", re.DOTALL),
    }

    def parse(self, response: str) -> ParsedResponse:
        def extract(pattern: re.Pattern, default: str = "") -> str:
            match = pattern.search(response)
            return match.group(1).strip() if match else default

        confidence_str = extract(self.PATTERNS["confidence"], "0.5")
        try:
            confidence = float(confidence_str)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        except ValueError:
            confidence = 0.5  # Default if parsing fails

        solution = extract(self.PATTERNS["solution"])

        # Penalize empty or trivial solutions
        if not solution or len(solution.strip()) < 10:
            confidence = min(confidence, 0.2)  # Cap at 0.2 for empty

        return ParsedResponse(
            reasoning=extract(self.PATTERNS["reasoning"]),
            solution=solution,
            confidence=confidence,
            notes=extract(self.PATTERNS["notes"]) or None,
            raw_response=response,
        )

    def is_well_formed(self, response: str) -> bool:
        """Check if response follows expected format with non-empty solution."""
        has_reasoning = self.PATTERNS["reasoning"].search(response)
        solution_match = self.PATTERNS["solution"].search(response)

        if not (has_reasoning and solution_match):
            return False

        # Check solution is non-empty
        solution = solution_match.group(1).strip()
        return len(solution) >= 10  # Minimum viable solution
```

### Fallback for Malformed Responses

```python
def parse_with_fallback(self, response: str) -> ParsedResponse:
    """Parse response, using heuristics if not well-formed."""
    if self.is_well_formed(response):
        return self.parse(response)

    # Fallback: treat entire response as solution
    return ParsedResponse(
        reasoning="[Response did not follow expected format]",
        solution=response.strip(),
        confidence=0.3,  # Lower confidence for malformed
        notes="Warning: Response was not in expected format",
        raw_response=response,
    )
```

## Outcome Construction

### Building the Outcome

```python
class OutcomeBuilder:
    """Construct Fleet outcomes from parsed responses."""

    def __init__(self):
        self._pending_large_artifacts: list[dict] = []

    def build(
        self,
        delegation: FleetDelegation,
        parsed: ParsedResponse,
        specialist: AdapterInfo,
        execution_time_ms: int,
        tokens_used: TokenUsage,
    ) -> FleetOutcome:

        # Determine status based on confidence and format
        status = self._determine_status(parsed)

        # Build artifacts (may queue large ones for Beads storage)
        message_id = delegation.message.id
        artifacts = self._build_artifacts(delegation.task_type, parsed, message_id)

        # Check for surprise (unusually low/high confidence)
        surprise_flag, surprise_reason = self._check_surprise(
            parsed.confidence, specialist
        )

        return FleetOutcome(
            status=status,
            summary=self._build_summary(parsed, specialist),
            artifacts=artifacts,
            confidence=parsed.confidence,
            execution_time_ms=execution_time_ms,
            resources_used=ResourceMetrics(
                specialist=specialist.name,
                lora_int_id=specialist.lora_int_id,
                tokens_in=tokens_used.prompt_tokens,
                tokens_out=tokens_used.completion_tokens,
            ),
            surprise_flag=surprise_flag,
            surprise_reason=surprise_reason,
            error_info=None,
        )

    def _determine_status(self, parsed: ParsedResponse) -> OutcomeStatus:
        """Determine status based on confidence AND content quality."""
        # Check for content issues that override confidence
        solution_len = len(parsed.solution.strip()) if parsed.solution else 0
        has_reasoning = bool(parsed.reasoning and len(parsed.reasoning) > 5)

        # Empty or trivial solution = FAILED regardless of claimed confidence
        if solution_len < 10:
            return OutcomeStatus.FAILED

        # Missing reasoning with high confidence = suspicious, downgrade to PARTIAL
        if not has_reasoning and parsed.confidence >= 0.7:
            return OutcomeStatus.PARTIAL

        # Standard confidence-based determination
        if parsed.confidence >= 0.7:
            return OutcomeStatus.SUCCESS
        elif parsed.confidence >= 0.4:
            return OutcomeStatus.PARTIAL
        else:
            return OutcomeStatus.FAILED

    def _build_artifacts(
        self,
        task_type: str,
        parsed: ParsedResponse,
        message_id: str,
    ) -> list[Artifact]:
        artifacts = []

        # Main solution artifact
        artifact_type = {
            "execute_code": "code",
            "execute_analysis": "analysis",
            "execute_test": "test_code",
            "execute_debug": "fix",
            "execute_refactor": "refactored_code",
        }.get(task_type, "output")

        # Inline if small, otherwise store in Beads
        solution_bytes = len(parsed.solution.encode('utf-8'))
        if solution_bytes < 1024:  # < 1KB inline
            artifacts.append(Artifact(
                artifact_type=artifact_type,
                content_ref=None,
                inline_content=parsed.solution,
                metadata={
                    "reasoning": parsed.reasoning,
                    "notes": parsed.notes,
                    "size_bytes": solution_bytes,
                },
            ))
        else:
            # Large artifact: store full content in Beads, reference here
            # Note: actual storage happens in store_large_artifact()
            content_ref = f"beads:execution/outputs/{message_id}/{artifact_type}"
            artifacts.append(Artifact(
                artifact_type=artifact_type,
                content_ref=content_ref,
                inline_content=None,  # Full content in Beads
                metadata={
                    "reasoning": parsed.reasoning,
                    "notes": parsed.notes,
                    "size_bytes": solution_bytes,
                    "full_content_at": content_ref,
                },
            ))
            # Store the FULL solution (not truncated) - see store_large_artifact()
            self._pending_large_artifacts.append({
                "ref": content_ref,
                "content": parsed.solution,  # Full content, not truncated!
                "metadata": {
                    "reasoning": parsed.reasoning,
                    "notes": parsed.notes,
                },
            })

        return artifacts

    def _check_surprise(
        self,
        confidence: float,
        specialist: AdapterInfo
    ) -> tuple[bool, str | None]:
        """Check if outcome is surprising based on historical performance."""
        # Surprisingly low confidence from a usually-reliable specialist
        if specialist.success_rate > 0.9 and confidence < 0.5:
            return True, f"Low confidence ({confidence}) from high-performing specialist"

        # Surprisingly high confidence on first attempts
        if specialist.request_count < 10 and confidence > 0.95:
            return True, f"Very high confidence ({confidence}) from new specialist"

        return False, None

    def _build_summary(self, parsed: ParsedResponse, specialist: AdapterInfo) -> str:
        status_word = "completed" if parsed.confidence >= 0.7 else "attempted"
        return f"Task {status_word} by {specialist.name} (confidence: {parsed.confidence:.2f})"
```

## Error Handling

### Error Types

```python
class FleetError(Exception):
    """Base class for Fleet errors."""
    pass

class SpecialistUnavailableError(FleetError):
    """Requested specialist not available."""
    pass

class VLLMConnectionError(FleetError):
    """Cannot connect to vLLM server."""
    pass

class ResponseParseError(FleetError):
    """Failed to parse specialist response."""
    pass
```

### Error Outcome Construction

```python
def _error_outcome(
    self,
    delegation: FleetDelegation,
    error: Exception,
    start_time: float
) -> FleetOutcome:
    """Construct outcome for error cases."""
    execution_time_ms = int((time.monotonic() - start_time) * 1000)

    error_type = type(error).__name__
    recoverable = isinstance(error, (asyncio.TimeoutError, VLLMConnectionError))

    return FleetOutcome(
        status=OutcomeStatus.FAILED,
        summary=f"Execution failed: {error_type}",
        artifacts=[],
        confidence=0.0,
        execution_time_ms=execution_time_ms,
        resources_used=ResourceMetrics(
            specialist=delegation.specialist_hint or "unknown",
            lora_int_id=0,
            tokens_in=0,
            tokens_out=0,
        ),
        surprise_flag=True,
        surprise_reason=f"Unexpected error: {error_type}",
        error_info=ErrorInfo(
            error_type=error_type,
            error_detail=str(error),
            recoverable=recoverable,
        ),
    )

def _timeout_outcome(
    self,
    delegation: FleetDelegation,
    start_time: float
) -> FleetOutcome:
    """Construct outcome for timeout cases."""
    return FleetOutcome(
        status=OutcomeStatus.TIMEOUT,
        summary=f"Execution timed out after {delegation.timeout_ms}ms",
        artifacts=[],
        confidence=0.0,
        execution_time_ms=delegation.timeout_ms,
        resources_used=ResourceMetrics(
            specialist=delegation.specialist_hint or "unknown",
            lora_int_id=0,
            tokens_in=0,  # Unknown, request may have been in-flight
            tokens_out=0,
        ),
        surprise_flag=False,  # Timeouts are expected sometimes
        surprise_reason=None,
        error_info=ErrorInfo(
            error_type="TimeoutError",
            error_detail=f"Exceeded {delegation.timeout_ms}ms deadline",
            recoverable=True,  # Can retry with longer timeout
        ),
    )
```

## Retry Logic

The Fleet itself does NOT retry - that's the Dispatcher's job. But it provides clear signals:

```python
# In OutcomePayload
error_info = ErrorInfo(
    error_type="VLLMConnectionError",
    error_detail="Connection refused to localhost:8000",
    recoverable=True,  # ← Dispatcher can retry
)
```

The Dispatcher uses `recoverable` to decide whether to:
1. Retry with same specialist
2. Route to alternate specialist
3. Escalate to Architect

## Beads Integration

### Writing Execution Artifacts

```python
class OutcomeBuilder:
    def __init__(self):
        self._pending_large_artifacts: list[dict] = []

    async def store_pending_artifacts(self, beads: BeadsClient):
        """Store all large artifacts that were queued during build."""
        for item in self._pending_large_artifacts:
            # Remove "beads:" prefix for the key
            key = item["ref"].replace("beads:", "")
            await beads.set(key, {
                "content": item["content"],  # Full content, not truncated
                "metadata": item["metadata"],
                "created_at": datetime.now().isoformat(),
            })
        self._pending_large_artifacts.clear()
```

**Important**: Large artifacts store the FULL solution content in Beads, not the truncated preview. The `inline_content` field is set to `None` and `content_ref` points to the Beads location.

### Logging Execution

```python
async def log_execution(
    self,
    delegation: FleetDelegation,
    outcome: FleetOutcome,
    beads: BeadsClient
):
    """Log execution details for monitoring and training."""
    log_key = f"execution/logs/{delegation.message.id}"

    await beads.set(log_key, {
        "delegation": {
            "task_type": delegation.task_type,
            "objective": delegation.objective,
            "specialist_hint": delegation.specialist_hint,
        },
        "outcome": {
            "status": outcome.status.value,
            "confidence": outcome.confidence,
            "execution_time_ms": outcome.execution_time_ms,
            "surprise_flag": outcome.surprise_flag,
        },
        "timestamp": datetime.now().isoformat(),
    })
```

## Configuration

### Fleet Configuration

```yaml
# fleet_config.yaml
vllm:
  base_url: "http://localhost:8000/v1"
  model: "Qwen/QwQ-32B-Preview-AWQ"
  default_timeout_ms: 30000
  max_tokens: 4096
  temperature: 0.1

execution:
  confidence_threshold_success: 0.7
  confidence_threshold_partial: 0.4
  max_context_chars: 10000
  truncate_long_responses: true

beads:
  store_artifacts_above_bytes: 1024
  log_all_executions: true
```

### Environment Variables

```bash
FLEET_VLLM_BASE_URL=http://localhost:8000/v1
FLEET_DEFAULT_TIMEOUT_MS=30000
FLEET_LOG_LEVEL=INFO
```

## Usage Example

### End-to-End Execution

```python
async def main():
    # Setup
    vllm_client = AsyncOpenAI(base_url="http://localhost:8000/v1")
    registry = AdapterRegistry.from_beads()
    beads = BeadsClient()

    executor = SpecialistExecutor(vllm_client, registry)

    # Receive delegation (from Dispatcher)
    delegation = FleetDelegation(
        message=incoming_message,
        task_type="execute_code",
        objective="Implement binary search that handles empty arrays",
        constraints=["Return -1 if not found", "O(log n) time"],
        context={"test_cases": "...", "signature": "def binary_search(...)"},
        specialist_hint="python-lora",
        timeout_ms=30000,
    )

    # Execute
    outcome = await executor.execute(delegation)

    # Log and store
    await executor.log_execution(delegation, outcome, beads)

    # Return to Dispatcher
    return outcome
```

## Testing Strategy

### Unit Tests

```python
def test_response_parser_well_formed():
    parser = ResponseParser()
    response = """
    <reasoning>Simple implementation</reasoning>
    <solution>def binary_search(arr, target): ...</solution>
    <confidence>0.9</confidence>
    """
    parsed = parser.parse(response)
    assert parsed.confidence == 0.9
    assert "binary_search" in parsed.solution

def test_response_parser_malformed():
    parser = ResponseParser()
    response = "Here is the code: def foo(): pass"
    parsed = parser.parse_with_fallback(response)
    assert parsed.confidence == 0.3  # Low confidence for malformed
    assert "foo" in parsed.solution
```

### Integration Tests

```python
@pytest.mark.integration
async def test_executor_with_vllm():
    executor = SpecialistExecutor(...)
    delegation = FleetDelegation(
        task_type="execute_code",
        objective="Print hello world",
        ...
    )
    outcome = await executor.execute(delegation)
    assert outcome.status in [OutcomeStatus.SUCCESS, OutcomeStatus.PARTIAL]
    assert outcome.artifacts[0].inline_content
```
