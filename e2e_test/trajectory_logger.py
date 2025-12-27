"""Trajectory logging for vLLM calls.

Implements issue hi_moe-iz9: Log every vLLM call for training data collection.

Log format (JSONL):
- ts: ISO timestamp
- task_id: Current task ID
- call_id: Unique call identifier
- tier: Which tier made the call (architect/dispatcher/fleet)
- specialist: Which specialist (if applicable)
- lora: Which LoRA adapter (if any)
- input: Messages sent to model
- output: Model response
- tokens_in: Prompt tokens
- tokens_out: Completion tokens
- latency_ms: Call latency
- status: success/error
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class VLLMCallRecord:
    """Record of a single vLLM API call."""

    ts: str  # ISO timestamp
    task_id: str
    call_id: str
    tier: str  # architect, dispatcher, fleet
    specialist: str | None = None
    lora: str | None = None
    input: list[dict] | None = None  # Messages
    output: str | None = None  # Response text
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0
    status: str = "success"  # success, error
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# Tier-specific records for training data collection (hi_moe-r8q)


@dataclass
class ArchitectRecord:
    """Record of Architect tier decision.

    Captures: goal, delegation, success criteria, revisions.
    Used for training coordination strategies.
    """

    ts: str  # ISO timestamp
    task_id: str
    goal: str  # What the Architect is trying to achieve
    delegation: dict | None = None  # Task delegated to Dispatcher
    success_criteria: list[str] | None = None  # How success is measured
    plan: str | None = None  # Generated execution plan
    revision_of: str | None = None  # If this is a revision, ID of original
    revision_reason: str | None = None  # Why revision was needed
    outcome_status: str | None = None  # completed/failed after execution
    outcome_summary: str | None = None  # Brief description of result
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {"type": "architect_decision", **asdict(self)}
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class DispatcherRecord:
    """Record of Dispatcher tier routing decision.

    Captures: task, routing decision, rationale, specialist selection.
    Used for training routing LoRA.
    """

    ts: str  # ISO timestamp
    task_id: str
    task_objective: str  # What needs to be done
    routing_decision: str  # "structured_plan" or "heuristic"
    plan_steps: list[dict] | None = None  # Structured plan if used
    rationale: str | None = None  # Why this routing was chosen
    specialist: str | None = None  # Single specialist if heuristic
    context_summary: str | None = None  # Relevant context for decision
    # Routing strategy fields (hi_moe-gr7)
    routing_strategy: str | None = None  # "math_first" or "python_direct"
    routing_signals: list[str] | None = None  # Signals that led to strategy choice
    outcome_status: str | None = None  # completed/failed
    outcome_error: str | None = None  # Error if failed
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {"type": "dispatcher_routing", **asdict(self)}
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class FleetRecord:
    """Record of Fleet tier execution.

    Captures: task, specialist, output, result.
    Used for training specialist LoRAs.
    """

    ts: str  # ISO timestamp
    task_id: str
    task_objective: str  # What the specialist was asked to do
    specialist: str  # Which specialist executed (python, math, etc.)
    prompt_used: str | None = None  # System prompt given to specialist
    output_code: str | None = None  # Generated code
    output_raw: str | None = None  # Raw LLM response
    execution_time_ms: float = 0
    status: str = "success"  # success, error
    error: str | None = None
    validation_result: dict | None = None  # If code was validated
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {"type": "fleet_execution", **asdict(self)}
        return {k: v for k, v in d.items() if v is not None}


class TrajectoryLogger:
    """Logs vLLM call trajectories to JSONL files.

    Each run creates a new file in the runs/ directory.
    Files are named: {run_id}.jsonl

    Usage:
        logger = TrajectoryLogger(runs_dir="./runs")
        logger.start_run("run-123", {"problem_id": "two-sum"})

        # Log individual calls
        logger.log_call(VLLMCallRecord(...))

        # End run
        logger.end_run({"success": True})
    """

    def __init__(self, runs_dir: Path | str = "./runs"):
        """Initialize trajectory logger.

        Args:
            runs_dir: Directory to store trajectory files
        """
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self._current_run_id: str | None = None
        self._current_file: Path | None = None
        self._call_count: int = 0

    def start_run(self, run_id: str, metadata: dict | None = None) -> None:
        """Start a new run trajectory.

        Args:
            run_id: Unique run identifier
            metadata: Optional run metadata (problem info, etc.)
        """
        self._current_run_id = run_id
        self._current_file = self.runs_dir / f"{run_id}.jsonl"
        self._call_count = 0

        # Write run start record
        start_record = {
            "type": "run_start",
            "run_id": run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        self._write_record(start_record)

        logger.info(f"[TrajectoryLogger] Started run {run_id}")

    def log_call(self, record: VLLMCallRecord) -> None:
        """Log a vLLM call record.

        Args:
            record: The call record to log
        """
        if not self._current_file:
            logger.warning("[TrajectoryLogger] No active run, call not logged")
            return

        self._call_count += 1
        self._write_record({"type": "vllm_call", **record.to_dict()})

    def log_architect(self, record: ArchitectRecord) -> None:
        """Log an Architect tier decision.

        Args:
            record: The architect decision record
        """
        if not self._current_file:
            logger.warning("[TrajectoryLogger] No active run, architect record not logged")
            return

        self._write_record(record.to_dict())

    def log_dispatcher(self, record: DispatcherRecord) -> None:
        """Log a Dispatcher tier routing decision.

        Args:
            record: The dispatcher routing record
        """
        if not self._current_file:
            logger.warning("[TrajectoryLogger] No active run, dispatcher record not logged")
            return

        self._write_record(record.to_dict())

    def log_fleet(self, record: FleetRecord) -> None:
        """Log a Fleet tier execution.

        Args:
            record: The fleet execution record
        """
        if not self._current_file:
            logger.warning("[TrajectoryLogger] No active run, fleet record not logged")
            return

        self._write_record(record.to_dict())

    def end_run(self, result: dict | None = None) -> None:
        """End the current run trajectory.

        Args:
            result: Optional run result summary
        """
        if not self._current_file:
            return

        end_record = {
            "type": "run_end",
            "run_id": self._current_run_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "total_calls": self._call_count,
            **(result or {}),
        }
        self._write_record(end_record)

        logger.info(
            f"[TrajectoryLogger] Ended run {self._current_run_id} "
            f"with {self._call_count} calls"
        )

        self._current_run_id = None
        self._current_file = None
        self._call_count = 0

    def _write_record(self, record: dict) -> None:
        """Write a record to the current trajectory file."""
        if not self._current_file:
            return

        with open(self._current_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    @property
    def current_run_id(self) -> str | None:
        """Get current run ID."""
        return self._current_run_id


class LLMClientProtocol(Protocol):
    """Protocol for LLM clients that can be wrapped for logging."""

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        ...


class LoggingLLMClient:
    """Wrapper that adds trajectory logging to any LLM client.

    Intercepts all generate() calls and logs them to a TrajectoryLogger.
    """

    def __init__(
        self,
        client: LLMClientProtocol,
        trajectory_logger: TrajectoryLogger,
        default_tier: str = "unknown",
    ):
        """Initialize logging wrapper.

        Args:
            client: The underlying LLM client
            trajectory_logger: Logger for recording calls
            default_tier: Default tier name for calls
        """
        self.client = client
        self.logger = trajectory_logger
        self.default_tier = default_tier

        # Track the last call ID for linking validations/retries (hi_moe-35o)
        self._last_call_id: str | None = None

        # Context for current call (set by caller)
        self._current_task_id: str | None = None
        self._current_tier: str | None = None
        self._current_specialist: str | None = None
        self._current_lora: str | None = None

    def set_context(
        self,
        task_id: str | None = None,
        tier: str | None = None,
        specialist: str | None = None,
        lora: str | None = None,
    ) -> None:
        """Set context for subsequent calls.

        Args:
            task_id: Current task ID
            tier: Current tier (architect/dispatcher/fleet)
            specialist: Current specialist
            lora: LoRA adapter in use
        """
        if task_id is not None:
            self._current_task_id = task_id
        if tier is not None:
            self._current_tier = tier
        if specialist is not None:
            self._current_specialist = specialist
        if lora is not None:
            self._current_lora = lora

    def clear_context(self) -> None:
        """Clear call context."""
        self._current_task_id = None
        self._current_tier = None
        self._current_specialist = None
        self._current_lora = None

    @property
    def last_call_id(self) -> str | None:
        """Get the ID of the most recent generate() call (hi_moe-35o).

        Returns:
            The call_id from the last generate() call, or None if no calls made.
        """
        return self._last_call_id

    async def get_available_adapters(self) -> list[str]:
        """Get available adapters from underlying client."""
        if hasattr(self.client, "get_available_adapters"):
            return await self.client.get_available_adapters()
        return ["base"]

    async def generate(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """Generate completion with logging.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            **kwargs: Additional arguments passed to underlying client

        Returns:
            Generated text
        """
        call_id = f"call-{uuid.uuid4().hex[:8]}"
        self._last_call_id = call_id  # Store for retrieval (hi_moe-35o)
        start_time = time.monotonic()
        ts = datetime.now(timezone.utc).isoformat()

        try:
            # Call underlying client
            result = await self.client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            # Estimate tokens (actual counts would come from response)
            tokens_in = sum(len(m.get("content", "")) // 4 for m in messages)
            tokens_out = len(result) // 4

            # Log successful call
            record = VLLMCallRecord(
                ts=ts,
                task_id=self._current_task_id or "unknown",
                call_id=call_id,
                tier=self._current_tier or self.default_tier,
                specialist=self._current_specialist,
                lora=self._current_lora,
                input=messages,
                output=result,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                status="success",
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            self.logger.log_call(record)

            return result

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000

            # Log failed call
            record = VLLMCallRecord(
                ts=ts,
                task_id=self._current_task_id or "unknown",
                call_id=call_id,
                tier=self._current_tier or self.default_tier,
                specialist=self._current_specialist,
                lora=self._current_lora,
                input=messages,
                output=None,
                tokens_in=sum(len(m.get("content", "")) // 4 for m in messages),
                tokens_out=0,
                latency_ms=latency_ms,
                status="error",
                error=str(e),
            )
            self.logger.log_call(record)

            raise


def create_logging_client(
    client: LLMClientProtocol,
    runs_dir: Path | str = "./runs",
) -> tuple[LoggingLLMClient, TrajectoryLogger]:
    """Create a logging-wrapped LLM client.

    Args:
        client: The underlying LLM client
        runs_dir: Directory for trajectory files

    Returns:
        Tuple of (wrapped client, trajectory logger)
    """
    trajectory_logger = TrajectoryLogger(runs_dir)
    logging_client = LoggingLLMClient(client, trajectory_logger)
    return logging_client, trajectory_logger


# Utility functions for analyzing trajectories

def load_trajectory(file_path: Path | str) -> list[dict]:
    """Load trajectory from JSONL file.

    Args:
        file_path: Path to trajectory file

    Returns:
        List of records
    """
    records = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def filter_successful_calls(records: list[dict]) -> list[dict]:
    """Filter to only successful vLLM calls.

    Args:
        records: All trajectory records

    Returns:
        Only successful vLLM call records
    """
    return [
        r for r in records
        if r.get("type") == "vllm_call" and r.get("status") == "success"
    ]


def compute_trajectory_stats(records: list[dict]) -> dict:
    """Compute statistics for a trajectory.

    Args:
        records: Trajectory records

    Returns:
        Statistics dict with call counts, token usage, latency
    """
    calls = [r for r in records if r.get("type") == "vllm_call"]

    if not calls:
        return {"total_calls": 0}

    total_tokens_in = sum(r.get("tokens_in", 0) for r in calls)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in calls)
    total_latency = sum(r.get("latency_ms", 0) for r in calls)

    success_calls = [r for r in calls if r.get("status") == "success"]
    error_calls = [r for r in calls if r.get("status") == "error"]

    tier_counts = {}
    for call in calls:
        tier = call.get("tier", "unknown")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    return {
        "total_calls": len(calls),
        "successful_calls": len(success_calls),
        "error_calls": len(error_calls),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_tokens": total_tokens_in + total_tokens_out,
        "total_latency_ms": total_latency,
        "avg_latency_ms": total_latency / len(calls) if calls else 0,
        "calls_by_tier": tier_counts,
    }


# Tier-specific analysis utilities (hi_moe-r8q)


def filter_architect_records(records: list[dict]) -> list[dict]:
    """Filter to only Architect tier decision records.

    Args:
        records: All trajectory records

    Returns:
        Only architect decision records
    """
    return [r for r in records if r.get("type") == "architect_decision"]


def filter_dispatcher_records(records: list[dict]) -> list[dict]:
    """Filter to only Dispatcher tier routing records.

    Args:
        records: All trajectory records

    Returns:
        Only dispatcher routing records
    """
    return [r for r in records if r.get("type") == "dispatcher_routing"]


def filter_fleet_records(records: list[dict]) -> list[dict]:
    """Filter to only Fleet tier execution records.

    Args:
        records: All trajectory records

    Returns:
        Only fleet execution records
    """
    return [r for r in records if r.get("type") == "fleet_execution"]


def compute_tier_stats(records: list[dict]) -> dict:
    """Compute statistics for tier-specific training data.

    Args:
        records: Trajectory records

    Returns:
        Statistics dict with per-tier counts and success rates
    """
    architect = filter_architect_records(records)
    dispatcher = filter_dispatcher_records(records)
    fleet = filter_fleet_records(records)

    def success_rate(recs: list[dict]) -> float:
        if not recs:
            return 0.0
        successful = sum(1 for r in recs if r.get("outcome_status") == "completed"
                        or r.get("status") == "success")
        return successful / len(recs)

    # Routing decision breakdown
    structured_routing = sum(1 for r in dispatcher if r.get("routing_decision") == "structured_plan")
    heuristic_routing = sum(1 for r in dispatcher if r.get("routing_decision") == "heuristic")

    # Routing strategy breakdown (hi_moe-gr7)
    math_first_count = sum(1 for r in dispatcher if r.get("routing_strategy") == "math_first")
    python_direct_count = sum(1 for r in dispatcher if r.get("routing_strategy") == "python_direct")

    # Specialist usage
    specialist_counts = {}
    for r in fleet:
        spec = r.get("specialist", "unknown")
        specialist_counts[spec] = specialist_counts.get(spec, 0) + 1

    return {
        "architect_decisions": len(architect),
        "architect_success_rate": success_rate(architect),
        "dispatcher_routings": len(dispatcher),
        "dispatcher_success_rate": success_rate(dispatcher),
        "structured_routing_count": structured_routing,
        "heuristic_routing_count": heuristic_routing,
        # Routing strategy stats (hi_moe-gr7)
        "math_first_count": math_first_count,
        "python_direct_count": python_direct_count,
        "fleet_executions": len(fleet),
        "fleet_success_rate": success_rate(fleet),
        "specialist_usage": specialist_counts,
    }


def extract_training_pairs(records: list[dict], tier: str) -> list[dict]:
    """Extract input/output pairs for training from tier records.

    Args:
        records: Trajectory records
        tier: Which tier to extract ("architect", "dispatcher", "fleet")

    Returns:
        List of training pairs with input context and expected output
    """
    pairs = []

    if tier == "architect":
        for r in filter_architect_records(records):
            if r.get("outcome_status") == "completed":
                pairs.append({
                    "input": {
                        "goal": r.get("goal"),
                        "context": r.get("metadata", {}).get("context"),
                    },
                    "output": {
                        "plan": r.get("plan"),
                        "delegation": r.get("delegation"),
                        "success_criteria": r.get("success_criteria"),
                    },
                    "outcome": r.get("outcome_summary"),
                })

    elif tier == "dispatcher":
        for r in filter_dispatcher_records(records):
            if r.get("outcome_status") == "completed":
                pairs.append({
                    "input": {
                        "task": r.get("task_objective"),
                        "context": r.get("context_summary"),
                    },
                    "output": {
                        "routing_decision": r.get("routing_decision"),
                        "plan_steps": r.get("plan_steps"),
                        "rationale": r.get("rationale"),
                        # Routing strategy (hi_moe-gr7)
                        "routing_strategy": r.get("routing_strategy"),
                        "routing_signals": r.get("routing_signals"),
                    },
                })

    elif tier == "fleet":
        for r in filter_fleet_records(records):
            if r.get("status") == "success":
                pairs.append({
                    "input": {
                        "task": r.get("task_objective"),
                        "specialist": r.get("specialist"),
                        "prompt": r.get("prompt_used"),
                    },
                    "output": {
                        "code": r.get("output_code"),
                    },
                    "validation": r.get("validation_result"),
                })

    return pairs


# Training data collection utilities (hi_moe-vo8)


@dataclass
class TrainingExample:
    """A single training example in SFT format.

    Matches the format expected by modal_app/training.py.
    """

    domain: str  # python, math, etc.
    problem: str  # The problem description
    reasoning: str  # Chain-of-thought reasoning
    solution: str  # The final solution (code)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "domain": self.domain,
            "problem": self.problem,
            "reasoning": self.reasoning,
            "solution": self.solution,
        }


class TrajectoryDataCollector:
    """Collects trajectory data and converts to training format.

    Bridges the gap between trajectory logs (JSONL from runs) and
    the training pipeline format expected by SFTTrainer.

    Usage:
        collector = TrajectoryDataCollector(runs_dir="./runs")
        examples = collector.collect_fleet_examples()
        collector.write_training_file(examples, "python_training.jsonl")
    """

    def __init__(self, runs_dir: Path | str = "./runs"):
        """Initialize collector.

        Args:
            runs_dir: Directory containing trajectory JSONL files
        """
        self.runs_dir = Path(runs_dir)

    def load_all_trajectories(self) -> list[dict]:
        """Load all trajectory records from all run files.

        Returns:
            Combined list of all records from all runs
        """
        all_records = []

        if not self.runs_dir.exists():
            logger.warning(f"[TrajectoryDataCollector] Runs dir not found: {self.runs_dir}")
            return all_records

        for jsonl_file in self.runs_dir.glob("*.jsonl"):
            try:
                records = load_trajectory(jsonl_file)
                all_records.extend(records)
                logger.debug(f"Loaded {len(records)} records from {jsonl_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {jsonl_file}: {e}")

        logger.info(f"[TrajectoryDataCollector] Loaded {len(all_records)} total records")
        return all_records

    def collect_fleet_examples(
        self,
        require_validation: bool = True,
        min_code_length: int = 10,
    ) -> list[TrainingExample]:
        """Collect training examples from Fleet tier executions.

        Only includes successful executions with validated code.

        Args:
            require_validation: Only include examples that passed validation
            min_code_length: Minimum code length to include

        Returns:
            List of TrainingExample objects ready for training
        """
        records = self.load_all_trajectories()
        pairs = extract_training_pairs(records, "fleet")

        examples = []
        for pair in pairs:
            # Extract fields
            task = pair.get("input", {}).get("task", "")
            specialist = pair.get("input", {}).get("specialist", "python")
            prompt = pair.get("input", {}).get("prompt", "")
            code = pair.get("output", {}).get("code", "")
            validation = pair.get("validation")

            # Skip if no code
            if not code or len(code) < min_code_length:
                continue

            # Skip if validation required but failed
            if require_validation:
                if not validation or not validation.get("passed"):
                    continue

            # Map specialist to domain
            domain = self._specialist_to_domain(specialist)

            # Build reasoning from prompt and context
            reasoning = self._build_reasoning(prompt, task, specialist)

            examples.append(TrainingExample(
                domain=domain,
                problem=task,
                reasoning=reasoning,
                solution=code,
            ))

        logger.info(f"[TrajectoryDataCollector] Collected {len(examples)} fleet examples")
        return examples

    def collect_dispatcher_examples(self) -> list[TrainingExample]:
        """Collect training examples from Dispatcher tier routing decisions.

        Returns:
            List of TrainingExample objects for routing training
        """
        records = self.load_all_trajectories()
        pairs = extract_training_pairs(records, "dispatcher")

        examples = []
        for pair in pairs:
            task = pair.get("input", {}).get("task", "")
            context = pair.get("input", {}).get("context", "")
            output = pair.get("output", {})

            routing_decision = output.get("routing_decision", "")
            rationale = output.get("rationale", "")
            routing_strategy = output.get("routing_strategy", "")

            # Build solution as structured routing decision
            solution = self._format_routing_decision(output)
            if not solution:
                continue

            # Reasoning is the rationale
            reasoning = rationale or f"Route using {routing_strategy or routing_decision}"

            examples.append(TrainingExample(
                domain="routing",
                problem=f"Route task: {task}\nContext: {context}",
                reasoning=reasoning,
                solution=solution,
            ))

        logger.info(f"[TrajectoryDataCollector] Collected {len(examples)} dispatcher examples")
        return examples

    def write_training_file(
        self,
        examples: list[TrainingExample],
        output_path: Path | str,
    ) -> int:
        """Write training examples to JSONL file.

        Args:
            examples: List of training examples
            output_path: Path to write JSONL file

        Returns:
            Number of examples written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")

        logger.info(f"[TrajectoryDataCollector] Wrote {len(examples)} examples to {output_path}")
        return len(examples)

    def _specialist_to_domain(self, specialist: str) -> str:
        """Map specialist name to training domain.

        Args:
            specialist: Specialist name (python, math, etc.)

        Returns:
            Domain string for training
        """
        domain_map = {
            "python": "python",
            "math": "math",
            "algorithm": "algorithms",
            "data_structures": "data_structures",
            "optimization": "optimization",
        }
        return domain_map.get(specialist, "python")

    def _build_reasoning(self, prompt: str, task: str, specialist: str) -> str:
        """Build reasoning chain from context.

        Args:
            prompt: System prompt used
            task: Task description
            specialist: Specialist name

        Returns:
            Reasoning string for training
        """
        parts = []

        if prompt:
            # Extract key reasoning hints from prompt
            if "step by step" in prompt.lower():
                parts.append("Approach this step by step.")
            if "edge cases" in prompt.lower():
                parts.append("Consider edge cases.")

        parts.append(f"Using {specialist} specialist to solve: {task[:100]}")

        return " ".join(parts) if parts else "Solving the given problem."

    def _format_routing_decision(self, output: dict) -> str:
        """Format routing decision as structured output.

        Args:
            output: Routing decision output dict

        Returns:
            Formatted routing decision string
        """
        decision = output.get("routing_decision", "")
        strategy = output.get("routing_strategy", "")
        signals = output.get("routing_signals", [])
        plan_steps = output.get("plan_steps", [])

        if decision == "structured_plan" and plan_steps:
            steps_str = "\n".join(f"- {s}" for s in plan_steps[:5])
            return f"ROUTING: structured_plan\nSTRATEGY: {strategy}\nSTEPS:\n{steps_str}"
        elif decision == "heuristic":
            signals_str = ", ".join(signals) if signals else "default"
            return f"ROUTING: heuristic\nSTRATEGY: {strategy}\nSIGNALS: {signals_str}"

        return ""


def collect_training_data(
    runs_dir: Path | str = "./runs",
    output_dir: Path | str = "./training_data",
    include_fleet: bool = True,
    include_dispatcher: bool = True,
) -> dict:
    """Convenience function to collect all training data from trajectories.

    Args:
        runs_dir: Directory containing trajectory JSONL files
        output_dir: Directory to write training files
        include_fleet: Include Fleet tier examples
        include_dispatcher: Include Dispatcher tier examples

    Returns:
        Summary dict with counts of examples collected
    """
    collector = TrajectoryDataCollector(runs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"fleet_examples": 0, "dispatcher_examples": 0}

    if include_fleet:
        fleet_examples = collector.collect_fleet_examples()
        if fleet_examples:
            collector.write_training_file(
                fleet_examples,
                output_dir / "fleet_training.jsonl",
            )
            summary["fleet_examples"] = len(fleet_examples)

    if include_dispatcher:
        dispatcher_examples = collector.collect_dispatcher_examples()
        if dispatcher_examples:
            collector.write_training_file(
                dispatcher_examples,
                output_dir / "dispatcher_training.jsonl",
            )
            summary["dispatcher_examples"] = len(dispatcher_examples)

    logger.info(f"[collect_training_data] Summary: {summary}")
    return summary
