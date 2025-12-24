# LoRA Training Specification

> Training pipeline for specialist LoRA adapters, starting with python-lora and math-lora.

## Overview

This spec defines how to train domain-specific LoRA adapters for the Specialized Fleet. Initial focus is on competitive programming data which provides:
- Objective correctness signals (code passes tests or doesn't)
- Clear domain separation (algorithms vs implementation)
- High-quality reasoning chains

## Training Data Sources

### Competitive Programming Datasets

| Dataset | Size | Use Case |
|---------|------|----------|
| [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) | 13K problems | Primary training source |
| [APPS](https://huggingface.co/datasets/codeparrot/apps) | 10K problems | Additional Python examples |
| [TACO](https://huggingface.co/datasets/BAAI/TACO) | 26K problems | Diverse algorithms |
| [LiveCodeBench](https://livecodebench.github.io/) | Ongoing | Validation (unseen problems) |

### Data Preparation

```python
from dataclasses import dataclass
from enum import Enum
import json
import hashlib


class Domain(Enum):
    PYTHON = "python"
    MATH = "math"
    ALGORITHMS = "algorithms"
    DATA_STRUCTURES = "data_structures"


@dataclass
class TrainingExample:
    """Single training example for LoRA fine-tuning."""
    problem_id: str
    problem_statement: str
    solution_code: str
    reasoning_trace: str | None  # Chain-of-thought if available
    domain: Domain
    difficulty: str  # "easy", "medium", "hard"
    test_results: dict  # {"passed": 10, "total": 10}

    @property
    def is_valid(self) -> bool:
        """Only use examples that pass all tests."""
        return self.test_results["passed"] == self.test_results["total"]

    @property
    def example_id(self) -> str:
        """Unique identifier for deduplication."""
        content = f"{self.problem_id}:{self.solution_code}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class TrainingDataset:
    """Collection of training examples for a specialist."""
    specialist_name: str
    examples: list[TrainingExample]

    def filter_valid(self) -> "TrainingDataset":
        """Keep only examples that pass all tests."""
        valid = [e for e in self.examples if e.is_valid]
        return TrainingDataset(self.specialist_name, valid)

    def deduplicate(self) -> "TrainingDataset":
        """Remove duplicate solutions."""
        seen: set[str] = set()
        unique: list[TrainingExample] = []
        for ex in self.examples:
            if ex.example_id not in seen:
                seen.add(ex.example_id)
                unique.append(ex)
        return TrainingDataset(self.specialist_name, unique)

    def to_jsonl(self, path: str) -> None:
        """Export as JSONL for training."""
        with open(path, "w") as f:
            for ex in self.examples:
                f.write(json.dumps({
                    "problem": ex.problem_statement,
                    "reasoning": ex.reasoning_trace or "",
                    "solution": ex.solution_code,
                    "domain": ex.domain.value,
                    "difficulty": ex.difficulty
                }) + "\n")
```

### Domain Classification

Classify problems for routing to correct specialist:

```python
DOMAIN_SIGNALS = {
    Domain.MATH: [
        "prove", "theorem", "lemma", "mathematical", "formula",
        "probability", "expected value", "modular arithmetic",
        "number theory", "combinatorics", "geometry"
    ],
    Domain.ALGORITHMS: [
        "dynamic programming", "dp", "greedy", "binary search",
        "graph", "tree", "shortest path", "bfs", "dfs",
        "segment tree", "fenwick", "suffix array"
    ],
    Domain.DATA_STRUCTURES: [
        "heap", "stack", "queue", "linked list", "hash",
        "trie", "union find", "disjoint set"
    ],
    Domain.PYTHON: [
        "python", "pythonic", "list comprehension",
        "generator", "decorator", "class"
    ]
}


def classify_problem(problem: str, tags: list[str]) -> Domain:
    """Classify a problem into a training domain."""
    text = problem.lower()

    # Check explicit tags first
    tag_set = {t.lower() for t in tags}
    if tag_set & {"math", "number theory", "combinatorics"}:
        return Domain.MATH
    if tag_set & {"dp", "graphs", "trees", "greedy"}:
        return Domain.ALGORITHMS

    # Fall back to keyword matching
    scores = {domain: 0 for domain in Domain}
    for domain, keywords in DOMAIN_SIGNALS.items():
        for keyword in keywords:
            if keyword in text:
                scores[domain] += 1

    # Default to PYTHON for implementation-focused problems
    max_domain = max(scores, key=scores.get)
    return max_domain if scores[max_domain] > 0 else Domain.PYTHON
```

## Training Format

### Prompt Template

Use a consistent format matching production inference:

```python
TRAINING_TEMPLATE = """<|im_start|>system
You are a {specialist_type} specialist. Solve the problem step by step, then provide working code.
<|im_end|>
<|im_start|>user
{problem_statement}
<|im_end|>
<|im_start|>assistant
{reasoning_trace}

```python
{solution_code}
```
<|im_end|>"""


def format_example(ex: TrainingExample) -> str:
    """Format a single example for training."""
    specialist_type = {
        Domain.PYTHON: "Python programming",
        Domain.MATH: "mathematical reasoning",
        Domain.ALGORITHMS: "algorithm design",
        Domain.DATA_STRUCTURES: "data structure"
    }[ex.domain]

    return TRAINING_TEMPLATE.format(
        specialist_type=specialist_type,
        problem_statement=ex.problem_statement,
        reasoning_trace=ex.reasoning_trace or "Let me solve this step by step.",
        solution_code=ex.solution_code
    )
```

### Generating Reasoning Traces

Many datasets lack reasoning chains. Generate them:

```python
import asyncio
from openai import AsyncOpenAI

REASONING_PROMPT = """Given this competitive programming problem and its correct solution, explain the reasoning step by step.

Problem:
{problem}

Correct Solution:
```python
{solution}
```

Provide a clear explanation of:
1. What the problem is asking
2. Key observations and insights
3. Algorithm choice and why
4. Implementation approach

Explanation:"""


async def generate_reasoning(
    client: AsyncOpenAI,
    problem: str,
    solution: str
) -> str:
    """Generate reasoning trace for a solution."""
    response = await client.chat.completions.create(
        model="Qwen/QwQ-32B-Preview-AWQ",  # Base model
        messages=[{
            "role": "user",
            "content": REASONING_PROMPT.format(
                problem=problem,
                solution=solution
            )
        }],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content


async def enrich_dataset(
    client: AsyncOpenAI,
    dataset: TrainingDataset,
    concurrency: int = 10
) -> TrainingDataset:
    """Add reasoning traces to examples that lack them."""
    semaphore = asyncio.Semaphore(concurrency)

    async def enrich_one(ex: TrainingExample) -> TrainingExample:
        if ex.reasoning_trace:
            return ex
        async with semaphore:
            reasoning = await generate_reasoning(
                client, ex.problem_statement, ex.solution_code
            )
            return TrainingExample(
                problem_id=ex.problem_id,
                problem_statement=ex.problem_statement,
                solution_code=ex.solution_code,
                reasoning_trace=reasoning,
                domain=ex.domain,
                difficulty=ex.difficulty,
                test_results=ex.test_results
            )

    enriched = await asyncio.gather(*[enrich_one(ex) for ex in dataset.examples])
    return TrainingDataset(dataset.specialist_name, list(enriched))
```

## Training Configuration

### python-lora

```yaml
# configs/python_lora.yaml
adapter_name: python-lora
base_model: Qwen/QwQ-32B-Preview-AWQ

# LoRA configuration
lora:
  r: 32                    # Rank
  lora_alpha: 64           # Scaling factor (2x rank is common)
  lora_dropout: 0.05
  target_modules:          # Which layers to adapt
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: none
  task_type: CAUSAL_LM

# Training hyperparameters
training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 8  # Effective batch size = 32
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  weight_decay: 0.01
  max_seq_length: 4096

  # Memory optimization
  gradient_checkpointing: true
  optim: adamw_8bit        # 8-bit optimizer
  bf16: true

# Data configuration
data:
  train_file: data/python_train.jsonl
  eval_file: data/python_eval.jsonl
  domains:
    - python
    - debugging
    - testing
  min_examples: 5000

# Validation
validation:
  eval_steps: 500
  metric: pass@1           # Primary metric for competitive programming
  early_stopping_patience: 3
```

### math-lora

```yaml
# configs/math_lora.yaml
adapter_name: math-lora
base_model: Qwen/QwQ-32B-Preview-AWQ

lora:
  r: 64                    # Higher rank for reasoning-heavy tasks
  lora_alpha: 128
  lora_dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  bias: none
  task_type: CAUSAL_LM

training:
  num_epochs: 5            # More epochs for reasoning
  batch_size: 2            # Smaller batch, longer sequences
  gradient_accumulation_steps: 16
  learning_rate: 1e-4      # Lower LR for stability
  lr_scheduler: cosine
  warmup_ratio: 0.05
  weight_decay: 0.01
  max_seq_length: 8192     # Longer context for proofs

  gradient_checkpointing: true
  optim: adamw_8bit
  bf16: true

data:
  train_file: data/math_train.jsonl
  eval_file: data/math_eval.jsonl
  domains:
    - math
    - algorithms
    - proofs
  min_examples: 3000

validation:
  eval_steps: 300
  metric: pass@1
  early_stopping_patience: 5
```

## Data Generation

Generate the JSONL files referenced in configs:

```python
#!/usr/bin/env python3
"""Generate training data for LoRA specialists from competitive programming datasets."""

import json
from pathlib import Path

from datasets import load_dataset


def generate_training_data(output_dir: str = "data"):
    """Download datasets and generate domain-specific JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load datasets
    codecontests = load_dataset("deepmind/code_contests", split="train")
    apps = load_dataset("codeparrot/apps", split="train")

    # Collect examples by domain
    python_examples = []
    math_examples = []

    # Process CodeContests
    for problem in codecontests:
        solutions = problem.get("solutions", {}).get("python3", [])
        if not solutions:
            continue

        domain = classify_problem(problem["description"], problem.get("cf_tags", []))
        example = {
            "problem": problem["description"],
            "reasoning": "",  # Will be enriched later
            "solution": solutions[0],  # Use first correct solution
            "domain": domain.value,
            "difficulty": problem.get("difficulty", "unknown")
        }

        if domain in (Domain.PYTHON, Domain.DATA_STRUCTURES):
            python_examples.append(example)
        elif domain in (Domain.MATH, Domain.ALGORITHMS):
            math_examples.append(example)

    # Process APPS (all Python)
    for problem in apps:
        if problem.get("solutions"):
            solutions = json.loads(problem["solutions"])
            if solutions:
                python_examples.append({
                    "problem": problem["question"],
                    "reasoning": "",
                    "solution": solutions[0],
                    "domain": "python",
                    "difficulty": problem.get("difficulty", "unknown")
                })

    # Split train/eval (90/10)
    def split_and_save(examples: list, prefix: str):
        split_idx = int(len(examples) * 0.9)
        train, eval_set = examples[:split_idx], examples[split_idx:]

        train_path = output_path / f"{prefix}_train.jsonl"
        eval_path = output_path / f"{prefix}_eval.jsonl"

        with open(train_path, "w") as f:
            for ex in train:
                f.write(json.dumps(ex) + "\n")

        with open(eval_path, "w") as f:
            for ex in eval_set:
                f.write(json.dumps(ex) + "\n")

        print(f"{prefix}: {len(train)} train, {len(eval_set)} eval")

    split_and_save(python_examples, "python")
    split_and_save(math_examples, "math")


if __name__ == "__main__":
    generate_training_data()
```

Run before training:
```bash
python generate_data.py
# Output:
# python: 8500 train, 944 eval
# math: 2700 train, 300 eval
```

## Training Script

```python
#!/usr/bin/env python3
"""Train a LoRA adapter for hi_moe specialists."""

import argparse
import logging
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_quantization() -> BitsAndBytesConfig:
    """Configure 4-bit quantization for training."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


def setup_lora(config: dict) -> LoraConfig:
    """Create LoRA configuration from config dict."""
    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"]
    )


def format_training_example(example: dict) -> str:
    """Format dataset example into training prompt."""
    return f"""<|im_start|>system
You are a specialist. Solve the problem step by step, then provide working code.
<|im_end|>
<|im_start|>user
{example['problem']}
<|im_end|>
<|im_start|>assistant
{example['reasoning']}

```python
{example['solution']}
```
<|im_end|>"""


def train(config_path: str, output_dir: str):
    """Main training function."""
    config = load_config(config_path)
    train_cfg = config["training"]
    data_cfg = config["data"]

    logger.info(f"Training {config['adapter_name']} with rank {config['lora']['r']}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load quantized base model
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=setup_quantization(),
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_cfg["train_file"],
            "eval": data_cfg["eval_file"]
        }
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        optim=train_cfg["optim"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config["validation"]["eval_steps"],
        save_strategy="steps",
        save_steps=config["validation"]["eval_steps"],
        load_best_model_at_end=True,
        report_to=["tensorboard"]
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        formatting_func=format_training_example,
        max_seq_length=train_cfg["max_seq_length"],
        tokenizer=tokenizer
    )

    # Train
    trainer.train()

    # Save adapter
    adapter_path = Path(output_dir) / "final"
    model.save_pretrained(adapter_path)
    logger.info(f"Saved adapter to {adapter_path}")

    return adapter_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(args.config, args.output)
```

## Validation Pipeline

### Pass@k Evaluation

```python
import asyncio
import logging
from dataclasses import dataclass

from code_runner import CodeRunner, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    problem_id: str
    passed: bool
    solutions_tried: int
    first_pass_index: int | None  # Which attempt first passed
    error_types: list[str]


async def evaluate_pass_at_k(
    adapter_name: str,
    eval_problems: list[dict],
    k: int = 5,
    runner: CodeRunner | None = None
) -> dict:
    """
    Evaluate pass@k on a set of problems.

    Args:
        adapter_name: Which adapter to use for generation
        eval_problems: List of problems with test cases
        k: Number of attempts per problem
        runner: CodeRunner instance (created if not provided)

    Returns:
        {"pass@1": 0.45, "pass@5": 0.72, "details": [...]}
    """
    if runner is None:
        runner = CodeRunner()

    results: list[ValidationResult] = []

    for problem in eval_problems:
        problem_passed = False
        first_pass = None
        errors = []
        attempts_made = 0

        for attempt in range(k):
            attempts_made = attempt + 1

            # Generate solution
            solution = await generate_solution(
                adapter_name,
                problem["statement"],
                temperature=0.8  # Higher temp for diversity
            )

            # Execute against test cases
            exec_result = await runner.execute(
                code=solution,
                language="python",
                test_input=problem["test_input"],
                expected_output=problem["expected_output"],
                timeout_ms=5000
            )

            if exec_result.passed:
                if not problem_passed:
                    first_pass = attempt
                problem_passed = True
                break
            else:
                errors.append(exec_result.error_type)

        results.append(ValidationResult(
            problem_id=problem["id"],
            passed=problem_passed,
            solutions_tried=attempts_made,
            first_pass_index=first_pass,
            error_types=errors
        ))

    # Calculate metrics
    pass_at_1 = sum(1 for r in results if r.first_pass_index == 0) / len(results)
    pass_at_k = sum(1 for r in results if r.passed) / len(results)

    return {
        "pass@1": pass_at_1,
        f"pass@{k}": pass_at_k,
        "total_problems": len(results),
        "details": results
    }


async def generate_solution(
    adapter_name: str,
    problem: str,
    temperature: float = 0.3
) -> str:
    """Generate a solution using specified adapter."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="http://localhost:8000/v1")

    try:
        response = await client.chat.completions.create(
            model=adapter_name,
            messages=[{
                "role": "system",
                "content": "You are a competitive programming specialist. Provide clean, working Python code."
            }, {
                "role": "user",
                "content": problem
            }],
            temperature=temperature,
            max_tokens=2048
        )

        # Extract code from response
        content = response.choices[0].message.content
        return extract_code_block(content)
    except Exception as e:
        logger.error(f"Failed to generate solution: {e}")
        raise


def extract_code_block(text: str) -> str:
    """Extract Python code from markdown response."""
    import re

    # Try explicit python code block first
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try generic code block (no language tag)
    match = re.search(r"```\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: assume entire response is code
    return text.strip()
```

### Baseline Comparison

Always compare against base model:

```python
async def compare_to_baseline(
    adapter_name: str,
    eval_problems: list[dict],
    k: int = 5
) -> dict:
    """Compare adapter performance against base model."""
    # Evaluate adapter
    adapter_results = await evaluate_pass_at_k(
        adapter_name, eval_problems, k
    )

    # Evaluate base model (no adapter)
    base_results = await evaluate_pass_at_k(
        "Qwen/QwQ-32B-Preview-AWQ", eval_problems, k
    )

    improvement = {
        "pass@1_delta": adapter_results["pass@1"] - base_results["pass@1"],
        f"pass@{k}_delta": adapter_results[f"pass@{k}"] - base_results[f"pass@{k}"],
        "adapter": adapter_results,
        "baseline": base_results
    }

    return improvement
```

## Deployment Process

### 1. Train Adapter

```bash
# Train python-lora
python train_lora.py \
    --config configs/python_lora.yaml \
    --output /adapters/python-specialist/v1.0.0

# Train math-lora
python train_lora.py \
    --config configs/math_lora.yaml \
    --output /adapters/math-specialist/v1.0.0
```

### 2. Validate Before Deployment

```python
async def validate_for_deployment(
    adapter_path: str,
    min_pass_at_1: float = 0.35,
    min_improvement: float = 0.05
) -> tuple[bool, dict]:
    """Check if adapter is ready for deployment."""
    # Load validation set
    eval_problems = load_validation_set()

    # Run comparison
    results = await compare_to_baseline(
        adapter_path, eval_problems, k=5
    )

    # Check thresholds
    ready = (
        results["adapter"]["pass@1"] >= min_pass_at_1 and
        results["pass@1_delta"] >= min_improvement
    )

    return ready, results
```

### 3. Register in Beads

```python
from beads_client import BeadsClient
from datetime import datetime


async def register_adapter(
    beads: BeadsClient,
    adapter_name: str,
    adapter_path: str,
    validation_results: dict
):
    """Register a new adapter version in the registry."""
    # Get next ID
    next_id = await beads.increment("system/adapters/next_id")

    # Get current active version for rollback chain
    registry = await beads.get("system/adapters/registry") or {}
    current = registry.get(adapter_name)
    previous_version = current["version"] if current else None

    # Create registry entry
    entry = {
        "name": adapter_name,
        "lora_int_id": next_id,
        "path": adapter_path,
        "rank": 32 if "python" in adapter_name else 64,
        "domains": get_domains_for_adapter(adapter_name),
        "base_model": "Qwen/QwQ-32B-Preview-AWQ",
        "version": extract_version(adapter_path),
        "created_at": datetime.now().isoformat(),
        "previous_version": previous_version,
        "avg_latency_ms": 0,
        "success_rate": validation_results["adapter"]["pass@1"],
        "request_count": 0,
        "active": False,  # Start inactive for A/B testing
        "last_used": None,
        "validation": validation_results
    }

    # Save to registry
    registry[adapter_name] = entry
    await beads.set("system/adapters/registry", registry)

    return entry


def get_domains_for_adapter(name: str) -> list[str]:
    """Map adapter name to domains."""
    mapping = {
        "python-lora": ["python", "debugging", "testing", "refactoring"],
        "math-lora": ["math", "algorithms", "proofs", "number_theory"]
    }
    return mapping.get(name, [])


def extract_version(path: str) -> str:
    """Extract version from adapter path."""
    import re
    match = re.search(r"v(\d+\.\d+\.\d+)", path)
    return match.group(1) if match else "1.0.0"
```

### 4. A/B Testing

```python
async def enable_ab_test(
    beads: BeadsClient,
    adapter_name: str,
    traffic_percentage: float = 0.1
):
    """Route a percentage of traffic to new adapter version."""
    await beads.set(f"system/adapters/{adapter_name}/ab_test", {
        "enabled": True,
        "traffic_percentage": traffic_percentage,
        "started_at": datetime.now().isoformat(),
        "metrics": {
            "requests": 0,
            "successes": 0
        }
    })
```

### 5. Promote to Production

```python
async def promote_adapter(beads: BeadsClient, adapter_name: str):
    """Promote adapter to active after successful A/B test."""
    registry = await beads.get("system/adapters/registry")

    # Deactivate old version
    for name, entry in registry.items():
        if name == adapter_name and entry.get("active"):
            entry["active"] = False

    # Activate new version
    registry[adapter_name]["active"] = True

    await beads.set("system/adapters/registry", registry)

    # Disable A/B test
    await beads.delete(f"system/adapters/{adapter_name}/ab_test")
```

## Training Schedule

### Initial Training (Week 1)

1. **Data Preparation**
   - Download CodeContests, APPS, TACO datasets
   - Run domain classification on all problems
   - Generate reasoning traces for solutions lacking them
   - Split into train/eval (90/10)

2. **python-lora Training**
   - 5000+ examples from Python-classified problems
   - Train for 3 epochs (~8 hours on A100)
   - Target: pass@1 > 0.40

3. **math-lora Training**
   - 3000+ examples from math/algorithm problems
   - Train for 5 epochs (~12 hours on A100)
   - Target: pass@1 > 0.35

### Validation Gate

Before any adapter reaches production:

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| pass@1 | > 0.35 | Must beat random baseline |
| pass@1 improvement | > 5% | Must exceed base model |
| Latency increase | < 10% | Shouldn't slow inference |
| Error rate | < 5% | No regressions in valid code |

### Continuous Improvement

After initial deployment:

1. **Collect production data** - Successful executions become training candidates
2. **Filter by confidence** - Only use results with confidence > 0.9
3. **Monthly retraining** - Incorporate new examples
4. **A/B test** - Every new version validated against current

## Resource Requirements

### Training

| Resource | python-lora | math-lora |
|----------|-------------|-----------|
| GPU | 1x A100 40GB | 1x A100 40GB |
| Training time | ~8 hours | ~12 hours |
| Peak VRAM | ~35GB | ~38GB |
| Storage | ~5GB | ~8GB |

### Validation

| Resource | Requirement |
|----------|-------------|
| GPU | 1x A100 40GB (inference) |
| Time per problem | ~10s (5 attempts) |
| Full eval (500 problems) | ~1.5 hours |

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Memory-efficient fine-tuning
- [PEFT Documentation](https://huggingface.co/docs/peft) - LoRA implementation
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) - Supervised fine-tuning
- [CodeContests Paper](https://arxiv.org/abs/2203.07814) - Competition-level dataset
