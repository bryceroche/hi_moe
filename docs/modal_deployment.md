# Modal Deployment Specification

> Deploy hi_moe inference infrastructure on Modal with vLLM and LoRA support.

## Overview

Modal provides serverless GPU infrastructure with:
- A100 40GB/80GB GPUs
- Pay-per-second billing (no idle costs)
- Simple Python SDK
- Built-in secrets management
- Container caching for fast cold starts

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Modal                             │
│  ┌─────────────────────────────────────────────────┐    │
│  │              vLLM Server (A100)                  │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  Qwen/QwQ-32B-Preview-AWQ (base model)  │    │    │
│  │  │  + LoRA adapters (python, math, etc.)   │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
│                          ▲                               │
│                          │ OpenAI-compatible API         │
└──────────────────────────┼──────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────┐
│                     hi_moe Harness                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │
│  │ Progress │→│ Architect │→│Dispatcher │→│  Fleet  │  │
│  │ Monitor  │  │          │  │          │  │         │  │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Modal Setup

### Installation

```bash
pip install modal
modal token new  # Authenticate with Modal
```

### Project Structure

```
hi_moe/
├── modal_app/
│   ├── __init__.py
│   ├── vllm_server.py      # vLLM deployment
│   ├── training.py         # LoRA training jobs
│   └── config.py           # Shared configuration
├── adapters/               # LoRA adapter weights (Modal Volume)
└── data/                   # Training data (Modal Volume)
```

## vLLM Server Deployment

### Server Definition

```python
# modal_app/vllm_server.py
import modal

app = modal.App("hi-moe-inference")

# Persistent volume for model weights and adapters
model_volume = modal.Volume.from_name("hi-moe-models", create_if_missing=True)
adapter_volume = modal.Volume.from_name("hi-moe-adapters", create_if_missing=True)

# Container image with vLLM
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MODEL_ID = "Qwen/QwQ-32B-Preview-AWQ"
ADAPTERS_PATH = "/adapters"
MODEL_PATH = "/models"


@app.cls(
    gpu=modal.gpu.A100(size="40GB"),
    image=vllm_image,
    volumes={
        MODEL_PATH: model_volume,
        ADAPTERS_PATH: adapter_volume,
    },
    timeout=3600,  # 1 hour max request time
    container_idle_timeout=300,  # Keep warm for 5 min
    allow_concurrent_inputs=100,  # Batch requests
)
class VLLMServer:
    @modal.build()
    def download_model(self):
        """Download model weights at build time (cached in image)."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            MODEL_ID,
            local_dir=f"{MODEL_PATH}/{MODEL_ID}",
            ignore_patterns=["*.md", "*.txt"],
        )

    @modal.enter()
    def start_server(self):
        """Initialize vLLM engine on container start."""
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        import os

        # Discover available adapters
        self.adapters = {}
        if os.path.exists(ADAPTERS_PATH):
            for adapter_name in os.listdir(ADAPTERS_PATH):
                adapter_path = f"{ADAPTERS_PATH}/{adapter_name}"
                if os.path.isdir(adapter_path):
                    self.adapters[adapter_name] = adapter_path

        # Initialize vLLM with LoRA support
        self.llm = LLM(
            model=f"{MODEL_PATH}/{MODEL_ID}",
            quantization="awq",
            enable_lora=True,
            max_loras=16,
            max_lora_rank=64,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
        )

        self.lora_id_counter = 1
        self.lora_requests = {}

        # Pre-register discovered adapters
        for name, path in self.adapters.items():
            self.lora_requests[name] = LoRARequest(
                lora_name=name,
                lora_int_id=self.lora_id_counter,
                lora_local_path=path,
            )
            self.lora_id_counter += 1
            print(f"Registered adapter: {name}")

    @modal.method()
    def generate(
        self,
        prompt: str,
        adapter_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> dict:
        """Generate completion with optional LoRA adapter."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
        )

        lora_request = None
        if adapter_name and adapter_name in self.lora_requests:
            lora_request = self.lora_requests[adapter_name]

        outputs = self.llm.generate(
            [prompt],
            sampling_params,
            lora_request=lora_request,
        )

        output = outputs[0]
        return {
            "text": output.outputs[0].text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "adapter_used": adapter_name,
            "finish_reason": output.outputs[0].finish_reason,
        }

    @modal.method()
    def chat(
        self,
        messages: list[dict],
        adapter_name: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> dict:
        """Chat completion with ChatML formatting."""
        # Format messages as ChatML
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return self.generate(
            prompt=prompt,
            adapter_name=adapter_name,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>"],
        )

    @modal.method()
    def list_adapters(self) -> list[str]:
        """List available LoRA adapters."""
        return list(self.lora_requests.keys())

    @modal.method()
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": MODEL_ID,
            "adapters": list(self.lora_requests.keys()),
            "max_model_len": 8192,
        }


    @modal.asgi_app()
    def serve(self):
        """Serve OpenAI-compatible API endpoint on the same container."""
        from fastapi import FastAPI
        from pydantic import BaseModel

        api = FastAPI(title="hi-moe vLLM API")

        class ChatRequest(BaseModel):
            model: str = "base"  # "base" or adapter name
            messages: list[dict]
            max_tokens: int = 2048
            temperature: float = 0.7

        class ChatResponse(BaseModel):
            id: str
            choices: list[dict]
            usage: dict

        @api.post("/v1/chat/completions")
        def chat_completions(request: ChatRequest) -> ChatResponse:
            adapter = None if request.model == "base" else request.model

            result = self.chat(
                messages=request.messages,
                adapter_name=adapter,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            return ChatResponse(
                id="chatcmpl-" + format(abs(hash(result["text"])), "x")[:8],
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    "finish_reason": result["finish_reason"],
                }],
                usage={
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                },
            )

        @api.get("/v1/models")
        def list_models():
            adapters = self.list_adapters()
            models = [{"id": "base", "object": "model"}]
            for adapter in adapters:
                models.append({"id": adapter, "object": "model"})
            return {"data": models}

        @api.get("/health")
        def health():
            return self.health()

        return api
```

### Deployment Commands

```bash
# Deploy the vLLM server
modal deploy modal_app/vllm_server.py

# Get the endpoint URL
modal app list  # Shows: https://bryce-roche--hi-moe-inference-serve.modal.run

# Test the deployment
curl https://bryce-roche--hi-moe-inference-serve.modal.run/health
```

## LoRA Training on Modal

### Training Job

```python
# modal_app/training.py
import modal

app = modal.App("hi-moe-training")

# Volumes for data and output
data_volume = modal.Volume.from_name("hi-moe-data", create_if_missing=True)
adapter_volume = modal.Volume.from_name("hi-moe-adapters", create_if_missing=True)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "trl",
        "datasets",
        "bitsandbytes",
        "accelerate",
    )
)

DATA_PATH = "/data"
ADAPTERS_PATH = "/adapters"


@app.function(
    gpu=modal.gpu.A100(size="40GB"),
    image=training_image,
    volumes={
        DATA_PATH: data_volume,
        ADAPTERS_PATH: adapter_volume,
    },
    timeout=86400,  # 24 hours max
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_lora(
    adapter_name: str,
    train_file: str,
    eval_file: str,
    rank: int = 32,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
):
    """Train a LoRA adapter on Modal A100."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"Training {adapter_name} with rank {rank}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model_id = "Qwen/QwQ-32B-Preview"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATA_PATH}/{train_file}",
            "eval": f"{DATA_PATH}/{eval_file}",
        },
    )

    # Format function with domain-specific specialist types
    SPECIALIST_TYPES = {
        "python": "Python programming",
        "math": "mathematical reasoning",
        "algorithms": "algorithm design",
        "data_structures": "data structure",
    }

    def format_example(example):
        specialist_type = SPECIALIST_TYPES.get(example.get("domain", ""), "programming")
        return f"""<|im_start|>system
You are a {specialist_type} specialist. Solve the problem step by step, then provide working code.
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

    # Training args
    output_dir = f"{ADAPTERS_PATH}/{adapter_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
    )

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        formatting_func=format_example,
        max_seq_length=4096,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save adapter (volume auto-commits on function exit)
    model.save_pretrained(output_dir)
    print(f"Saved adapter to {output_dir}")

    return {"adapter_name": adapter_name, "output_dir": output_dir}


@app.local_entrypoint()
def main(
    adapter_name: str = "python-lora",
    train_file: str = "python_train.jsonl",
    eval_file: str = "python_eval.jsonl",
    rank: int = 32,
):
    """Local entrypoint to trigger training."""
    result = train_lora.remote(
        adapter_name=adapter_name,
        train_file=train_file,
        eval_file=eval_file,
        rank=rank,
    )
    print(f"Training complete: {result}")
```

### Upload Training Data

```python
# scripts/upload_data.py
import modal

volume = modal.Volume.from_name("hi-moe-data", create_if_missing=True)

# Upload local files to Modal volume
with volume.batch_upload() as batch:
    batch.put_file("data/python_train.jsonl", "python_train.jsonl")
    batch.put_file("data/python_eval.jsonl", "python_eval.jsonl")
    batch.put_file("data/math_train.jsonl", "math_train.jsonl")
    batch.put_file("data/math_eval.jsonl", "math_eval.jsonl")

print("Data uploaded to Modal volume")
```

### Run Training

```bash
# Upload training data first
python scripts/upload_data.py

# Train python-lora
modal run modal_app/training.py --adapter-name python-lora --rank 32

# Train math-lora
modal run modal_app/training.py --adapter-name math-lora --train-file math_train.jsonl --eval-file math_eval.jsonl --rank 64
```

## Client Integration

### Python Client

```python
# hi_moe/modal_client.py
from dataclasses import dataclass
from openai import AsyncOpenAI


@dataclass
class ModalConfig:
    endpoint: str  # e.g., "https://bryce-roche--hi-moe-inference-serve.modal.run"
    api_key: str = "not-needed"  # Modal doesn't require API key for public endpoints


class ModalClient:
    """Client for hi_moe Modal deployment."""

    def __init__(self, config: ModalConfig):
        self.endpoint = config.endpoint.rstrip("/")
        self.client = AsyncOpenAI(
            base_url=f"{self.endpoint}/v1",
            api_key=config.api_key,
        )

    async def generate(
        self,
        messages: list[dict],
        adapter: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        """Generate completion, optionally using a LoRA adapter."""
        model = adapter or "base"

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content

    async def list_adapters(self) -> list[str]:
        """List available adapters."""
        models = await self.client.models.list()
        return [m.id for m in models.data if m.id != "base"]

    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        import httpx

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{self.endpoint}/health")
                return resp.status_code == 200
            except Exception:
                return False
```

### Integration with Specialized Fleet

```python
# hi_moe/fleet_executor.py
from modal_client import ModalClient, ModalConfig


class FleetExecutor:
    """Execute specialist tasks via Modal-hosted vLLM."""

    def __init__(self, modal_endpoint: str):
        self.client = ModalClient(ModalConfig(endpoint=modal_endpoint))

    async def execute(
        self,
        task: str,
        specialist: str | None = None,
        context: str = "",
    ) -> dict:
        """Execute a task with optional specialist adapter."""
        messages = [
            {"role": "system", "content": self._get_system_prompt(specialist)},
            {"role": "user", "content": f"{context}\n\n{task}" if context else task},
        ]

        response = await self.client.generate(
            messages=messages,
            adapter=specialist,
            temperature=0.3,
        )

        return {
            "response": response,
            "specialist_used": specialist,
        }

    def _get_system_prompt(self, specialist: str | None) -> str:
        prompts = {
            "python-lora": "You are a Python programming specialist. Write clean, efficient code.",
            "math-lora": "You are a mathematical reasoning specialist. Show your work step by step.",
            None: "You are a helpful AI assistant skilled in programming and problem-solving.",
        }
        return prompts.get(specialist, prompts[None])
```

## Cost Estimation

### Inference

| GPU | $/hour | Typical usage | Monthly estimate |
|-----|--------|---------------|------------------|
| A100 40GB | $3.00 | 100 hrs | $300 |
| A100 80GB | $4.50 | 100 hrs | $450 |

With container_idle_timeout=300s, you only pay when actively processing requests.

### Training

| Task | GPU hours | Cost |
|------|-----------|------|
| python-lora (3 epochs, 5K examples) | ~8 hrs | ~$24 |
| math-lora (5 epochs, 3K examples) | ~12 hrs | ~$36 |
| Total initial training | ~20 hrs | ~$60 |

## Deployment Checklist

### Initial Setup

```bash
# 1. Install Modal CLI
pip install modal

# 2. Authenticate
modal token new

# 3. Create secrets (if using HuggingFace private models)
modal secret create huggingface HF_TOKEN=your_token_here

# 4. Deploy vLLM server
modal deploy modal_app/vllm_server.py

# 5. Test endpoint
curl https://your-endpoint.modal.run/health
```

### First End-to-End Test

```python
# test_modal.py
import asyncio
from modal_client import ModalClient, ModalConfig


async def test():
    client = ModalClient(ModalConfig(
        endpoint="https://bryce-roche--hi-moe-inference-serve.modal.run"
    ))

    # Health check
    healthy = await client.health_check()
    print(f"Server healthy: {healthy}")

    # Test generation
    response = await client.generate(
        messages=[{"role": "user", "content": "Write a Python function to check if a number is prime."}],
        adapter=None,  # Use base model first
    )
    print(f"Response:\n{response}")


asyncio.run(test())
```

## Monitoring

### Modal Dashboard

Modal provides built-in monitoring at https://modal.com/apps:
- Request counts and latency
- GPU utilization
- Error rates
- Cost tracking

### Custom Metrics

Log to Beads for hi_moe-specific tracking:

```python
from datetime import datetime


async def log_request(beads, request_info: dict):
    await beads.append("system/metrics/modal_requests", {
        "timestamp": datetime.now().isoformat(),
        "adapter": request_info.get("adapter"),
        "latency_ms": request_info.get("latency_ms"),
        "tokens": request_info.get("total_tokens"),
    })
```

## References

- [Modal Documentation](https://modal.com/docs)
- [Modal vLLM Example](https://modal.com/docs/examples/vllm_inference)
- [vLLM LoRA Guide](https://docs.vllm.ai/en/stable/features/lora/)
