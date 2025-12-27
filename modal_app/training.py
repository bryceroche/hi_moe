"""LoRA training jobs on Modal."""
from __future__ import annotations

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
        "trl==0.9.6",  # Pin to stable version with known API
        "datasets",
        "bitsandbytes",
        "accelerate",
        "rich",  # Required by TRL
    )
)

DATA_PATH = "/data"
ADAPTERS_PATH = "/adapters"

# Domain-specific specialist types
SPECIALIST_TYPES = {
    "python": "Python programming",
    "math": "mathematical reasoning",
    "algorithms": "algorithm design",
    "data_structures": "data structure",
}


@app.function(
    gpu="A100-80GB",  # Need 80GB for Qwen3-32B training
    image=training_image,
    volumes={
        DATA_PATH: data_volume,
        ADAPTERS_PATH: adapter_volume,
    },
    timeout=86400,  # 24 hours max
    # HuggingFace secret optional for public models like QwQ
)
def train_lora(
    adapter_name: str,
    train_file: str,
    eval_file: str,
    rank: int = 16,  # Reduced for memory
    epochs: int = 3,
    batch_size: int = 1,  # Reduced for memory
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

    # Load model - must match vllm_server.py base model for LoRA compatibility
    # Using Qwen3-32B (same architecture as Qwen3-32B-AWQ used in inference)
    model_id = "Qwen/Qwen3-32B"
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

    # Pre-format examples into "text" field for SFTTrainer
    def format_example(example):
        domain = example.get("domain", "")
        specialist_type = SPECIALIST_TYPES.get(domain, "programming")
        text = f"""<|im_start|>system
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
        return {"text": text}

    # Apply formatting to create "text" column
    dataset = dataset.map(format_example)
    print(f"Formatted dataset: {dataset}")

    # Training args
    output_dir = f"{ADAPTERS_PATH}/{adapter_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16,  # Increased to compensate for batch=1
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

    # Train with pre-formatted "text" field
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="text",
        max_seq_length=1024,  # Reduced from 4096 for memory
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
    rank: int = 16,  # Match function default for memory safety
    epochs: int = 3,
):
    """Local entrypoint to trigger training."""
    result = train_lora.remote(
        adapter_name=adapter_name,
        train_file=train_file,
        eval_file=eval_file,
        rank=rank,
        epochs=epochs,
    )
    print(f"Training complete: {result}")
