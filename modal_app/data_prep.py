"""Prepare training data for LoRA fine-tuning."""
from __future__ import annotations

import modal

app = modal.App("hi-moe-data-prep")

data_volume = modal.Volume.from_name("hi-moe-data", create_if_missing=True)
DATA_PATH = "/data"

prep_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("datasets", "huggingface_hub")
)


@app.function(
    image=prep_image,
    volumes={DATA_PATH: data_volume},
    timeout=1800,
)
def prepare_mbpp_dataset(max_samples: int = 500) -> dict:
    """Download and format MBPP (Mostly Basic Python Problems) for training.

    MBPP is a benchmark of ~1000 Python programming problems with solutions.
    """
    import json
    import os
    from datasets import load_dataset

    print("Downloading MBPP dataset...")
    dataset = load_dataset("mbpp", split="train")

    os.makedirs(DATA_PATH, exist_ok=True)

    train_data = []
    eval_data = []

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        # Format for training
        entry = {
            "domain": "python",
            "problem": example["text"],  # Problem description
            "reasoning": f"Let me solve this step by step.\n\nThe task is: {example['text']}\n\nI'll write a Python function to solve this.",
            "solution": example["code"],  # Solution code
        }

        # 90/10 train/eval split
        if i % 10 == 0:
            eval_data.append(entry)
        else:
            train_data.append(entry)

    # Write files
    train_file = f"{DATA_PATH}/python_train.jsonl"
    eval_file = f"{DATA_PATH}/python_eval.jsonl"

    with open(train_file, "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry) + "\n")

    with open(eval_file, "w") as f:
        for entry in eval_data:
            f.write(json.dumps(entry) + "\n")

    # Commit volume
    data_volume.commit()

    print(f"Created {len(train_data)} training examples")
    print(f"Created {len(eval_data)} eval examples")
    print(f"Files: {train_file}, {eval_file}")

    return {
        "train_samples": len(train_data),
        "eval_samples": len(eval_data),
        "train_file": "python_train.jsonl",
        "eval_file": "python_eval.jsonl",
    }


@app.function(
    image=prep_image,
    volumes={DATA_PATH: data_volume},
)
def list_data_files() -> list[dict]:
    """List files in the data volume."""
    import os

    files = []
    if os.path.exists(DATA_PATH):
        for name in os.listdir(DATA_PATH):
            path = f"{DATA_PATH}/{name}"
            if os.path.isfile(path):
                size = os.path.getsize(path)
                # Count lines
                with open(path) as f:
                    lines = sum(1 for _ in f)
                files.append({"name": name, "size": size, "lines": lines})

    return files


@app.function(
    image=prep_image,
    volumes={DATA_PATH: data_volume},
)
def preview_data(filename: str, n: int = 3) -> list[dict]:
    """Preview entries from a data file."""
    import json

    entries = []
    with open(f"{DATA_PATH}/{filename}") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            entries.append(json.loads(line))

    return entries


@app.local_entrypoint()
def main(action: str = "prepare", max_samples: int = 500):
    """Prepare training data.

    Actions:
        prepare: Download and format MBPP dataset
        list: List files in data volume
        preview: Preview data file contents

    Examples:
        modal run modal_app/data_prep.py --action=prepare --max-samples=100
        modal run modal_app/data_prep.py --action=list
    """
    if action == "prepare":
        result = prepare_mbpp_dataset.remote(max_samples)
        print(f"\nDataset prepared:")
        print(f"  Train: {result['train_samples']} samples -> {result['train_file']}")
        print(f"  Eval: {result['eval_samples']} samples -> {result['eval_file']}")

    elif action == "list":
        files = list_data_files.remote()
        if not files:
            print("No data files found.")
        else:
            print(f"Found {len(files)} file(s):")
            for f in files:
                print(f"  - {f['name']}: {f['lines']} lines ({f['size']} bytes)")

    elif action == "preview":
        entries = preview_data.remote("python_train.jsonl", 2)
        for i, entry in enumerate(entries):
            print(f"\n--- Example {i+1} ---")
            print(f"Domain: {entry['domain']}")
            print(f"Problem: {entry['problem'][:100]}...")
            print(f"Solution: {entry['solution'][:100]}...")

    else:
        print(f"Unknown action: {action}")
