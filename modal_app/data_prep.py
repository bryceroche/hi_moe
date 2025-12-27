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
    timeout=1800,
)
def prepare_gsm8k_dataset(max_samples: int = 500) -> dict:
    """Download and format GSM8K (Grade School Math 8K) for training.

    GSM8K contains ~8K grade school math word problems with step-by-step solutions.
    """
    import json
    import os
    import re
    from datasets import load_dataset

    print("Downloading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    os.makedirs(DATA_PATH, exist_ok=True)

    train_data = []
    eval_data = []

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break

        # GSM8K has 'question' and 'answer' fields
        # Answer contains step-by-step reasoning ending with "#### <final_answer>"
        question = example["question"]
        answer = example["answer"]

        # Extract the final numeric answer
        final_match = re.search(r"####\s*([\d,.-]+)", answer)
        final_answer = final_match.group(1).replace(",", "") if final_match else ""

        # Extract reasoning (everything before ####)
        reasoning = re.sub(r"####.*$", "", answer).strip()

        # Format for training
        entry = {
            "domain": "math",
            "problem": question,
            "reasoning": reasoning,
            "solution": f"The answer is {final_answer}",
        }

        # 90/10 train/eval split
        if i % 10 == 0:
            eval_data.append(entry)
        else:
            train_data.append(entry)

    # Write files
    train_file = f"{DATA_PATH}/math_train.jsonl"
    eval_file = f"{DATA_PATH}/math_eval.jsonl"

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
        "train_file": "math_train.jsonl",
        "eval_file": "math_eval.jsonl",
    }


@app.function(
    image=prep_image,
    volumes={DATA_PATH: data_volume},
    timeout=1800,
)
def prepare_code_contests_dataset(max_samples: int = 200) -> dict:
    """Download and format CodeContests for algorithm training.

    CodeContests contains competitive programming problems with solutions.
    Language codes: 1=C++, 2=C++, 3=Python3, 4=Python, 5=Java, etc.
    """
    import json
    import os
    from datasets import load_dataset

    print("Downloading CodeContests dataset...")
    # Use the train split which has more data
    dataset = load_dataset("deepmind/code_contests", split="train")

    os.makedirs(DATA_PATH, exist_ok=True)

    train_data = []
    eval_data = []
    sample_count = 0

    # Language codes for Python
    PYTHON_LANGS = {3, 4}  # Python3 and Python

    for example in dataset:
        if sample_count >= max_samples:
            break

        # Get problem description
        problem = example["description"]
        if not problem or len(problem) < 50:
            continue

        # Get solutions - structure is {language: [int], solution: [str]}
        solutions = example.get("solutions", {})
        languages = solutions.get("language", [])
        codes = solutions.get("solution", [])

        # Find Python solutions
        python_solution = None
        for lang, code in zip(languages, codes):
            if lang in PYTHON_LANGS and code and len(code) > 20:
                python_solution = code
                break

        if not python_solution:
            continue  # Skip if no Python solution

        # Format for training
        entry = {
            "domain": "algorithms",
            "problem": problem[:2000],  # Truncate long problems
            "reasoning": f"This is a competitive programming problem. Let me analyze the requirements and implement an efficient solution.",
            "solution": python_solution,
        }

        # 90/10 train/eval split
        if sample_count % 10 == 0:
            eval_data.append(entry)
        else:
            train_data.append(entry)

        sample_count += 1

    # Write files
    train_file = f"{DATA_PATH}/algorithms_train.jsonl"
    eval_file = f"{DATA_PATH}/algorithms_eval.jsonl"

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
        "train_file": "algorithms_train.jsonl",
        "eval_file": "algorithms_eval.jsonl",
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
def main(action: str = "prepare", dataset: str = "python", max_samples: int = 500):
    """Prepare training data.

    Actions:
        prepare: Download and format a dataset
        list: List files in data volume
        preview: Preview data file contents
        all: Prepare all datasets

    Datasets:
        python: MBPP Python problems
        math: GSM8K math word problems
        algorithms: CodeContests competitive programming

    Examples:
        modal run modal_app/data_prep.py --action=prepare --dataset=python --max-samples=100
        modal run modal_app/data_prep.py --action=prepare --dataset=math
        modal run modal_app/data_prep.py --action=prepare --dataset=algorithms
        modal run modal_app/data_prep.py --action=all
        modal run modal_app/data_prep.py --action=list
    """
    if action == "prepare":
        if dataset == "python":
            result = prepare_mbpp_dataset.remote(max_samples)
        elif dataset == "math":
            result = prepare_gsm8k_dataset.remote(max_samples)
        elif dataset == "algorithms":
            result = prepare_code_contests_dataset.remote(max_samples)
        else:
            print(f"Unknown dataset: {dataset}")
            print("Available: python, math, algorithms")
            return

        print(f"\nDataset prepared:")
        print(f"  Train: {result['train_samples']} samples -> {result['train_file']}")
        print(f"  Eval: {result['eval_samples']} samples -> {result['eval_file']}")

    elif action == "all":
        print("Preparing all datasets...")
        for ds_name, prep_fn in [
            ("python", prepare_mbpp_dataset),
            ("math", prepare_gsm8k_dataset),
            ("algorithms", prepare_code_contests_dataset),
        ]:
            print(f"\n--- {ds_name.upper()} ---")
            result = prep_fn.remote(max_samples)
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
        print("Valid actions: prepare, all, list, preview")
