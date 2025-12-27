"""Prepare GSM8K dataset for math specialist training.

Downloads GSM8K from HuggingFace and formats it for our training pipeline.
Output format matches the training.py expected schema:
- domain: "math"
- problem: the math word problem
- reasoning: step-by-step solution
- solution: Python code that computes the answer
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset


def extract_reasoning_and_answer(answer_text: str) -> tuple[str, str]:
    """Extract reasoning steps and final numerical answer from GSM8K format.

    GSM8K format: "Step 1... Step 2... #### 42"
    Returns: (reasoning_text, final_answer)
    """
    # Split on #### to get reasoning and answer
    parts = answer_text.split("####")
    if len(parts) == 2:
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
    else:
        # Fallback: use whole text as reasoning, try to extract number
        reasoning = answer_text.strip()
        # Try to find a number at the end
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        final_answer = numbers[-1] if numbers else "0"

    return reasoning, final_answer


def format_solution_code(answer: str) -> str:
    """Create a simple Python solution that produces the answer.

    This keeps the format consistent with the code-based training examples.
    """
    # Clean the answer (remove commas, spaces)
    clean_answer = answer.replace(",", "").strip()

    # Determine if it's an integer or float
    try:
        if "." in clean_answer:
            num = float(clean_answer)
            return f"answer = {num}\nprint(answer)"
        else:
            num = int(clean_answer)
            return f"answer = {num}\nprint(answer)"
    except ValueError:
        # If we can't parse, just use string
        return f'answer = "{clean_answer}"\nprint(answer)'


def prepare_gsm8k(
    output_dir: Path,
    train_samples: int = 500,
    eval_samples: int = 50,
) -> dict:
    """Download and prepare GSM8K dataset.

    Args:
        output_dir: Directory to write output files
        train_samples: Number of training examples
        eval_samples: Number of eval examples

    Returns:
        Dict with stats about the prepared data
    """
    print("Loading GSM8K dataset from HuggingFace...")
    dataset = load_dataset("gsm8k", "main")

    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"GSM8K train size: {len(train_data)}")
    print(f"GSM8K test size: {len(test_data)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process training data
    train_file = output_dir / "math_train.jsonl"
    train_count = 0

    with open(train_file, "w") as f:
        for i, example in enumerate(train_data):
            if i >= train_samples:
                break

            reasoning, answer = extract_reasoning_and_answer(example["answer"])
            solution_code = format_solution_code(answer)

            formatted = {
                "domain": "math",
                "problem": example["question"],
                "reasoning": reasoning,
                "solution": solution_code,
            }
            f.write(json.dumps(formatted) + "\n")
            train_count += 1

    print(f"Wrote {train_count} training examples to {train_file}")

    # Process eval data (from test split)
    eval_file = output_dir / "math_eval.jsonl"
    eval_count = 0

    with open(eval_file, "w") as f:
        for i, example in enumerate(test_data):
            if i >= eval_samples:
                break

            reasoning, answer = extract_reasoning_and_answer(example["answer"])
            solution_code = format_solution_code(answer)

            formatted = {
                "domain": "math",
                "problem": example["question"],
                "reasoning": reasoning,
                "solution": solution_code,
            }
            f.write(json.dumps(formatted) + "\n")
            eval_count += 1

    print(f"Wrote {eval_count} eval examples to {eval_file}")

    return {
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "train_count": train_count,
        "eval_count": eval_count,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare GSM8K for training")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data"),
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=500,
        help="Number of training samples",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="Number of eval samples",
    )

    args = parser.parse_args()

    stats = prepare_gsm8k(
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
    )

    print("\nDone! Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
