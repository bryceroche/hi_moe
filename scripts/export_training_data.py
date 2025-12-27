#!/usr/bin/env python3
"""Export training data from call_db for LoRA fine-tuning (hi_moe-juo).

Usage:
    python scripts/export_training_data.py --db runs/hi_moe.db --output training/

This script:
1. Exports successful examples from call_db
2. Formats them for training.py (problem/reasoning/solution/domain schema)
3. Splits into train/eval sets
4. Uploads to Modal data volume if requested
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from e2e_test.call_db import CallDB


def extract_code_from_output(output: str) -> str | None:
    """Extract Python code from model output."""
    import re

    # Try to find code in ```python blocks
    matches = re.findall(r"```python\s*(.*?)```", output, re.DOTALL)
    if matches:
        return matches[-1].strip()  # Return last code block

    # Try generic ``` blocks
    matches = re.findall(r"```\s*(.*?)```", output, re.DOTALL)
    if matches:
        return matches[-1].strip()

    return None


def export_from_jsonl(
    runs_dir: Path,
    output_dir: Path,
    eval_ratio: float = 0.1,
) -> dict[str, int]:
    """Export training data from JSONL trajectory files.

    This is the preferred method as JSONL files contain full output data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []

    # Find all JSONL trajectory files
    jsonl_files = list(runs_dir.rglob("run-*.jsonl"))
    print(f"Found {len(jsonl_files)} trajectory files")

    for jsonl_file in jsonl_files:
        try:
            entries = []
            with open(jsonl_file) as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            # Check if run succeeded (all tests passed)
            run_end = next((e for e in entries if e.get("type") == "run_end"), None)
            if not run_end or run_end.get("status") != "completed":
                continue

            # Get problem info from run_start
            run_start = next((e for e in entries if e.get("type") == "run_start"), None)
            if not run_start:
                continue

            problem_id = run_start.get("problem_id", "")

            # Get problem text from the first vllm_call input (user message)
            problem = ""
            vllm_calls = [e for e in entries if e.get("type") == "vllm_call"]
            if vllm_calls:
                first_input = vllm_calls[0].get("input", "")
                # Parse messages format
                if isinstance(first_input, list):
                    for msg in first_input:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            # Extract just the task description
                            if "Task:" in content:
                                problem = content.split("Task:")[1].split("Context:")[0].strip()
                            else:
                                problem = content
                            break
                elif isinstance(first_input, str):
                    problem = first_input

            # Get the final vllm_call with working code
            vllm_calls = [e for e in entries if e.get("type") == "vllm_call"]
            if not vllm_calls:
                continue

            # Use the last call (usually the one that produced working code)
            last_call = vllm_calls[-1]
            output = last_call.get("output", "")

            # Extract code from output
            code = extract_code_from_output(output)
            if not code:
                continue

            # Extract reasoning (text before code block)
            reasoning = output
            if "```python" in output:
                reasoning = output.split("```python")[0]
            elif "```" in output:
                reasoning = output.split("```")[0]

            # Clean up reasoning (remove <think> tags)
            import re
            reasoning = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()
            if reasoning.startswith("<think>"):
                # Handle unclosed think tags
                if "</think>" in reasoning:
                    reasoning = reasoning.split("</think>", 1)[-1].strip()
                else:
                    reasoning = ""  # All thinking, no visible reasoning

            # Determine domain from metadata
            metadata = last_call.get("metadata", {})
            specialist = metadata.get("specialist", "python")
            domain_map = {
                "python": "python",
                "python-lora": "python",
                "math": "math",
                "math-lora": "math",
                "algorithms": "algorithms",
                "algorithms-lora": "algorithms",
                "data_structures": "data_structures",
            }
            domain = domain_map.get(specialist, "python")

            examples.append({
                "problem": problem[:4000] if problem else "",
                "reasoning": reasoning[:2000] if reasoning else "",
                "solution": code[:4000] if code else "",
                "domain": domain,
                "problem_id": problem_id,
                "source_file": str(jsonl_file.name),
            })

        except Exception as e:
            print(f"  Error processing {jsonl_file.name}: {e}")
            continue

    print(f"Found {len(examples)} successful examples with code")

    if len(examples) == 0:
        print("No training data available yet.")
        print("\nTo generate training data:")
        print("1. Ensure Modal inference is working")
        print("2. Run e2e tests: python -m e2e_test.run_e2e --all --log-dir ./runs")
        print("3. Re-run this script")
        return {"train": 0, "eval": 0, "total": 0}

    # Shuffle and split
    random.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_ratio))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    # Group by domain
    domains: dict[str, list[dict]] = {}
    for ex in examples:
        domain = ex["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(ex)

    # Write combined files
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(output_dir / "eval.jsonl", "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    # Write domain-specific files
    for domain, domain_examples in domains.items():
        n_eval_domain = max(1, int(len(domain_examples) * eval_ratio))

        with open(output_dir / f"{domain}_train.jsonl", "w") as f:
            for ex in domain_examples[n_eval_domain:]:
                f.write(json.dumps(ex) + "\n")

        with open(output_dir / f"{domain}_eval.jsonl", "w") as f:
            for ex in domain_examples[:n_eval_domain]:
                f.write(json.dumps(ex) + "\n")

        print(f"  {domain}: {len(domain_examples)} examples")

    result = {
        "train": len(train_examples),
        "eval": len(eval_examples),
        "total": len(examples),
        "domains": {k: len(v) for k, v in domains.items()},
    }

    print(f"\nExported to {output_dir}:")
    print(f"  train.jsonl: {result['train']} examples")
    print(f"  eval.jsonl: {result['eval']} examples")

    return result


def get_problem_statement(db: CallDB, problem_id: str) -> str | None:
    """Look up original problem statement from trajectory logs."""
    with db._connect() as conn:
        row = conn.execute(
            """
            SELECT input_preview FROM calls
            WHERE problem_id = ? AND tier = 'fleet'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (problem_id,),
        ).fetchone()
        if row:
            return row["input_preview"]
    return None


def format_for_training(row: dict, problem_statement: str | None = None) -> dict:
    """Transform call_db row to training.py format.

    training.py expects:
        - problem: The problem statement
        - reasoning: Step-by-step reasoning (extracted from output)
        - solution: The code solution
        - domain: Specialist domain (python/math/algorithms)
    """
    # Extract reasoning from output (before code block)
    output = row.get("output", "") or ""
    code = row.get("code", "") or ""

    # Try to separate reasoning from code in output
    reasoning = output
    if "```python" in output:
        reasoning = output.split("```python")[0].strip()
    elif "```" in output:
        reasoning = output.split("```")[0].strip()

    # Clean up reasoning (remove <think> tags if present)
    if "<think>" in reasoning:
        import re
        reasoning = re.sub(r"<think>.*?</think>", "", reasoning, flags=re.DOTALL).strip()

    # Use problem statement from input or look it up
    problem = problem_statement or row.get("input", "")
    if isinstance(problem, str) and problem.startswith("[{"):
        # Parse JSON messages format
        try:
            messages = json.loads(problem)
            for msg in messages:
                if msg.get("role") == "user":
                    problem = msg.get("content", problem)
                    break
        except json.JSONDecodeError:
            pass

    # Map specialist to domain
    specialist = row.get("specialist", "python")
    domain_map = {
        "python": "python",
        "python-lora": "python",
        "math": "math",
        "math-lora": "math",
        "algorithms": "algorithms",
        "algorithms-lora": "algorithms",
        "data_structures": "data_structures",
    }
    domain = domain_map.get(specialist, "python")

    return {
        "problem": problem[:4000] if problem else "",  # Truncate for safety
        "reasoning": reasoning[:2000] if reasoning else "",
        "solution": code[:4000] if code else "",
        "domain": domain,
        "problem_id": row.get("problem_id", ""),
        "tests_passed": row.get("tests_passed", 0),
        "tests_total": row.get("tests_total", 0),
    }


def export_training_data(
    db_path: Path,
    output_dir: Path,
    min_tests_passed: int = 1,
    require_all_tests: bool = True,
    eval_ratio: float = 0.1,
) -> dict[str, int]:
    """Export and format training data from call_db.

    Args:
        db_path: Path to hi_moe.db
        output_dir: Output directory for JSONL files
        min_tests_passed: Minimum tests passed to include
        require_all_tests: Only include if all tests passed
        eval_ratio: Fraction to hold out for evaluation

    Returns:
        Stats on exported data
    """
    db = CallDB(db_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get training stats first
    stats = db.get_training_stats()
    print(f"Database stats: {stats}")

    # Query successful examples
    with db._connect() as conn:
        if require_all_tests:
            rows = conn.execute(
                """
                SELECT * FROM training_sft
                WHERE tests_passed = tests_total AND tests_total > 0
                """
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM training_sft
                WHERE tests_passed >= ?
                """,
                (min_tests_passed,),
            ).fetchall()

    print(f"Found {len(rows)} examples matching criteria")

    if len(rows) == 0:
        print("No training data available yet.")
        print("\nTo generate training data:")
        print("1. Fix Modal inference (hi_moe-7l1)")
        print("2. Run e2e tests: python -m e2e_test.run_e2e --all")
        print("3. Re-run this script")
        return {"train": 0, "eval": 0, "total": 0}

    # Format examples
    examples = []
    for row in rows:
        formatted = format_for_training(dict(row))
        if formatted["solution"]:  # Only include if we have actual code
            examples.append(formatted)

    print(f"Formatted {len(examples)} examples with code")

    # Shuffle and split
    random.shuffle(examples)
    n_eval = max(1, int(len(examples) * eval_ratio))
    eval_examples = examples[:n_eval]
    train_examples = examples[n_eval:]

    # Group by domain for domain-specific adapters
    domains: dict[str, list[dict]] = {}
    for ex in examples:
        domain = ex["domain"]
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(ex)

    # Write combined files
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(output_dir / "eval.jsonl", "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    # Write domain-specific files
    for domain, domain_examples in domains.items():
        n_eval_domain = max(1, int(len(domain_examples) * eval_ratio))

        with open(output_dir / f"{domain}_train.jsonl", "w") as f:
            for ex in domain_examples[n_eval_domain:]:
                f.write(json.dumps(ex) + "\n")

        with open(output_dir / f"{domain}_eval.jsonl", "w") as f:
            for ex in domain_examples[:n_eval_domain]:
                f.write(json.dumps(ex) + "\n")

        print(f"  {domain}: {len(domain_examples)} examples")

    result = {
        "train": len(train_examples),
        "eval": len(eval_examples),
        "total": len(examples),
        "domains": {k: len(v) for k, v in domains.items()},
    }

    print(f"\nExported to {output_dir}:")
    print(f"  train.jsonl: {result['train']} examples")
    print(f"  eval.jsonl: {result['eval']} examples")

    return result


def upload_to_modal(data_dir: Path) -> None:
    """Upload training data to Modal volume."""
    import subprocess

    print("\nUploading to Modal data volume...")

    # Use modal volume put command
    for jsonl_file in data_dir.glob("*.jsonl"):
        cmd = f"modal volume put hi-moe-data {jsonl_file} /{jsonl_file.name}"
        print(f"  {cmd}")
        subprocess.run(cmd.split(), check=True)

    print("Upload complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Export training data from call_db or JSONL files for LoRA fine-tuning"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Directory containing run-*.jsonl trajectory files (default: runs/)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Path to hi_moe.db (deprecated, use --runs-dir instead)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training"),
        help="Output directory (default: training/)",
    )
    parser.add_argument(
        "--min-tests",
        type=int,
        default=1,
        help="Minimum tests passed to include (default: 1)",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        default=True,
        help="Only include if all tests passed (default: True)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to Modal data volume after export",
    )
    args = parser.parse_args()

    # Prefer JSONL export (contains full output data)
    if args.runs_dir.exists():
        print(f"Exporting from JSONL files in {args.runs_dir}")
        result = export_from_jsonl(
            runs_dir=args.runs_dir,
            output_dir=args.output,
        )
    elif args.db and args.db.exists():
        print(f"Exporting from database {args.db}")
        result = export_training_data(
            db_path=args.db,
            output_dir=args.output,
            min_tests_passed=args.min_tests,
            require_all_tests=args.require_all,
        )
    else:
        print(f"No data source found.")
        print(f"  Runs directory: {args.runs_dir} (not found)")
        if args.db:
            print(f"  Database: {args.db} (not found)")
        print("\nRun e2e tests first to generate training data:")
        print("  python -m e2e_test.run_e2e --all --log-dir ./runs")
        return 1

    if result["total"] > 0 and args.upload:
        upload_to_modal(args.output)

    return 0


if __name__ == "__main__":
    exit(main())
