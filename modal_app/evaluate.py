"""Evaluate adapter quality by running benchmark problems."""
from __future__ import annotations

import modal

app = modal.App("hi-moe-evaluate")

data_volume = modal.Volume.from_name("hi-moe-data", create_if_missing=True)
DATA_PATH = "/data"

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("httpx", "openai")
)


@app.function(
    image=eval_image,
    volumes={DATA_PATH: data_volume},
    timeout=1800,
)
def evaluate_adapter(
    endpoint: str,
    adapter: str,
    eval_file: str,
    max_samples: int = 20,
) -> dict:
    """Evaluate an adapter on a benchmark dataset.

    Returns metrics like pass rate and average response quality.
    """
    import json
    import httpx
    import time

    # Load eval data from volume
    eval_path = f"{DATA_PATH}/{eval_file}"

    results = []
    total_time = 0

    with open(eval_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break

            example = json.loads(line)
            problem = example["problem"]
            expected_domain = example["domain"]

            # Generate response
            start = time.time()
            try:
                response = httpx.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "model": adapter,
                        "messages": [
                            {"role": "system", "content": f"You are a {expected_domain} expert."},
                            {"role": "user", "content": problem},
                        ],
                        "max_tokens": 1024,
                        "temperature": 0.2,
                    },
                    timeout=120,
                )
                elapsed = time.time() - start
                total_time += elapsed

                if response.status_code == 200:
                    data = response.json()
                    output = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})

                    results.append({
                        "problem_idx": i,
                        "domain": expected_domain,
                        "success": True,
                        "output_length": len(output),
                        "has_code": "```" in output or "def " in output,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "elapsed_seconds": elapsed,
                    })
                else:
                    results.append({
                        "problem_idx": i,
                        "domain": expected_domain,
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "elapsed_seconds": elapsed,
                    })

            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "problem_idx": i,
                    "domain": expected_domain,
                    "success": False,
                    "error": str(e),
                    "elapsed_seconds": elapsed,
                })

    # Compute metrics
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    metrics = {
        "adapter": adapter,
        "eval_file": eval_file,
        "total_samples": len(results),
        "success_rate": len(successes) / len(results) if results else 0,
        "failure_rate": len(failures) / len(results) if results else 0,
        "avg_response_time": total_time / len(results) if results else 0,
        "code_generation_rate": sum(1 for r in successes if r.get("has_code")) / len(successes) if successes else 0,
        "avg_output_length": sum(r.get("output_length", 0) for r in successes) / len(successes) if successes else 0,
        "avg_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in successes) / len(successes) if successes else 0,
        "avg_completion_tokens": sum(r.get("completion_tokens", 0) for r in successes) / len(successes) if successes else 0,
    }

    return {"metrics": metrics, "results": results}


@app.function(
    image=eval_image,
    volumes={DATA_PATH: data_volume},
    timeout=3600,
)
def compare_adapters(
    endpoint: str,
    adapters: list[str],
    eval_files: list[str],
    max_samples: int = 10,
) -> dict:
    """Compare multiple adapters across multiple eval datasets."""
    comparisons = []

    for adapter in adapters:
        for eval_file in eval_files:
            print(f"Evaluating {adapter} on {eval_file}...")
            result = evaluate_adapter.remote(
                endpoint=endpoint,
                adapter=adapter,
                eval_file=eval_file,
                max_samples=max_samples,
            )
            comparisons.append(result["metrics"])

    return {"comparisons": comparisons}


@app.local_entrypoint()
def main(
    endpoint: str = "https://bryce-roche--hi-moe-inference-vllmserver-serve.modal.run",
    adapter: str = "base",
    eval_file: str = "python_eval.jsonl",
    max_samples: int = 10,
    compare: bool = False,
):
    """Evaluate adapter performance.

    Examples:
        # Evaluate base model on Python
        modal run modal_app/evaluate.py --adapter=base --eval-file=python_eval.jsonl

        # Evaluate custom adapter
        modal run modal_app/evaluate.py --adapter=python-test --eval-file=python_eval.jsonl

        # Compare base vs adapter
        modal run modal_app/evaluate.py --compare
    """
    if compare:
        # Compare base vs all available adapters
        import httpx

        # Get available adapters
        try:
            resp = httpx.get(f"{endpoint}/v1/models", timeout=30)
            models = [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            models = ["base"]

        eval_files = ["python_eval.jsonl"]

        print(f"Comparing adapters: {models}")
        print(f"Eval files: {eval_files}")

        result = compare_adapters.remote(
            endpoint=endpoint,
            adapters=models,
            eval_files=eval_files,
            max_samples=max_samples,
        )

        print("\n=== Comparison Results ===\n")
        for metrics in result["comparisons"]:
            print(f"Adapter: {metrics['adapter']}")
            print(f"  Eval: {metrics['eval_file']}")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            print(f"  Code Gen Rate: {metrics['code_generation_rate']:.1%}")
            print(f"  Avg Response Time: {metrics['avg_response_time']:.2f}s")
            print(f"  Avg Output Length: {metrics['avg_output_length']:.0f} chars")
            print()

    else:
        # Single adapter evaluation
        print(f"Evaluating {adapter} on {eval_file}...")
        result = evaluate_adapter.remote(
            endpoint=endpoint,
            adapter=adapter,
            eval_file=eval_file,
            max_samples=max_samples,
        )

        metrics = result["metrics"]
        print("\n=== Evaluation Results ===\n")
        print(f"Adapter: {metrics['adapter']}")
        print(f"Eval File: {metrics['eval_file']}")
        print(f"Samples: {metrics['total_samples']}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        print(f"Code Generation Rate: {metrics['code_generation_rate']:.1%}")
        print(f"Avg Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"Avg Output Length: {metrics['avg_output_length']:.0f} chars")
        print(f"Avg Prompt Tokens: {metrics['avg_prompt_tokens']:.0f}")
        print(f"Avg Completion Tokens: {metrics['avg_completion_tokens']:.0f}")
