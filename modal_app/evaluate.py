"""Evaluate hi_moe system - tier routing vs raw adapter comparison."""
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
    timeout=3600,  # 1 hour for full evaluation
)
def evaluate_adapter(
    endpoint: str,
    adapter: str,
    eval_file: str,
    max_samples: int = 20,
) -> dict:
    """Evaluate a single adapter on benchmark problems."""
    import json
    import httpx
    import time

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

            start = time.time()
            try:
                # 5 minute timeout for QwQ reasoning model
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
                    timeout=300.0,  # 5 minutes for QwQ
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
                    print(f"  [{adapter}] Problem {i}: OK ({elapsed:.1f}s)")
                else:
                    results.append({
                        "problem_idx": i,
                        "domain": expected_domain,
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "elapsed_seconds": elapsed,
                    })
                    print(f"  [{adapter}] Problem {i}: FAIL HTTP {response.status_code}")

            except Exception as e:
                elapsed = time.time() - start
                results.append({
                    "problem_idx": i,
                    "domain": expected_domain,
                    "success": False,
                    "error": str(e)[:100],
                    "elapsed_seconds": elapsed,
                })
                print(f"  [{adapter}] Problem {i}: ERROR {str(e)[:50]}")

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
    }

    return {"metrics": metrics, "results": results}


@app.function(
    image=eval_image,
    volumes={DATA_PATH: data_volume},
    timeout=7200,  # 2 hours for parallel evaluation
)
def compare_adapters_parallel(
    endpoint: str,
    adapters: list[str],
    eval_file: str,
    max_samples: int = 10,
) -> dict:
    """Compare adapters in PARALLEL - much faster than sequential."""
    print(f"Evaluating {len(adapters)} adapters in parallel on {eval_file}...")

    # Spawn all adapter evaluations concurrently
    handles = []
    for adapter in adapters:
        print(f"  Spawning evaluation for {adapter}...")
        handle = evaluate_adapter.spawn(
            endpoint=endpoint,
            adapter=adapter,
            eval_file=eval_file,
            max_samples=max_samples,
        )
        handles.append((adapter, handle))

    # Collect results as they complete
    comparisons = []
    for adapter, handle in handles:
        print(f"  Waiting for {adapter}...")
        result = handle.get()
        comparisons.append(result["metrics"])
        print(f"  {adapter}: {result['metrics']['success_rate']:.1%} success")

    return {"comparisons": comparisons}


@app.local_entrypoint()
def main(
    endpoint: str = "https://bryce-roche--hi-moe-inference-vllmserver-serve.modal.run",
    adapter: str = "base",
    eval_file: str = "python_eval.jsonl",
    max_samples: int = 3,
    compare: bool = False,
    parallel: bool = True,
):
    """Evaluate adapter performance.

    Examples:
        # Quick single adapter test (1 sample)
        modal run modal_app/evaluate.py --adapter=base --max-samples=1

        # Compare all adapters in parallel (fast!)
        modal run modal_app/evaluate.py --compare --parallel --max-samples=3

        # Compare adapters sequentially (slow, for debugging)
        modal run modal_app/evaluate.py --compare --no-parallel --max-samples=3
    """
    if compare:
        import httpx

        # Get available adapters
        try:
            resp = httpx.get(f"{endpoint}/v1/models", timeout=30)
            models = [m["id"] for m in resp.json().get("data", [])]
        except Exception:
            models = ["base"]

        print(f"Comparing adapters: {models}")
        print(f"Eval file: {eval_file}")
        print(f"Max samples: {max_samples}")
        print(f"Mode: {'parallel' if parallel else 'sequential'}")
        print()

        if parallel:
            result = compare_adapters_parallel.remote(
                endpoint=endpoint,
                adapters=models,
                eval_file=eval_file,
                max_samples=max_samples,
            )
        else:
            # Sequential fallback
            comparisons = []
            for model in models:
                print(f"Evaluating {model}...")
                r = evaluate_adapter.remote(
                    endpoint=endpoint,
                    adapter=model,
                    eval_file=eval_file,
                    max_samples=max_samples,
                )
                comparisons.append(r["metrics"])
            result = {"comparisons": comparisons}

        print("\n" + "=" * 50)
        print("COMPARISON RESULTS")
        print("=" * 50 + "\n")

        for metrics in result["comparisons"]:
            print(f"Adapter: {metrics['adapter']}")
            print(f"  Success Rate: {metrics['success_rate']:.1%}")
            print(f"  Code Gen Rate: {metrics['code_generation_rate']:.1%}")
            print(f"  Avg Response Time: {metrics['avg_response_time']:.1f}s")
            print(f"  Avg Output Length: {metrics['avg_output_length']:.0f} chars")
            print()

    else:
        # Single adapter evaluation
        print(f"Evaluating {adapter} on {eval_file} ({max_samples} samples)...")
        result = evaluate_adapter.remote(
            endpoint=endpoint,
            adapter=adapter,
            eval_file=eval_file,
            max_samples=max_samples,
        )

        metrics = result["metrics"]
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50 + "\n")
        print(f"Adapter: {metrics['adapter']}")
        print(f"Samples: {metrics['total_samples']}")
        print(f"Success Rate: {metrics['success_rate']:.1%}")
        print(f"Code Generation Rate: {metrics['code_generation_rate']:.1%}")
        print(f"Avg Response Time: {metrics['avg_response_time']:.1f}s")
        print(f"Avg Output Length: {metrics['avg_output_length']:.0f} chars")
