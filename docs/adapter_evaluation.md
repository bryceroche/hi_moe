# LoRA Adapter Evaluation: HuggingFace vs Custom Training (hi_moe-z0a)

## Current Setup

Our Fleet tier uses LoRA adapters to specialize the base Qwen3-32B model for different tasks:

```python
SPECIALIST_TO_ADAPTER = {
    "python": ["python", "code", "programming"],
    "math": ["math", "reasoning", "gsm"],
    "algorithms": ["algorithm", "algo", "contest", "competitive"],
    "data_structures": ["data", "struct", "ds"],
    "debugging": ["debug", "fix", "bug"],
    "refactoring": ["refactor", "clean", "improve"],
}
```

Adapters are served via vLLM with dynamic loading. The system queries available adapters and matches specialists to adapter names.

## Available HuggingFace Options

### Math/Reasoning Adapters

| Adapter | Base Model | Focus | Performance |
|---------|------------|-------|-------------|
| [t83714/qwen2.5-32b-instruct-limo-lora-adapter](https://huggingface.co/t83714/qwen2.5-32b-instruct-limo-lora-adapter) | Qwen2.5-32B | Math reasoning | 85% Math 500 pass@1 |

- Trained on LIMO dataset using LoRA (rank 8)
- Target layers: v_proj, o_proj, q_proj, k_proj
- Based on "LIMO: Less is More for Reasoning" paper

### Code-Focused Models

| Model | Type | Notes |
|-------|------|-------|
| [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | Full model | SOTA open-source code LLM, matches GPT-4o |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | Full model | Built-in thinking mode for reasoning |

Note: These are full models, not LoRA adapters. Could use as base instead of training adapters.

### Multi-Task Framework

[AdapterHub](https://adapterhub.ml/blog/2025/05/adapters-for-any-transformer/) provides:
- Pre-built Qwen 2/3 support
- MTL-LoRA for multi-task learning (Adapters v1.2.0)
- Easier adapter composition

## Comparison Matrix

| Factor | HuggingFace Adapters | Custom Training |
|--------|---------------------|-----------------|
| **Quality** | Proven benchmarks (e.g., 85% Math 500) | Unknown until validated |
| **Domain Fit** | General purpose | Tailored to our problems |
| **Setup Time** | Minutes (download) | Days (data + training) |
| **Customization** | Limited | Full control |
| **Maintenance** | Community updates | Self-maintained |
| **Cost** | Free | GPU compute for training |
| **Composability** | May conflict | Designed together |

## Recommendations

### Short-term (Immediate Wins)

1. **Try LIMO adapter for math**: Download `t83714/qwen2.5-32b-instruct-limo-lora-adapter` and test on our math problems. 85% Math 500 is a strong baseline.

2. **Consider Qwen2.5-Coder-32B as base**: Instead of Qwen3-32B + code adapter, use Qwen2.5-Coder-32B directly for code specialists. It's SOTA for code.

### Medium-term (Hybrid Approach)

1. **Use HF adapters for well-defined domains**: Math reasoning (LIMO), general coding
2. **Train custom adapters for niche needs**: Algorithm contests, specific debugging patterns
3. **A/B test**: Compare HF vs custom on same problem set

### Long-term (Full Custom)

If we have enough training data and the hierarchy proves valuable:
1. Train specialized adapters on our validated solutions
2. Use MTL-LoRA for multi-specialist composition
3. Fine-tune on failure patterns identified by error_analyzer

## Next Steps

1. [ ] Download and test LIMO adapter on math problems
2. [ ] Benchmark Qwen2.5-Coder-32B vs Qwen3-32B for code
3. [ ] Set up A/B test framework (already have hi_moe-c5m)
4. [ ] Evaluate adapter loading latency with vLLM

## Sources

- [LIMO LoRA Adapter](https://huggingface.co/t83714/qwen2.5-32b-instruct-limo-lora-adapter)
- [Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
- [AdapterHub](https://adapterhub.ml/blog/2025/05/adapters-for-any-transformer/)
