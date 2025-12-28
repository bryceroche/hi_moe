# LoRA Adapter Evaluation: HuggingFace vs Custom Training (hi_moe-z0a, hi_moe-hlv, hi_moe-ka5)

> **Decision (hi_moe-ka5):** Stop custom adapter training. Use HuggingFace adapters exclusively.
> Custom training is a massive compute sink when community adapters already achieve SOTA.

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

## Qwen 2.5 vs Qwen 3: HuggingFace Ecosystem Comparison (hi_moe-hlv)

### Adapter & Finetune Counts (Dec 2025)

| Model | Adapters | Finetunes | Merges | Quantizations |
|-------|----------|-----------|--------|---------------|
| [Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) | 5 | 100 | 67 | 77 |
| [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | **56** | **1,195** | 52 | 142 |
| [Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 42 | 107 | 32 | 113 |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) | **171** | 161 | 15 | 132 |

### Key Observations

1. **Qwen 2.5-32B-Instruct has the largest finetune ecosystem** (1,195 finetunes) - 7x more than Qwen3
2. **Qwen 3-32B has more raw adapters** (171 vs 56) but fewer production finetunes
3. **Qwen 2.5 has specialized variants** (Coder, Math) while Qwen 3 is unified with thinking mode
4. **Notable quality adapters exist for Qwen 2.5**:
   - [LIMO math adapter](https://huggingface.co/t83714/qwen2.5-32b-instruct-limo-lora-adapter): 85% Math 500 pass@1
   - Qwen2.5-Coder-32B: SOTA code, matches GPT-4o

### Qwen 3 Adapter Examples

| Adapter | Focus | Notes |
|---------|-------|-------|
| [qwen3-32b-verilog-lora](https://huggingface.co/sonyashijin/qwen3-32b-verilog-lora) | Verilog code | Hardware design |
| [DeepNews-LoRA-Qwen3-32B](https://huggingface.co/flyfishxu/DeepNews-LoRA-Qwen3-32B) | News analysis | Credibility assessment |

### Recommendation: Use Qwen 2.5 Family

**For the hi-moe hierarchy, Qwen 2.5 is the better choice:**

1. **Proven math adapter** - LIMO gives 85% Math 500 out of the box
2. **Specialized code model** - Qwen2.5-Coder-32B is SOTA, no adapter needed
3. **Larger finetune ecosystem** - More community validation and options
4. **Architecture match** - Qwen 2.5 Coder/Math share architecture with base model

**Suggested setup:**
- **Base model**: Qwen2.5-32B-Instruct (for Monitor, Architect, Dispatcher)
- **Code specialists**: Qwen2.5-Coder-32B-Instruct (no adapter needed)
- **Math specialists**: Qwen2.5-32B-Instruct + LIMO adapter
- **Debugging specialists**: Qwen2.5-Coder-32B-Instruct (no adapter needed - see below)
- **Refactoring specialists**: Qwen2.5-Coder-32B-Instruct (no adapter needed - see below)

### Debugging & Refactoring: No Adapters Needed (hi_moe-g2p)

**Finding:** No HuggingFace adapters exist for debugging/refactoring, but **Qwen2.5-Coder-32B already excels at these tasks**:

| Capability | Benchmark | Score | Notes |
|------------|-----------|-------|-------|
| Code Repair | [Aider](https://aider.chat/docs/leaderboards/) | 73.7% | Comparable to GPT-4o, 4th overall |
| Multi-lang Repair | MdEval | 75.2 | **#1 among open-source models** |
| Refactoring | Qualitative | Strong | "identify issues, suggest optimizations, refactor legacy code" |

**Recommendation:** Use Qwen2.5-Coder-32B-Instruct directly for debugging and refactoring specialists. No adapter overhead, SOTA performance built-in

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

## 7B Dispatcher Prompt (hi_moe-jhf)

For sub-100ms routing, use `e2e_test/prompts/dispatcher_7b.py`:

```python
from e2e_test.prompts import build_prompt, VLLM_GUIDED_CONFIG

# Build prompt for a problem
messages = build_prompt("Write a function to find two sum")

# Use with vLLM guided decoding
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    extra_body=VLLM_GUIDED_CONFIG,  # Enforces JSON schema
    max_tokens=30,
)
# Output: {"specialist": "python", "confidence": 0.75}
```

**Token budget:**
- With few-shot: ~360 input + 30 output
- Minimal: ~130 input + 30 output
- Target TTFT: <100ms on A100/H100

## Sources

- [LIMO LoRA Adapter](https://huggingface.co/t83714/qwen2.5-32b-instruct-limo-lora-adapter)
- [Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)
- [AdapterHub](https://adapterhub.ml/blog/2025/05/adapters-for-any-transformer/)
