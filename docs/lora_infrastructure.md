# LoRA Infrastructure Specification

> Defines how hi_moe serves multiple specialist LoRA adapters concurrently.

## Background

### S-LoRA vs vLLM Native

**S-LoRA** (archived Dec 2024) introduced unified paging - a single memory pool managing adapter weights and KV cache to serve thousands of adapters. Its ideas influenced vLLM's native LoRA support.

**vLLM** now has built-in multi-LoRA serving with:
- Unified memory management (inspired by S-LoRA)
- LRU caching of adapters in CPU memory
- Dynamic adapter loading at runtime
- Up to 4x throughput vs naive approaches

**Decision**: Use vLLM's native LoRA support rather than the archived S-LoRA project.

## Quantization Strategy

QwQ-32B in bf16 requires ~64GB just for weights - too large for single-GPU deployment. We use quantization to fit the model with room for KV cache and adapters.

### Recommended: AWQ 4-bit

```bash
vllm serve Qwen/QwQ-32B-Preview-AWQ \
  --quantization awq \
  --enable-lora \
  --max-loras 16
```

**Memory footprint with AWQ:**
- Base model (32B params × 0.5 bytes): ~16GB
- KV cache: ~8-12GB (sequence-dependent)
- LoRA adapters: ~1GB for 16 adapters
- **Total**: ~25-30GB → fits on single 40GB A100 or 48GB A6000

### LoRA + Quantization Compatibility

| Quantization | LoRA Support | Notes |
|--------------|--------------|-------|
| AWQ | ✓ Full | Recommended for production |
| GPTQ | ✓ Full | Alternative to AWQ |
| bitsandbytes | ✓ Limited | 4-bit/8-bit, less efficient |
| FP8 | ✓ Full | Requires Hopper GPUs (H100) |

### Multi-GPU (Tensor Parallelism)

For unquantized deployment or higher throughput:

```bash
vllm serve Qwen/QwQ-32B-Preview \
  --tensor-parallel-size 2 \
  --enable-lora \
  --max-loras 16
```

With TP=2 on 2×80GB A100s: full bf16 precision with headroom for large KV cache.

## vLLM Configuration

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--enable-lora` | Enable LoRA serving | Required |
| `--max-loras` | Concurrent LoRAs in a single batch | 12-16 |
| `--max-cpu-loras` | LRU cache size in CPU memory | 32+ |
| `--max-lora-rank` | Maximum rank across all adapters | Match actual max (e.g., 64) |

### Example Server Launch

```bash
vllm serve Qwen/QwQ-32B-Preview \
  --enable-lora \
  --max-loras 16 \
  --max-cpu-loras 32 \
  --max-lora-rank 64 \
  --lora-modules python-lora=/adapters/python \
                 cuda-lora=/adapters/cuda \
                 math-lora=/adapters/math
```

### Dynamic Loading (Runtime)

Enable with environment variables:
```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_PLUGINS=lora_filesystem_resolver
export VLLM_LORA_RESOLVER_CACHE_DIR=/adapters
```

When a request specifies an unknown adapter, vLLM looks in the cache directory and loads it on-demand.

## Adapter Registry

The Dispatcher needs to know which specialists exist and their capabilities.

### Registry Schema

```python
@dataclass
class AdapterInfo:
    name: str                   # e.g., "python-lora"
    lora_int_id: int            # Unique integer ID (auto-assigned, never reused)
    path: str                   # Filesystem path to adapter weights
    rank: int                   # LoRA rank (8, 16, 32, 64)
    domains: list[str]          # What this specialist handles
    base_model: str             # Must match the running base model

    # Versioning
    version: str                # Semantic version (e.g., "1.2.0")
    created_at: datetime        # When this version was registered
    previous_version: str|None  # For rollback chain

    # Performance characteristics (updated continuously)
    avg_latency_ms: float       # Exponential moving average
    success_rate: float         # Rolling window success rate
    request_count: int          # Total requests served

    # Status
    active: bool                # Is this the active version for this specialist?
    last_used: datetime         # For usage tracking
```

### ID Management

`lora_int_id` is required by vLLM for internal tracking. Rules:
- IDs are auto-assigned starting from 1
- IDs are **never reused**, even after adapter deletion
- Store next available ID in `beads:system/adapters/next_id`
- Each adapter version gets a new ID (enables A/B testing)

### Registry Storage

Store in Beads for shared access:

```
beads:system/adapters/registry          # Full adapter registry
beads:system/adapters/loaded            # Currently loaded adapter names
beads:system/adapters/{name}/stats      # Per-adapter performance stats
```

### Example Registry Entry

```json
{
  "name": "python-lora",
  "lora_int_id": 3,
  "path": "/adapters/python-specialist/v1.2.0",
  "rank": 32,
  "domains": ["python", "debugging", "testing", "refactoring"],
  "base_model": "Qwen/QwQ-32B-Preview-AWQ",
  "version": "1.2.0",
  "created_at": "2025-01-10T08:00:00Z",
  "previous_version": "1.1.0",
  "avg_latency_ms": 2800,
  "success_rate": 0.94,
  "request_count": 15420,
  "active": true,
  "last_used": "2025-01-15T10:30:00Z"
}
```

### Versioning Strategy

**Directory structure:**
```
/adapters/
  python-specialist/
    v1.0.0/
    v1.1.0/
    v1.2.0/  ← active
  math-specialist/
    v1.0.0/  ← active
```

**Version lifecycle:**
1. **Deploy**: Register new version with `active=false`
2. **A/B Test**: Route small % of traffic to new version
3. **Promote**: Set `active=true`, deactivate old version
4. **Rollback**: Re-activate previous version if issues detected

**Rollback command:**
```python
async def rollback_adapter(name: str):
    current = registry.get_active(name)
    if current.previous_version:
        previous = registry.get_version(name, current.previous_version)
        previous.active = True
        current.active = False
        registry.save(previous, current)
```

## Specialist Fleet

### Initial Specialists

| Specialist | Domains | Rank | Notes |
|------------|---------|------|-------|
| `python-lora` | Python, debugging, testing | 32 | General Python tasks |
| `cuda-lora` | CUDA, GPU optimization | 32 | Performance-critical code |
| `math-lora` | Math, algorithms, proofs | 64 | Higher rank for reasoning |
| `systems-lora` | C, C++, low-level | 32 | Systems programming |
| `web-lora` | JavaScript, TypeScript, HTML/CSS | 32 | Frontend/backend web |
| `data-lora` | SQL, pandas, data pipelines | 32 | Data engineering |
| `devops-lora` | Docker, K8s, CI/CD, shell | 16 | Infrastructure |
| `docs-lora` | Documentation, markdown | 16 | Lower rank sufficient |

### Training Pipeline (Future)

1. Collect successful executions from Fleet
2. Filter by confidence > 0.9
3. Group by domain
4. Fine-tune LoRA adapters on domain-specific data
5. A/B test new adapters against existing
6. Promote winners to production

## Dispatcher Integration

### Specialist Selection Flow

```
Dispatcher receives task
    │
    ▼
Extract domain signals from task
    │
    ▼
Query adapter registry for matching domains
    │
    ▼
Score candidates by:
  - Domain match strength
  - Historical success rate
  - Request count (prefer battle-tested)
    │
    ▼
Select top specialist (or fallback to base model)
    │
    ▼
Issue delegation with specialist_hint
```

### Domain Extraction Strategy

Three-tier approach (fast → accurate):

**Tier 1: Keyword matching (< 1ms)**
```python
DOMAIN_KEYWORDS = {
    "python": ["python", "pip", "pytest", "django", "flask", ".py"],
    "cuda": ["cuda", "gpu", "kernel", "nvidia", "tensor"],
    "math": ["proof", "theorem", "algorithm", "complexity", "optimize"],
    # ...
}

def extract_keywords(task: str) -> list[str]:
    task_lower = task.lower()
    return [domain for domain, kws in DOMAIN_KEYWORDS.items()
            if any(kw in task_lower for kw in kws)]
```

**Tier 2: File extension inference (< 1ms)**
```python
EXTENSION_MAP = {
    ".py": "python", ".pyx": "python",
    ".cu": "cuda", ".cuh": "cuda",
    ".js": "web", ".ts": "web", ".tsx": "web",
    ".sql": "data",
    # ...
}
```

**Tier 3: Embedding similarity (10-50ms, optional)**
- Pre-compute domain embeddings from specialist training data
- Embed incoming task, find nearest domain
- Use only when Tier 1+2 are ambiguous

**Selection logic:**
```python
def select_specialist(task: DelegationPayload) -> AdapterInfo | None:
    # Fast path: keyword + extension
    domains = extract_keywords(task.objective)
    domains += infer_from_extensions(task.context_refs)

    if not domains:
        # Slow path: embedding similarity (if enabled)
        domains = embedding_match(task.objective, threshold=0.7)

    if not domains:
        return None  # Fallback to base model

    # Score and select
    candidates = registry.find_by_domains(domains)
    return max(candidates, key=lambda a: (
        a.success_rate * 0.6 +
        min(a.request_count / 1000, 1.0) * 0.3 +
        domain_match_score(a.domains, domains) * 0.1
    ))
```

### Fallback Behavior

When no specialist matches:

**1. Use base model directly**
```python
if specialist is None:
    # No adapter, use base model
    response = await client.chat.completions.create(
        model="Qwen/QwQ-32B-Preview-AWQ",  # Base model, no LoRA
        messages=[{"role": "user", "content": task_prompt}]
    )
```

**2. Log routing gap**
```python
# Track unrouted tasks for future specialist creation
await beads.append("system/routing-gaps", {
    "task_objective": task.objective,
    "extracted_domains": domains,
    "timestamp": datetime.now().isoformat()
})
```

**3. Report in outcome**
```python
outcome.resources_used = {
    "specialist": None,
    "fallback_reason": "no_domain_match",
    "extracted_domains": domains
}
```

The Progress Monitor can analyze `routing-gaps` to identify patterns for new specialist training.

### Request Format

When calling vLLM with a specific adapter:

```python
# OpenAI-compatible API
response = client.chat.completions.create(
    model="Qwen/QwQ-32B-Preview",
    messages=[{"role": "user", "content": task_prompt}],
    extra_body={
        "lora_request": {
            "lora_name": "python-lora",
            "lora_int_id": 1,  # Unique ID per adapter
            "lora_local_path": "/adapters/python"
        }
    }
)
```

Or via the `--lora-modules` name:

```python
response = client.chat.completions.create(
    model="python-lora",  # Use adapter name directly
    messages=[{"role": "user", "content": task_prompt}]
)
```

## Performance Optimization

### Warm-Up Strategy

Pre-load frequently-used adapters on startup (in parallel):

```python
async def warmup_adapters(registry: AdapterRegistry):
    """Send dummy requests to load top adapters into GPU memory."""
    top_adapters = registry.get_by_usage(limit=MAX_LORAS)

    async def warmup_one(adapter: AdapterInfo):
        try:
            await client.chat.completions.create(
                model=adapter.name,
                messages=[{"role": "user", "content": "x"}],
                max_tokens=1  # Minimal work, just trigger load
            )
            logger.info(f"Warmed up {adapter.name}")
        except Exception as e:
            logger.error(f"Failed to warm {adapter.name}: {e}")

    # Parallel warmup - all adapters at once
    await asyncio.gather(*[warmup_one(a) for a in top_adapters])
```

### Health Check

Before routing, verify vLLM is healthy:

```python
async def health_check() -> bool:
    """Check vLLM server health and adapter availability."""
    try:
        # Check server is responding
        response = await client.models.list()

        # Verify expected adapters are registered
        model_names = {m.id for m in response.data}
        expected = {a.name for a in registry.get_active_adapters()}

        missing = expected - model_names
        if missing:
            logger.warning(f"Missing adapters: {missing}")
            return False

        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

Add to startup and periodic checks:
```python
# Startup
if not await health_check():
    raise RuntimeError("vLLM health check failed")

# Periodic (every 30s)
@scheduler.interval(seconds=30)
async def periodic_health():
    healthy = await health_check()
    await beads.set("system/health/vllm", {
        "healthy": healthy,
        "checked_at": datetime.now().isoformat()
    })
```

### Batching Strategy

vLLM batches requests using the same adapter more efficiently. The Dispatcher can optimize by:

1. Grouping pending tasks by likely specialist
2. Issuing batches to minimize adapter swaps
3. Using `first_success` aggregation for parallel attempts

### Memory Budget

**With AWQ quantization (recommended for single GPU):**

For 40-48GB GPU (A100-40GB, A6000):
- Base model (QwQ-32B AWQ 4-bit): ~16GB
- KV cache (4K context): ~8GB
- LoRA adapters: ~1GB for 16 adapters
- Headroom: ~15GB

**LoRA adapter memory formula:**

```
Adapter memory ≈ 2 × rank × hidden_dim × num_target_layers × dtype_size

For QwQ-32B (hidden_dim ≈ 5120, 64 layers, targeting q/k/v/o projections):
  ≈ 2 × 32 × 5120 × 64 × 4 × 2 bytes (bf16 adapters on quantized base)
  ≈ 167MB per rank-32 adapter

With max_loras=16: 16 × 167MB ≈ 2.7GB in GPU memory
```

**Note**: LoRA weights are typically kept in bf16 even when base model is quantized, for training stability. Actual memory depends on which layers have adapters.

## Monitoring

### Key Metrics

Track in `beads:system/metrics/`:

| Metric | Description |
|--------|-------------|
| `adapter_load_latency_ms` | Time to load adapter from CPU→GPU |
| `adapter_cache_hits` | LRU cache hit rate |
| `adapter_utilization` | Requests per adapter |
| `specialist_success_rate` | Success rate by specialist |

### Alerting

- Alert if `adapter_load_latency_ms` > 500ms (cold start problem)
- Alert if any specialist `success_rate` < 0.8
- Alert if single adapter handles > 50% of traffic (imbalanced routing)

## Implementation Phases

### Phase 1: Single Adapter (MVP)

1. Deploy vLLM with base model + one LoRA adapter
2. Hardcode specialist selection in Dispatcher
3. Validate handoff protocol works end-to-end

### Phase 2: Multi-Adapter

1. Add adapter registry to Beads
2. Implement Dispatcher specialist selection logic
3. Deploy 4-6 initial specialists
4. Add warmup and basic monitoring

### Phase 3: Dynamic Loading

1. Enable runtime adapter loading
2. Implement adapter training pipeline
3. Add A/B testing for new adapters
4. Automatic specialist discovery based on task patterns

## References

- [vLLM LoRA Documentation](https://docs.vllm.ai/en/stable/features/lora/)
- [S-LoRA Paper](https://arxiv.org/abs/2311.03285)
- [Anyscale Multi-LoRA Guide](https://docs.anyscale.com/llm/serving/multi-lora)
- [NVIDIA Triton Multi-LoRA Tutorial](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/docs/llama_multi_lora_tutorial.html)
