# hi_moe

A harness for mixture of experts to cooperate on long-horizon tasks, solving the context drift problem through structured coordination and persistent state.

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Deploy inference endpoint (Modal)
modal deploy modal_app/vllm_server.py

# 3. Run evaluation
python -m e2e_test.run_e2e --problem 0              # Single problem
python -m e2e_test.run_e2e --all                    # All problems
python -m e2e_test.run_e2e --mock                   # Mock mode (no Modal)

# 4. Compare baseline vs hierarchy
python -m e2e_test.run_baseline_comparison
```

## Architecture

```
┌─────────────────────────────────────────┐
│         PROGRESS MONITOR                │  ← Meta-evaluation
├─────────────────────────────────────────┤
│         ABSTRACT ARCHITECT              │  ← Strategic planning
├─────────────────────────────────────────┤
│         ROUTING DISPATCHER              │  ← Task decomposition + routing
├─────────────────────────────────────────┤
│         SPECIALIZED FLEET               │  ← Domain execution (LoRA adapters)
└─────────────────────────────────────────┘
```

### Routing Modes

```bash
--routing-mode=winner_take_all   # Pick best specialist (default)
--routing-mode=probabilistic     # Sample by similarity scores
--routing-mode=ensemble          # Run 3 specialists, pick best result
```

## Key Features

- **Four-tier hierarchy** with structured handoffs between tiers
- **LoRA hot-swap** via S-LoRA + vLLM (12+ concurrent specialists)
- **Embedding-based routing** - semantic similarity for specialist selection
- **Ensemble voting** - run multiple specialists, pick best solution
- **Self-correction** - retry with test feedback on validation failure
- **Beads state management** - shared context all tiers can read

## CLI Tools

### Evaluation

| Command | Description |
|---------|-------------|
| `python -m e2e_test.run_e2e` | Run problems through tier hierarchy |
| `python -m e2e_test.run_baseline_comparison` | Compare baseline vs hierarchy |
| `python scripts/ab_test.py` | A/B test prompt variants |
| `python scripts/benchmark.py` | Run full benchmark suite |

### Analysis

| Command | Description |
|---------|-------------|
| `python -m e2e_test.error_analyzer` | Cluster and analyze failure patterns |
| `python -m e2e_test.aggregate_results` | Aggregate results from multiple runs |
| `python scripts/cost_dashboard.py` | View token usage and costs |

### Training Data

| Command | Description |
|---------|-------------|
| `python scripts/export_training_data.py` | Export routing decisions for training |
| `python scripts/prepare_gsm8k.py` | Prepare GSM8K dataset |

### Infrastructure

| Command | Description |
|---------|-------------|
| `modal deploy modal_app/vllm_server.py` | Deploy vLLM inference endpoint |
| `python -m e2e_test.modal_status` | Check Modal endpoint status |

## Documentation

- [Project Brief](project_brief.md) - Vision, architecture, and open questions
- [Handoff Protocol](docs/handoff_protocol.md) - Inter-tier communication spec

## Development

This project uses [beads](https://github.com/anthropics/beads) for issue tracking:

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status=in_progress  # Claim work
bd close <id>         # Complete work
bd stats              # Project statistics
```

## Project Structure

```
hi_moe/
├── e2e_test/           # Evaluation framework
│   ├── tiers.py        # Core tier implementations
│   ├── runner.py       # Unified runner
│   ├── validator.py    # Solution validator
│   ├── embedding_router.py  # Semantic routing
│   └── call_db.py      # Routing decision logging
├── modal_app/          # Modal deployment
│   └── vllm_server.py  # vLLM inference server
├── scripts/            # CLI tools
├── data/               # Training data, adapters
└── runs/               # Evaluation logs
```

## Status

Active development. Run `bd ready` for available work items.
