# hi_moe

A harness for mixture of experts to cooperate on long-horizon tasks, solving the context drift problem through structured coordination and persistent state.

## Architecture

```
┌─────────────────────────────────────────┐
│         PROGRESS MONITOR                │  ← Meta-evaluation
├─────────────────────────────────────────┤
│         ABSTRACT ARCHITECT              │  ← Strategic planning
├─────────────────────────────────────────┤
│         ROUTING DISPATCHER              │  ← Task decomposition
├─────────────────────────────────────────┤
│         SPECIALIZED FLEET               │  ← Domain execution (LoRA adapters)
└─────────────────────────────────────────┘
```

## Key Features

- **Four-tier hierarchy** with structured handoffs between tiers
- **LoRA hot-swap** via S-LoRA + vLLM (12+ concurrent specialists)
- **Beads state management** - shared context all tiers can read
- **Surprise detection** - flag unexpected outcomes for learning
- **Routine caching** - save successful patterns for reuse

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
```

## Status

Early development. See `bd list --status=open` for current work items.
