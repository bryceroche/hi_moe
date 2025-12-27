"""Self-reflection and architecture visibility (hi_moe-3ml).

Enables the system to model itself - introspect capabilities,
bottlenecks, and current state. Provides a unified view of
the system's architecture for both humans and the system itself.

Key capabilities:
- Architecture introspection (tiers, specialists, tools)
- Bottleneck detection (slow/failing components)
- State snapshot (current configuration and health)
- Capability map (what the system can/cannot do)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TierInfo:
    """Information about a single tier."""
    name: str
    level: int  # 1=Fleet, 2=Dispatcher, 3=Architect, 4=Monitor
    description: str
    capabilities: list[str]
    dependencies: list[str]
    health_status: str = "unknown"  # healthy, degraded, failed, unknown
    metrics: dict = field(default_factory=dict)


@dataclass
class SpecialistInfo:
    """Information about a specialist."""
    name: str
    domains: list[str]
    adapter: str | None = None
    success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    invocations: int = 0


@dataclass
class BottleneckInfo:
    """Detected performance bottleneck."""
    component: str
    bottleneck_type: str  # latency, failure_rate, capacity
    severity: float  # 0-1
    description: str
    recommendation: str


@dataclass
class CapabilityGap:
    """Detected capability gap."""
    domain: str
    description: str
    evidence: list[str]
    priority: str  # high, medium, low


@dataclass
class SystemState:
    """Snapshot of system state."""
    timestamp: str
    tiers: list[TierInfo]
    specialists: list[SpecialistInfo]
    bottlenecks: list[BottleneckInfo]
    capability_gaps: list[CapabilityGap]
    health_summary: str
    metrics: dict = field(default_factory=dict)


class SelfReflection:
    """System self-reflection and introspection (hi_moe-3ml).

    Provides the system with visibility into its own architecture,
    capabilities, and current state.

    Usage:
        reflection = SelfReflection()

        # Register components
        reflection.register_tier(...)
        reflection.register_specialist(...)

        # Update metrics
        reflection.update_tier_metrics("dispatcher", {...})

        # Get introspection
        state = reflection.get_state()
        bottlenecks = reflection.detect_bottlenecks()
        gaps = reflection.detect_capability_gaps()

        # Generate self-description
        description = reflection.describe_self()
    """

    def __init__(self):
        self.tiers: dict[str, TierInfo] = {}
        self.specialists: dict[str, SpecialistInfo] = {}
        self.metric_history: list[dict] = []
        self._initialize_default_architecture()

    def _initialize_default_architecture(self) -> None:
        """Initialize with default hi_moe architecture."""
        # Register tiers
        self.register_tier(TierInfo(
            name="monitor",
            level=4,
            description="Progress Monitor - Meta-evaluation and retry orchestration",
            capabilities=[
                "task_tracking",
                "retry_orchestration",
                "progress_monitoring",
                "surprise_detection",
            ],
            dependencies=["architect"],
        ))

        self.register_tier(TierInfo(
            name="architect",
            level=3,
            description="Abstract Architect - Strategic planning and task decomposition",
            capabilities=[
                "strategic_planning",
                "task_decomposition",
                "plan_revision",
                "failure_memory",
            ],
            dependencies=["dispatcher"],
        ))

        self.register_tier(TierInfo(
            name="dispatcher",
            level=2,
            description="Routing Dispatcher - Specialist selection and task routing",
            capabilities=[
                "specialist_routing",
                "structured_planning",
                "heuristic_routing",
                "learned_routing",
                "context_tracking",
            ],
            dependencies=["fleet"],
        ))

        self.register_tier(TierInfo(
            name="fleet",
            level=1,
            description="Specialized Fleet - Domain execution with LoRA adapters",
            capabilities=[
                "code_generation",
                "code_validation",
                "self_healing",
                "adapter_selection",
            ],
            dependencies=[],
        ))

        # Register default specialists
        default_specialists = [
            ("python", ["python", "debugging", "testing"]),
            ("math", ["mathematics", "proofs", "analysis"]),
            ("algorithms", ["algorithm_design", "complexity", "optimization"]),
            ("debugging", ["bug_finding", "error_tracing"]),
            ("refactoring", ["code_improvement", "pattern_application"]),
            ("general", ["fallback", "general_tasks"]),
        ]

        for name, domains in default_specialists:
            self.register_specialist(SpecialistInfo(
                name=name,
                domains=domains,
            ))

    def register_tier(self, tier: TierInfo) -> None:
        """Register a tier in the architecture."""
        self.tiers[tier.name] = tier
        logger.debug(f"[SelfReflection] Registered tier: {tier.name}")

    def register_specialist(self, specialist: SpecialistInfo) -> None:
        """Register a specialist."""
        self.specialists[specialist.name] = specialist
        logger.debug(f"[SelfReflection] Registered specialist: {specialist.name}")

    def update_tier_metrics(self, tier_name: str, metrics: dict) -> None:
        """Update metrics for a tier."""
        if tier_name in self.tiers:
            self.tiers[tier_name].metrics.update(metrics)
            self._update_tier_health(tier_name)

    def update_specialist_metrics(
        self,
        name: str,
        success_rate: float,
        avg_latency_ms: float,
        invocations: int,
    ) -> None:
        """Update metrics for a specialist."""
        if name in self.specialists:
            spec = self.specialists[name]
            spec.success_rate = success_rate
            spec.avg_latency_ms = avg_latency_ms
            spec.invocations = invocations

    def _update_tier_health(self, tier_name: str) -> None:
        """Update health status based on metrics."""
        tier = self.tiers.get(tier_name)
        if not tier:
            return

        metrics = tier.metrics
        success_rate = metrics.get("success_rate", 1.0)
        avg_latency = metrics.get("avg_latency_ms", 0)

        if success_rate < 0.5:
            tier.health_status = "failed"
        elif success_rate < 0.8 or avg_latency > 10000:
            tier.health_status = "degraded"
        else:
            tier.health_status = "healthy"

    def detect_bottlenecks(self) -> list[BottleneckInfo]:
        """Detect performance bottlenecks in the system."""
        bottlenecks = []

        # Check tier metrics
        for name, tier in self.tiers.items():
            metrics = tier.metrics

            # Latency bottleneck
            avg_latency = metrics.get("avg_latency_ms", 0)
            if avg_latency > 5000:
                bottlenecks.append(BottleneckInfo(
                    component=name,
                    bottleneck_type="latency",
                    severity=min(avg_latency / 10000, 1.0),
                    description=f"{name} tier has high latency: {avg_latency:.0f}ms avg",
                    recommendation="Consider caching, parallelization, or model optimization",
                ))

            # Failure rate bottleneck
            success_rate = metrics.get("success_rate", 1.0)
            if success_rate < 0.8:
                bottlenecks.append(BottleneckInfo(
                    component=name,
                    bottleneck_type="failure_rate",
                    severity=1 - success_rate,
                    description=f"{name} tier has low success rate: {success_rate:.1%}",
                    recommendation="Review prompts, add retries, or improve routing",
                ))

        # Check specialist metrics
        for name, spec in self.specialists.items():
            if spec.invocations > 0 and spec.success_rate < 0.7:
                bottlenecks.append(BottleneckInfo(
                    component=f"specialist:{name}",
                    bottleneck_type="failure_rate",
                    severity=1 - spec.success_rate,
                    description=f"{name} specialist has low success: {spec.success_rate:.1%}",
                    recommendation="Consider training LoRA adapter or improving prompts",
                ))

        return sorted(bottlenecks, key=lambda b: b.severity, reverse=True)

    def detect_capability_gaps(self) -> list[CapabilityGap]:
        """Detect missing capabilities in the system."""
        gaps = []

        # Check for missing specialist domains
        all_domains = set()
        for spec in self.specialists.values():
            all_domains.update(spec.domains)

        # Known domains that might be missing
        common_domains = {
            "sql": "SQL and database operations",
            "cuda": "GPU/CUDA programming",
            "web": "Web development (HTML/CSS/JS)",
            "devops": "DevOps and deployment",
            "testing": "Test writing and validation",
        }

        for domain, desc in common_domains.items():
            if domain not in all_domains:
                gaps.append(CapabilityGap(
                    domain=domain,
                    description=f"No specialist covers {desc}",
                    evidence=[f"Domain '{domain}' not in specialist list"],
                    priority="medium",
                ))

        # Check for underperforming specialists
        for name, spec in self.specialists.items():
            if spec.invocations >= 5 and spec.success_rate < 0.6:
                gaps.append(CapabilityGap(
                    domain=spec.domains[0] if spec.domains else name,
                    description=f"{name} specialist underperforming",
                    evidence=[f"Success rate: {spec.success_rate:.1%} over {spec.invocations} calls"],
                    priority="high",
                ))

        return gaps

    def get_state(self) -> SystemState:
        """Get current system state snapshot."""
        bottlenecks = self.detect_bottlenecks()
        gaps = self.detect_capability_gaps()

        # Determine overall health
        tier_health = [t.health_status for t in self.tiers.values()]
        if "failed" in tier_health:
            health_summary = "degraded - component failures"
        elif "degraded" in tier_health:
            health_summary = "degraded - performance issues"
        elif bottlenecks:
            health_summary = "healthy with bottlenecks"
        else:
            health_summary = "healthy"

        return SystemState(
            timestamp=datetime.now().isoformat(),
            tiers=list(self.tiers.values()),
            specialists=list(self.specialists.values()),
            bottlenecks=bottlenecks,
            capability_gaps=gaps,
            health_summary=health_summary,
            metrics={
                "total_specialists": len(self.specialists),
                "active_specialists": sum(1 for s in self.specialists.values() if s.invocations > 0),
                "bottleneck_count": len(bottlenecks),
                "capability_gap_count": len(gaps),
            },
        )

    def describe_self(self) -> str:
        """Generate natural language self-description."""
        state = self.get_state()

        lines = [
            "# System Self-Description",
            "",
            "## Architecture Overview",
            "",
            "I am a hierarchical mixture-of-experts (MoE) system with 4 tiers:",
            "",
        ]

        # Describe tiers
        for tier in sorted(state.tiers, key=lambda t: t.level, reverse=True):
            health_icon = {"healthy": "✓", "degraded": "⚠", "failed": "✗"}.get(tier.health_status, "?")
            lines.append(f"### Tier {tier.level}: {tier.name.title()} [{health_icon}]")
            lines.append(f"{tier.description}")
            lines.append(f"- Capabilities: {', '.join(tier.capabilities)}")
            lines.append("")

        # Describe specialists
        lines.append("## Specialists")
        lines.append("")
        for spec in state.specialists:
            status = ""
            if spec.invocations > 0:
                status = f" ({spec.success_rate:.0%} success, {spec.invocations} calls)"
            lines.append(f"- **{spec.name}**: {', '.join(spec.domains)}{status}")
        lines.append("")

        # Health summary
        lines.append(f"## Current Health: {state.health_summary}")
        lines.append("")

        # Bottlenecks
        if state.bottlenecks:
            lines.append("### Bottlenecks Detected")
            for b in state.bottlenecks[:3]:
                lines.append(f"- [{b.severity:.0%}] {b.description}")
            lines.append("")

        # Capability gaps
        if state.capability_gaps:
            lines.append("### Capability Gaps")
            for g in state.capability_gaps[:3]:
                lines.append(f"- [{g.priority}] {g.description}")
            lines.append("")

        return "\n".join(lines)

    def get_capability_summary(self) -> dict[str, list[str]]:
        """Get summary of what the system can and cannot do."""
        can_do = []
        cannot_do = []

        # Tier capabilities
        for tier in self.tiers.values():
            can_do.extend(tier.capabilities)

        # Specialist domains
        for spec in self.specialists.values():
            if spec.invocations == 0 or spec.success_rate >= 0.5:
                can_do.extend(spec.domains)
            else:
                cannot_do.extend([f"{d} (unreliable)" for d in spec.domains])

        # Known gaps
        for gap in self.detect_capability_gaps():
            cannot_do.append(gap.domain)

        return {
            "can_do": list(set(can_do)),
            "cannot_do": list(set(cannot_do)),
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        state = self.get_state()
        return {
            "timestamp": state.timestamp,
            "health": state.health_summary,
            "tiers": [
                {
                    "name": t.name,
                    "level": t.level,
                    "health": t.health_status,
                    "capabilities": t.capabilities,
                }
                for t in state.tiers
            ],
            "specialists": [
                {
                    "name": s.name,
                    "domains": s.domains,
                    "success_rate": s.success_rate,
                    "invocations": s.invocations,
                }
                for s in state.specialists
            ],
            "bottlenecks": [b.__dict__ for b in state.bottlenecks],
            "gaps": [g.__dict__ for g in state.capability_gaps],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    reflection = SelfReflection()

    # Simulate some metrics
    reflection.update_tier_metrics("fleet", {
        "success_rate": 0.85,
        "avg_latency_ms": 3000,
    })
    reflection.update_tier_metrics("dispatcher", {
        "success_rate": 0.90,
        "avg_latency_ms": 500,
    })
    reflection.update_tier_metrics("architect", {
        "success_rate": 0.75,
        "avg_latency_ms": 8000,
    })

    reflection.update_specialist_metrics("python", 0.90, 2500, 100)
    reflection.update_specialist_metrics("math", 0.70, 4000, 50)
    reflection.update_specialist_metrics("debugging", 0.50, 3000, 20)

    # Get self-description
    print(reflection.describe_self())

    # Get capability summary
    print("\n--- Capabilities ---")
    caps = reflection.get_capability_summary()
    print(f"Can do: {caps['can_do'][:10]}...")
    print(f"Cannot do: {caps['cannot_do']}")
