"""Cloud economics tracking and buy vs rent analysis (hi_moe-ihj).

Tracks Modal/cloud costs and provides decision criteria for
buy vs rent hardware decisions.

Key metrics:
- Cost per inference hour
- Monthly burn rate
- Break-even analysis for hardware purchase
- Hidden costs of ownership
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Cloud GPU pricing (Modal, as of Dec 2024)
CLOUD_PRICING = {
    "a100_40gb": {"hourly": 3.00, "name": "A100 40GB"},
    "a100_80gb": {"hourly": 4.50, "name": "A100 80GB"},
    "h100": {"hourly": 6.00, "name": "H100 80GB"},  # Estimated
    "a10g": {"hourly": 1.10, "name": "A10G 24GB"},
    "t4": {"hourly": 0.60, "name": "T4 16GB"},
}

# Hardware purchase costs (approximate, Dec 2024)
HARDWARE_COSTS = {
    "rtx_4090": {"purchase": 2000, "tdp_watts": 450, "vram_gb": 24},
    "rtx_3090": {"purchase": 1000, "tdp_watts": 350, "vram_gb": 24},
    "a100_40gb": {"purchase": 10000, "tdp_watts": 400, "vram_gb": 40},
    "a100_80gb": {"purchase": 15000, "tdp_watts": 400, "vram_gb": 80},
    "h100": {"purchase": 30000, "tdp_watts": 700, "vram_gb": 80},
}

# Hidden costs of ownership
OWNERSHIP_COSTS = {
    "electricity_per_kwh": 0.15,  # $/kWh average
    "cooling_multiplier": 1.3,   # PUE (Power Usage Effectiveness)
    "chassis_cpu_ram": 2000,     # One-time cost for workstation
    "maintenance_yearly": 500,   # Repairs, replacements
    "depreciation_years": 3,     # GPU value drops significantly
}


@dataclass
class UsageRecord:
    """Record of cloud GPU usage."""
    timestamp: str
    gpu_type: str
    duration_hours: float
    cost: float
    task_type: str = "inference"
    task_count: int = 1


@dataclass
class CostSummary:
    """Summary of costs for a period."""
    period_start: str
    period_end: str
    total_hours: float
    total_cost: float
    by_gpu: dict[str, float]
    by_task: dict[str, float]
    avg_cost_per_hour: float
    task_count: int


@dataclass
class BuyVsRentAnalysis:
    """Analysis of buy vs rent decision."""
    monthly_cloud_cost: float
    break_even_months: float
    hardware_option: str
    hardware_cost: float
    monthly_ownership_cost: float
    recommendation: str
    reasoning: list[str]


class CloudEconomics:
    """Track cloud costs and analyze buy vs rent decisions (hi_moe-ihj).

    Usage:
        economics = CloudEconomics()

        # Track usage
        economics.record_usage("a100_80gb", duration_hours=2.5, task_type="inference")

        # Get cost summary
        summary = economics.get_monthly_summary()

        # Analyze buy vs rent
        analysis = economics.analyze_buy_vs_rent()
    """

    def __init__(self, storage_path: Path | None = None):
        self.usage_records: list[UsageRecord] = []
        self.storage_path = storage_path

        if storage_path and storage_path.exists():
            self._load()

    def record_usage(
        self,
        gpu_type: str,
        duration_hours: float,
        task_type: str = "inference",
        task_count: int = 1,
    ) -> UsageRecord:
        """Record cloud GPU usage."""
        pricing = CLOUD_PRICING.get(gpu_type, {"hourly": 3.0})
        cost = duration_hours * pricing["hourly"]

        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            gpu_type=gpu_type,
            duration_hours=duration_hours,
            cost=cost,
            task_type=task_type,
            task_count=task_count,
        )
        self.usage_records.append(record)

        logger.info(
            f"[CloudEconomics] Recorded {duration_hours:.2f}h on {gpu_type}: ${cost:.2f}"
        )
        return record

    def get_summary(
        self,
        days: int = 30,
    ) -> CostSummary:
        """Get cost summary for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()

        relevant = [r for r in self.usage_records if r.timestamp >= cutoff_str]

        by_gpu: dict[str, float] = {}
        by_task: dict[str, float] = {}
        total_hours = 0.0
        total_cost = 0.0
        task_count = 0

        for r in relevant:
            total_hours += r.duration_hours
            total_cost += r.cost
            task_count += r.task_count
            by_gpu[r.gpu_type] = by_gpu.get(r.gpu_type, 0) + r.cost
            by_task[r.task_type] = by_task.get(r.task_type, 0) + r.cost

        return CostSummary(
            period_start=cutoff_str,
            period_end=datetime.now().isoformat(),
            total_hours=total_hours,
            total_cost=total_cost,
            by_gpu=by_gpu,
            by_task=by_task,
            avg_cost_per_hour=total_cost / total_hours if total_hours > 0 else 0,
            task_count=task_count,
        )

    def analyze_buy_vs_rent(
        self,
        monthly_hours: float | None = None,
        gpu_type: str = "a100_80gb",
    ) -> BuyVsRentAnalysis:
        """Analyze whether to buy hardware vs continue renting.

        Args:
            monthly_hours: Expected monthly usage (uses historical if None)
            gpu_type: GPU type to compare against
        """
        # Get monthly usage estimate
        if monthly_hours is None:
            summary = self.get_summary(days=30)
            monthly_hours = summary.total_hours
            monthly_cloud_cost = summary.total_cost
        else:
            pricing = CLOUD_PRICING.get(gpu_type, {"hourly": 4.50})
            monthly_cloud_cost = monthly_hours * pricing["hourly"]

        # Find comparable hardware
        hardware = HARDWARE_COSTS.get(gpu_type) or HARDWARE_COSTS.get("rtx_4090")
        hardware_name = gpu_type if gpu_type in HARDWARE_COSTS else "rtx_4090"

        # Calculate ownership costs
        purchase_cost = hardware["purchase"] + OWNERSHIP_COSTS["chassis_cpu_ram"]

        # Monthly electricity cost
        # Assume 50% utilization during usage hours
        monthly_kwh = (
            hardware["tdp_watts"] / 1000  # kW
            * monthly_hours
            * 0.5  # Average utilization
            * OWNERSHIP_COSTS["cooling_multiplier"]
        )
        monthly_electricity = monthly_kwh * OWNERSHIP_COSTS["electricity_per_kwh"]

        # Monthly depreciation
        monthly_depreciation = purchase_cost / (OWNERSHIP_COSTS["depreciation_years"] * 12)

        # Monthly maintenance
        monthly_maintenance = OWNERSHIP_COSTS["maintenance_yearly"] / 12

        # Total monthly ownership cost
        monthly_ownership = monthly_electricity + monthly_depreciation + monthly_maintenance

        # Break-even calculation
        if monthly_cloud_cost > monthly_ownership:
            break_even_months = purchase_cost / (monthly_cloud_cost - monthly_ownership)
        else:
            break_even_months = float("inf")

        # Generate recommendation
        reasoning = []

        if monthly_cloud_cost < 100:
            recommendation = "RENT"
            reasoning.append(f"Low usage (${monthly_cloud_cost:.0f}/mo) - cloud is more flexible")
        elif break_even_months > 24:
            recommendation = "RENT"
            reasoning.append(f"Break-even too long ({break_even_months:.0f} months)")
            reasoning.append("Hardware depreciation risk outweighs savings")
        elif break_even_months < 12 and monthly_cloud_cost > 500:
            recommendation = "CONSIDER BUYING"
            reasoning.append(f"Break-even in {break_even_months:.0f} months")
            reasoning.append(f"Saving ~${monthly_cloud_cost - monthly_ownership:.0f}/mo after break-even")
        else:
            recommendation = "RENT"
            reasoning.append("Flexibility of cloud outweighs marginal savings")
            reasoning.append("Re-evaluate if burning $1k+/month consistently")

        # Add hidden costs warning
        reasoning.append(f"Hidden costs: electricity ${monthly_electricity:.0f}/mo, "
                        f"depreciation ${monthly_depreciation:.0f}/mo")

        return BuyVsRentAnalysis(
            monthly_cloud_cost=monthly_cloud_cost,
            break_even_months=break_even_months,
            hardware_option=hardware_name,
            hardware_cost=purchase_cost,
            monthly_ownership_cost=monthly_ownership,
            recommendation=recommendation,
            reasoning=reasoning,
        )

    def get_report(self) -> str:
        """Generate human-readable economics report."""
        summary = self.get_summary(days=30)
        analysis = self.analyze_buy_vs_rent()

        lines = [
            "=" * 60,
            "CLOUD ECONOMICS REPORT",
            "=" * 60,
            "",
            "## Last 30 Days",
            f"  Total hours: {summary.total_hours:.1f}",
            f"  Total cost: ${summary.total_cost:.2f}",
            f"  Avg cost/hour: ${summary.avg_cost_per_hour:.2f}",
            f"  Tasks completed: {summary.task_count}",
            "",
        ]

        if summary.by_gpu:
            lines.append("  By GPU:")
            for gpu, cost in sorted(summary.by_gpu.items(), key=lambda x: -x[1]):
                lines.append(f"    {gpu}: ${cost:.2f}")

        lines.extend([
            "",
            "## Buy vs Rent Analysis",
            f"  Monthly cloud cost: ${analysis.monthly_cloud_cost:.2f}",
            f"  Hardware option: {analysis.hardware_option}",
            f"  Hardware cost: ${analysis.hardware_cost:.0f}",
            f"  Monthly ownership: ${analysis.monthly_ownership_cost:.2f}",
            f"  Break-even: {analysis.break_even_months:.0f} months"
            if analysis.break_even_months < 1000 else "  Break-even: Never",
            "",
            f"  **Recommendation: {analysis.recommendation}**",
        ])

        for reason in analysis.reasoning:
            lines.append(f"    - {reason}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def project_annual_cost(self, monthly_hours: float | None = None) -> dict:
        """Project annual costs for cloud vs owned hardware."""
        if monthly_hours is None:
            summary = self.get_summary(days=30)
            monthly_hours = summary.total_hours

        results = {
            "monthly_hours": monthly_hours,
            "annual_hours": monthly_hours * 12,
            "cloud": {},
            "owned": {},
        }

        # Cloud costs
        for gpu_type, pricing in CLOUD_PRICING.items():
            annual = monthly_hours * 12 * pricing["hourly"]
            results["cloud"][gpu_type] = {
                "annual_cost": annual,
                "hourly_rate": pricing["hourly"],
            }

        # Owned costs
        for hw_type, hw_info in HARDWARE_COSTS.items():
            analysis = self.analyze_buy_vs_rent(monthly_hours, hw_type)
            year1 = hw_info["purchase"] + OWNERSHIP_COSTS["chassis_cpu_ram"] + (analysis.monthly_ownership_cost * 12)
            year2_plus = analysis.monthly_ownership_cost * 12

            results["owned"][hw_type] = {
                "year1_cost": year1,
                "year2_plus_cost": year2_plus,
                "break_even_months": analysis.break_even_months,
            }

        return results

    def save(self) -> None:
        """Save usage records to storage."""
        if not self.storage_path:
            return

        data = {
            "records": [
                {
                    "timestamp": r.timestamp,
                    "gpu_type": r.gpu_type,
                    "duration_hours": r.duration_hours,
                    "cost": r.cost,
                    "task_type": r.task_type,
                    "task_count": r.task_count,
                }
                for r in self.usage_records
            ],
            "summary": self.get_summary(days=30).__dict__,
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"[CloudEconomics] Saved to {self.storage_path}")

    def _load(self) -> None:
        """Load usage records from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            for r in data.get("records", []):
                self.usage_records.append(UsageRecord(**r))

            logger.info(f"[CloudEconomics] Loaded {len(self.usage_records)} records")
        except Exception as e:
            logger.warning(f"[CloudEconomics] Failed to load: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    economics = CloudEconomics()

    # Simulate usage patterns
    # Light dev usage: ~10 hours/month
    for _ in range(4):
        economics.record_usage("a100_80gb", 2.5, "inference", task_count=10)

    print(economics.get_report())

    # Project for heavier usage
    print("\n--- Projections for 100 hours/month ---")
    projections = economics.project_annual_cost(monthly_hours=100)
    print(f"Cloud A100-80GB: ${projections['cloud']['a100_80gb']['annual_cost']:.0f}/year")
    print(f"Owned RTX 4090 Year 1: ${projections['owned']['rtx_4090']['year1_cost']:.0f}")
    print(f"Owned RTX 4090 Year 2+: ${projections['owned']['rtx_4090']['year2_plus_cost']:.0f}/year")
