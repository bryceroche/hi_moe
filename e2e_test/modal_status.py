#!/usr/bin/env python3
"""Check Modal instance status for hi-moe-inference (hi_moe-qho).

Usage:
    python -m e2e_test.modal_status
    python -m e2e_test.modal_status --watch  # Continuous monitoring
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime


def get_modal_apps() -> list[dict]:
    """Get list of Modal apps using CLI (parses rich table output)."""
    try:
        result = subprocess.run(
            ["modal", "app", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"Modal CLI error: {result.stderr}")
            return []

        # Parse rich table output
        apps = []
        for line in result.stdout.split("\n"):
            # Look for lines with app IDs (start with │ ap-)
            if "│" in line and "ap-" in line:
                parts = [p.strip() for p in line.split("│") if p.strip()]
                if len(parts) >= 3:
                    apps.append({
                        "id": parts[0].replace("…", ""),
                        "name": parts[1].replace("…", ""),
                        "state": parts[2],
                        "tasks": parts[3] if len(parts) > 3 else "0",
                    })
        return apps
    except FileNotFoundError:
        print("Modal CLI not found. Install with: pip install modal")
        return []
    except subprocess.TimeoutExpired:
        print("Modal CLI timed out")
        return []


def get_running_containers() -> list[dict]:
    """Get running containers via Modal CLI (parses rich table output)."""
    try:
        result = subprocess.run(
            ["modal", "container", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return []

        # Parse rich table output
        containers = []
        for line in result.stdout.split("\n"):
            # Look for lines with task IDs (ta-)
            if "│" in line and "ta-" in line:
                parts = [p.strip() for p in line.split("│") if p.strip()]
                if len(parts) >= 3:
                    containers.append({
                        "id": parts[0].replace("…", ""),
                        "app_id": parts[1].replace("…", "") if len(parts) > 1 else "",
                        "app_name": parts[2].replace("…", "") if len(parts) > 2 else "",
                        "start_time": parts[3] if len(parts) > 3 else "",
                    })
        return containers
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def check_endpoint_health(endpoint_url: str) -> dict:
    """Check if vLLM endpoint is healthy."""
    import urllib.request
    import urllib.error

    if not endpoint_url:
        return {"healthy": False, "error": "No endpoint URL"}

    # Append /health if not present
    health_url = endpoint_url.rstrip("/")
    if not health_url.endswith("/health"):
        health_url += "/health"

    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"healthy": resp.status == 200, "status": resp.status}
    except urllib.error.HTTPError as e:
        return {"healthy": False, "status": e.code, "error": str(e)}
    except urllib.error.URLError as e:
        return {"healthy": False, "error": str(e.reason)}
    except Exception as e:
        return {"healthy": False, "error": str(e)}


def get_endpoint_url() -> str | None:
    """Get the deployed vLLM endpoint URL."""
    try:
        result = subprocess.run(
            ["modal", "app", "list"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse for hi-moe-inference URL
        for line in result.stdout.split("\n"):
            if "hi-moe-inference" in line:
                # Extract URL if present
                parts = line.split()
                for part in parts:
                    if part.startswith("https://"):
                        return part
        return None
    except Exception:
        return None


def format_status(apps: list, containers: list, health: dict, endpoint: str | None) -> str:
    """Format status as human-readable output."""
    lines = [
        "=" * 60,
        f"MODAL STATUS - {datetime.now().strftime('%H:%M:%S')}",
        "=" * 60,
        "",
    ]

    # Filter for hi-moe apps
    hi_moe_apps = [a for a in apps if "hi-moe" in a.get("name", "").lower() or "hi_moe" in a.get("name", "").lower()]

    lines.append("## Apps")
    if hi_moe_apps:
        for app in hi_moe_apps:
            name = app.get("name", "unknown")
            state = app.get("state", "unknown")
            tasks = app.get("tasks", "0")
            state_icon = "●" if state == "deployed" else "○" if state == "ephemeral" else "◌"
            lines.append(f"  {state_icon} {name:25s} {state:12s} tasks={tasks}")
    else:
        lines.append("  (No hi-moe apps found)")

    lines.append("")

    # Containers
    hi_moe_containers = [c for c in containers if "hi-moe" in c.get("app_name", "").lower()]
    lines.append(f"## Running Containers ({len(hi_moe_containers)})")
    if hi_moe_containers:
        for c in hi_moe_containers[:10]:
            container_id = c.get("id", "unknown")[:16]
            app_name = c.get("app_name", "unknown")
            start = c.get("start_time", "")
            lines.append(f"  {container_id}  {app_name:20s} {start}")
    else:
        lines.append("  (No active containers)")

    lines.append("")

    # Endpoint health
    lines.append("## vLLM Endpoint")
    if endpoint:
        lines.append(f"  URL: {endpoint}")
        if health.get("healthy"):
            lines.append("  Status: ✓ Healthy")
        else:
            lines.append(f"  Status: ✗ {health.get('error', 'Unhealthy')}")
    else:
        lines.append("  (Endpoint URL not in app list - check modal.com dashboard)")

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Check Modal instance status")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Watch interval (seconds)")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    while True:
        apps = get_modal_apps()
        containers = get_running_containers()
        endpoint = get_endpoint_url()
        health = check_endpoint_health(endpoint) if endpoint else {}

        if args.json:
            output = {
                "timestamp": datetime.now().isoformat(),
                "apps": apps,
                "containers": containers,
                "endpoint": endpoint,
                "health": health,
            }
            print(json.dumps(output, indent=2))
        else:
            print(format_status(apps, containers, health, endpoint))

        if not args.watch:
            break

        time.sleep(args.interval)
        print("\n")


if __name__ == "__main__":
    main()
