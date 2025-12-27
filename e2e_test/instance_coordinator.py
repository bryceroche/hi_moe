#!/usr/bin/env python3
"""Instance coordination for multi-Claude workflows (hi_moe-152).

Prevents conflicts when multiple Claude instances run Modal jobs concurrently.

Key features:
- Unique instance IDs based on terminal/process
- Run ID tagging for trajectory logs
- Lock files for exclusive operations
- Shared endpoint awareness

Usage:
    from e2e_test.instance_coordinator import get_instance_context

    ctx = get_instance_context()
    print(f"Instance {ctx.instance_id} starting run {ctx.run_id}")
"""
from __future__ import annotations

import hashlib
import os
import socket
import time
import fcntl
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class InstanceContext:
    """Context for this instance."""
    instance_id: str  # Unique ID for this terminal/process
    run_id: str       # Unique ID for this run
    hostname: str
    pid: int
    started_at: str

    def tag(self, name: str) -> str:
        """Tag a name with instance ID to avoid collisions."""
        return f"{name}-{self.instance_id[:8]}"

    def log_path(self, base: str = "runs") -> Path:
        """Get instance-specific log path."""
        return Path(base) / f"instance-{self.instance_id[:8]}"


def get_instance_id() -> str:
    """Generate a unique instance ID based on terminal session.

    Uses TTY device + PID to create a stable ID for this terminal session.
    Different terminals get different IDs, but same terminal gets same ID.
    """
    # Try to get terminal device
    try:
        tty = os.ttyname(0)
    except OSError:
        tty = "notty"

    # Combine with parent PID (shell) for stability across commands
    ppid = os.getppid()

    # Hash for consistent short ID
    key = f"{tty}-{ppid}-{socket.gethostname()}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def get_run_id() -> str:
    """Generate a unique run ID for this execution."""
    return f"run-{int(time.time() * 1000)}-{os.getpid()}"


def get_instance_context() -> InstanceContext:
    """Get the context for this instance."""
    return InstanceContext(
        instance_id=get_instance_id(),
        run_id=get_run_id(),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        started_at=datetime.now().isoformat(),
    )


class InstanceLock:
    """File-based lock for exclusive operations across instances.

    Usage:
        with InstanceLock("deploy"):
            # Only one instance can deploy at a time
            modal_deploy()
    """

    def __init__(self, name: str, lock_dir: Path = Path("/tmp/hi_moe_locks")):
        self.name = name
        self.lock_dir = lock_dir
        self.lock_file = lock_dir / f"{name}.lock"
        self._fd = None

    def __enter__(self):
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.lock_file, "w")
        try:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write owner info
            ctx = get_instance_context()
            self._fd.write(f"{ctx.instance_id}\n{ctx.pid}\n{ctx.started_at}\n")
            self._fd.flush()
            return self
        except BlockingIOError:
            self._fd.close()
            # Read who has the lock
            try:
                with open(self.lock_file) as f:
                    owner = f.read().strip().split("\n")[0]
                raise RuntimeError(f"Lock '{self.name}' held by instance {owner}")
            except Exception:
                raise RuntimeError(f"Lock '{self.name}' held by another instance")

    def __exit__(self, *args):
        if self._fd:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            self._fd.close()
            try:
                self.lock_file.unlink()
            except FileNotFoundError:
                pass


def register_instance() -> InstanceContext:
    """Register this instance in a shared registry.

    Allows other instances to see who's running.
    """
    ctx = get_instance_context()
    registry_dir = Path("/tmp/hi_moe_instances")
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Write our registration
    reg_file = registry_dir / f"{ctx.instance_id}.json"
    import json
    with open(reg_file, "w") as f:
        json.dump({
            "instance_id": ctx.instance_id,
            "run_id": ctx.run_id,
            "hostname": ctx.hostname,
            "pid": ctx.pid,
            "started_at": ctx.started_at,
        }, f)

    return ctx


def list_active_instances() -> list[dict]:
    """List all registered instances (may include stale entries)."""
    import json
    registry_dir = Path("/tmp/hi_moe_instances")
    if not registry_dir.exists():
        return []

    instances = []
    for f in registry_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
                # Check if process is still alive
                pid = data.get("pid")
                if pid:
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        data["alive"] = True
                    except OSError:
                        data["alive"] = False
                instances.append(data)
        except Exception:
            pass

    return instances


def main():
    """Show current instance info and active instances."""
    ctx = register_instance()

    print(f"This Instance:")
    print(f"  ID: {ctx.instance_id}")
    print(f"  Run: {ctx.run_id}")
    print(f"  Host: {ctx.hostname}")
    print(f"  PID: {ctx.pid}")
    print()

    instances = list_active_instances()
    print(f"Registered Instances ({len(instances)}):")
    for inst in instances:
        alive = "✓" if inst.get("alive") else "✗"
        print(f"  {alive} {inst['instance_id'][:12]} pid={inst['pid']} {inst['started_at']}")


if __name__ == "__main__":
    main()
