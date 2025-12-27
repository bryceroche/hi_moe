#!/usr/bin/env python3
"""Partition work across multiple instances (hi_moe-08k).

Enables deterministic work splitting so multiple instances don't duplicate work.

Usage:
    # Get partition for this instance
    python -m e2e_test.work_partition --instance 0 --total 3

    # Use in stress test
    from e2e_test.work_partition import get_my_problems
    my_problems = get_my_problems(all_problems, instance_id=0, total_instances=3)
"""
from __future__ import annotations

import argparse
import hashlib
import os
from typing import TypeVar

T = TypeVar("T")


def get_instance_id() -> int:
    """Get instance ID from environment or default to 0."""
    return int(os.environ.get("INSTANCE_ID", "0"))


def get_total_instances() -> int:
    """Get total instances from environment or default to 1."""
    return int(os.environ.get("TOTAL_INSTANCES", "1"))


def partition_by_index(items: list[T], instance_id: int, total_instances: int) -> list[T]:
    """Partition items by simple index modulo.

    Instance 0 gets items 0, 3, 6, ...
    Instance 1 gets items 1, 4, 7, ...
    Instance 2 gets items 2, 5, 8, ...
    """
    return [item for i, item in enumerate(items) if i % total_instances == instance_id]


def partition_by_hash(items: list[dict], instance_id: int, total_instances: int,
                      key: str = "id") -> list[dict]:
    """Partition items by hash of their ID for consistent distribution.

    This ensures the same problem always goes to the same instance,
    regardless of list ordering.
    """
    result = []
    for item in items:
        item_id = item.get(key, str(item))
        hash_val = int(hashlib.md5(str(item_id).encode()).hexdigest(), 16)
        if hash_val % total_instances == instance_id:
            result.append(item)
    return result


def get_my_problems(
    all_problems: list[dict],
    instance_id: int | None = None,
    total_instances: int | None = None,
    method: str = "hash",
) -> list[dict]:
    """Get the problems assigned to this instance.

    Args:
        all_problems: Full list of problems
        instance_id: This instance's ID (0-indexed). Uses env var if None.
        total_instances: Total number of instances. Uses env var if None.
        method: "hash" for consistent hashing, "index" for round-robin

    Returns:
        List of problems assigned to this instance
    """
    if instance_id is None:
        instance_id = get_instance_id()
    if total_instances is None:
        total_instances = get_total_instances()

    if total_instances <= 1:
        return all_problems

    if method == "hash":
        return partition_by_hash(all_problems, instance_id, total_instances)
    else:
        return partition_by_index(all_problems, instance_id, total_instances)


def show_partition(problems: list[dict], total_instances: int):
    """Display how problems would be partitioned."""
    print(f"Partitioning {len(problems)} problems across {total_instances} instances:\n")

    for instance_id in range(total_instances):
        my_problems = get_my_problems(problems, instance_id, total_instances)
        problem_ids = [p.get("id", "unknown") for p in my_problems]
        print(f"Instance {instance_id}: {len(my_problems)} problems")
        for pid in problem_ids:
            print(f"  - {pid}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Partition work across instances")
    parser.add_argument(
        "--instance", "-i",
        type=int,
        default=0,
        help="This instance's ID (0-indexed)"
    )
    parser.add_argument(
        "--total", "-t",
        type=int,
        default=3,
        help="Total number of instances"
    )
    parser.add_argument(
        "--show-all", "-a",
        action="store_true",
        help="Show partition for all instances"
    )
    parser.add_argument(
        "--method", "-m",
        choices=["hash", "index"],
        default="hash",
        help="Partitioning method"
    )
    args = parser.parse_args()

    # Import problem sets
    from .stress_test import PROBLEM_SETS

    # Flatten all problems
    all_problems = []
    for difficulty, problems in PROBLEM_SETS.items():
        for p in problems:
            all_problems.append(p)

    if args.show_all:
        show_partition(all_problems, args.total)
    else:
        my_problems = get_my_problems(
            all_problems,
            args.instance,
            args.total,
            args.method,
        )
        print(f"Instance {args.instance}/{args.total}: {len(my_problems)} problems")
        for p in my_problems:
            print(f"  - {p.get('id', 'unknown')} ({p.get('difficulty', 'unknown')})")


if __name__ == "__main__":
    main()
