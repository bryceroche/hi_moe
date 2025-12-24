# BeadsClient Specification

> Python client for Beads - the shared state layer all tiers can read/write.

## Related Specifications

- **[Handoff Protocol](./handoff_protocol.md)**: Uses Beads for deduplication and caching
- **[Abstract Architect](./abstract_architect.md)**: Stores context, progress, and routine history
- **[Routing Dispatcher](./routing_dispatcher.md)**: Stores graph state and training data
- **[Specialized Fleet](./specialized_fleet.md)**: Stores artifacts and execution logs

## Required Imports

```python
from __future__ import annotations
import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
```

## Overview

Beads provides persistent shared state across all tiers of the hi_moe architecture. It solves the "forgetting" problem - ensuring the Abstract Architect knows what actually happened in lower tiers, not just what it asked for.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           BEADS                                      │
│                                                                      │
│   Persistent Key-Value Store with Namespaced Hierarchies            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   history/   │  │   system/    │  │   tasks/     │              │
│  │   routines   │  │   health     │  │   {task_id}  │              │
│  │   patterns   │  │   messages   │  │   progress   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
│  Operations: get, set, append, exists, delete, list_keys            │
│  Features: TTL expiration, atomic append, namespace listing         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌───────────┐       ┌─────────┐
   │Architect│         │Dispatcher │       │  Fleet  │
   └─────────┘         └───────────┘       └─────────┘
```

## Key Namespaces

The following namespaces are used by the system:

| Namespace | Purpose | Example Keys |
|-----------|---------|--------------|
| `history/routines/` | Cached successful strategies | `history/routines/{signature}` |
| `history/patterns/` | Learned routing patterns | `history/patterns/{problem_class}` |
| `system/health/` | Component health status | `system/health/vllm`, `system/health/docker` |
| `system/seen-messages/` | Deduplication cache | `system/seen-messages/{tier}/{msg_id}` |
| `system/routing-gaps` | Gaps in routing coverage | Single append-only list |
| `routing/training-data` | Training data for routing LoRA | Append-only list |
| `tasks/{task_id}/` | Per-task state | `tasks/{id}/objective`, `tasks/{id}/progress` |
| `execution/results/` | Large execution results | `execution/results/{request_id}` |
| `artifacts/` | Large code artifacts | `artifacts/{artifact_id}` |

## Client Interface

### Core Data Types

```python
@dataclass
class BeadsEntry:
    """A single entry in Beads."""
    key: str
    value: Any
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    version: int = 1

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class BeadsError(Exception):
    """Base exception for Beads operations."""
    pass
```

### BeadsClient Class

```python
class BeadsClient:
    """
    Client for Beads shared state store.

    Provides async operations for get/set/append with TTL support,
    namespace listing, and atomic operations.
    """

    def __init__(
        self,
        backend: BeadsBackend | None = None,
        default_ttl_seconds: int | None = None,
    ):
        """
        Initialize BeadsClient.

        Args:
            backend: Storage backend (defaults to FileBackend)
            default_ttl_seconds: Default TTL for entries (None = no expiry)
        """
        self.backend = backend or FileBackend()
        self.default_ttl = default_ttl_seconds

    # ─────────────────────────────────────────────────────────────────
    # Core Operations
    # ─────────────────────────────────────────────────────────────────

    async def get(self, key: str) -> Any | None:
        """
        Get value by key.

        Args:
            key: The key to retrieve

        Returns:
            The value if found and not expired, None otherwise
        """
        entry = await self.backend.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            await self.backend.delete(key)
            return None
        return entry.value

    async def get_entry(self, key: str) -> BeadsEntry | None:
        """
        Get full entry with metadata.

        Args:
            key: The key to retrieve

        Returns:
            The full BeadsEntry if found, None otherwise
        """
        entry = await self.backend.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            await self.backend.delete(key)
            return None
        return entry

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """
        Set value at key.

        Args:
            key: The key to set
            value: The value to store (must be JSON-serializable)
            ttl: Time-to-live in seconds (None = use default or no expiry)

        Returns:
            True if successful
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl

        now = datetime.utcnow()
        expires_at = None
        if effective_ttl is not None:
            expires_at = datetime.utcfromtimestamp(now.timestamp() + effective_ttl)

        # Get existing entry for version
        existing = await self.backend.get(key)
        version = (existing.version + 1) if existing else 1

        entry = BeadsEntry(
            key=key,
            value=value,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            expires_at=expires_at,
            version=version,
        )

        await self.backend.set(key, entry)
        return True

    async def append(self, key: str, value: Any) -> bool:
        """
        Append value to list at key.

        If key doesn't exist, creates new list with value.
        If key exists but isn't a list, converts to list.

        Args:
            key: The key to append to
            value: The value to append

        Returns:
            True if successful
        """
        existing = await self.get(key)

        if existing is None:
            new_list = [value]
        elif isinstance(existing, list):
            new_list = existing + [value]
        else:
            # Convert existing value to list
            new_list = [existing, value]

        return await self.set(key, new_list)

    async def exists(self, key: str) -> bool:
        """
        Check if key exists and is not expired.

        Args:
            key: The key to check

        Returns:
            True if key exists and is not expired
        """
        entry = await self.backend.get(key)
        if entry is None:
            return False
        if entry.is_expired():
            await self.backend.delete(key)
            return False
        return True

    async def delete(self, key: str) -> bool:
        """
        Delete key.

        Args:
            key: The key to delete

        Returns:
            True if key was deleted, False if it didn't exist
        """
        return await self.backend.delete(key)

    # ─────────────────────────────────────────────────────────────────
    # Namespace Operations
    # ─────────────────────────────────────────────────────────────────

    async def list_keys(self, prefix: str) -> list[str]:
        """
        List all keys with given prefix.

        Args:
            prefix: Key prefix to match (e.g., "history/routines/")

        Returns:
            List of matching keys
        """
        return await self.backend.list_keys(prefix)

    async def delete_prefix(self, prefix: str) -> int:
        """
        Delete all keys with given prefix.

        Args:
            prefix: Key prefix to match

        Returns:
            Number of keys deleted
        """
        keys = await self.list_keys(prefix)
        count = 0
        for key in keys:
            if await self.delete(key):
                count += 1
        return count

    # ─────────────────────────────────────────────────────────────────
    # Atomic Operations
    # ─────────────────────────────────────────────────────────────────

    async def get_or_set(
        self,
        key: str,
        default_factory: Callable[[], Any],
        ttl: int | None = None,
    ) -> Any:
        """
        Get value or set default if not exists.

        Args:
            key: The key to get/set
            default_factory: Callable that returns default value (sync or async)
            ttl: TTL for newly created entry

        Returns:
            Existing or newly created value
        """
        existing = await self.get(key)
        if existing is not None:
            return existing

        # Support both sync and async callables
        if inspect.iscoroutinefunction(default_factory):
            value = await default_factory()
        else:
            value = default_factory()

        await self.set(key, value, ttl=ttl)
        return value

    async def increment(self, key: str, delta: int = 1) -> int:
        """
        Atomically increment integer value.

        Args:
            key: The key to increment
            delta: Amount to add (can be negative)

        Returns:
            New value after increment
        """
        existing = await self.get(key)
        if existing is None:
            new_value = delta
        elif isinstance(existing, int):
            new_value = existing + delta
        else:
            raise BeadsError(f"Cannot increment non-integer value at {key}")

        await self.set(key, new_value)
        return new_value

    async def compare_and_set(
        self,
        key: str,
        expected_version: int,
        new_value: Any,
        ttl: int | None = None,
    ) -> bool:
        """
        Set value only if version matches (optimistic locking).

        Args:
            key: The key to update
            expected_version: Expected current version
            new_value: New value to set
            ttl: TTL for entry

        Returns:
            True if update succeeded, False if version mismatch
        """
        entry = await self.get_entry(key)
        current_version = entry.version if entry else 0

        if current_version != expected_version:
            return False

        return await self.set(key, new_value, ttl=ttl)

    # ─────────────────────────────────────────────────────────────────
    # Convenience Methods
    # ─────────────────────────────────────────────────────────────────

    async def get_json(self, key: str) -> dict | list | None:
        """Get value, ensuring it's JSON-compatible."""
        return await self.get(key)

    async def set_json(self, key: str, value: dict | list, ttl: int | None = None) -> bool:
        """Set JSON value with validation."""
        # Validate JSON-serializable
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise BeadsError(f"Value is not JSON-serializable: {e}")
        return await self.set(key, value, ttl=ttl)

    async def touch(self, key: str, ttl: int | None = None) -> bool:
        """
        Update expiration without changing value.

        Args:
            key: The key to touch
            ttl: New TTL in seconds

        Returns:
            True if key exists and was touched
        """
        entry = await self.get_entry(key)
        if entry is None:
            return False
        return await self.set(key, entry.value, ttl=ttl)
```

## Storage Backend

### Backend Interface

```python
from abc import ABC, abstractmethod


class BeadsBackend(ABC):
    """Abstract backend for Beads storage."""

    @abstractmethod
    async def get(self, key: str) -> BeadsEntry | None:
        """Get entry by key."""
        pass

    @abstractmethod
    async def set(self, key: str, entry: BeadsEntry) -> None:
        """Set entry at key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key, return True if existed."""
        pass

    @abstractmethod
    async def list_keys(self, prefix: str) -> list[str]:
        """List keys with prefix."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close backend connections."""
        pass
```

### File Backend (Default)

```python
class FileBackend(BeadsBackend):
    """
    File-based backend using JSON files.

    Good for development and single-process usage.
    Keys are mapped to file paths: "a/b/c" -> "{base_dir}/a/b/c.json"
    """

    def __init__(self, base_dir: Path | str | None = None):
        """
        Initialize file backend.

        Args:
            base_dir: Base directory for storage (default: .beads/data)
        """
        if base_dir is None:
            base_dir = Path.cwd() / ".beads" / "data"
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _key_to_path(self, key: str) -> Path:
        """Convert key to file path."""
        # Percent-encode special characters for reversible mapping
        safe_key = (
            key.replace("%", "%25")  # Encode % first
               .replace(":", "%3A")
               .replace("..", "%2E%2E")
        )
        return self.base_dir / f"{safe_key}.json"

    def _path_to_key(self, path: Path) -> str:
        """Convert file path back to key."""
        relative = path.relative_to(self.base_dir)
        encoded_key = str(relative.with_suffix(""))
        # Reverse the percent-encoding
        return (
            encoded_key.replace("%2E%2E", "..")
                       .replace("%3A", ":")
                       .replace("%25", "%")
        )

    async def get(self, key: str) -> BeadsEntry | None:
        """Get entry from file."""
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            async with self._lock:
                data = json.loads(path.read_text())

            return BeadsEntry(
                key=data["key"],
                value=data["value"],
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
                version=data.get("version", 1),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupted entry at {key}: {e}")
            return None

    async def set(self, key: str, entry: BeadsEntry) -> None:
        """Write entry to file."""
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "version": entry.version,
        }

        async with self._lock:
            path.write_text(json.dumps(data, indent=2, default=str))

    async def delete(self, key: str) -> bool:
        """Delete file for key."""
        path = self._key_to_path(key)
        if not path.exists():
            return False

        async with self._lock:
            path.unlink()
        return True

    async def list_keys(self, prefix: str) -> list[str]:
        """List all keys matching prefix."""
        # Search from base_dir and filter by prefix after decoding
        if not self.base_dir.exists():
            return []

        keys = []
        for path in self.base_dir.rglob("*.json"):
            key = self._path_to_key(path)
            if key.startswith(prefix):
                keys.append(key)

        return sorted(keys)

    async def close(self) -> None:
        """No cleanup needed for file backend."""
        pass
```

### Redis Backend (Production)

```python
class RedisBackend(BeadsBackend):
    """
    Redis-based backend for production use.

    Provides:
    - Distributed access across processes
    - Built-in TTL support
    - Atomic operations
    - High performance
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "beads:",
        pool_size: int = 10,
    ):
        """
        Initialize Redis backend.

        Args:
            url: Redis connection URL
            prefix: Key prefix for all Beads keys
            pool_size: Connection pool size
        """
        self.url = url
        self.prefix = prefix
        self.pool_size = pool_size
        self._redis = None

    async def _get_redis(self):
        """Lazy initialize Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis
            self._redis = redis.from_url(
                self.url,
                max_connections=self.pool_size,
            )
        return self._redis

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> BeadsEntry | None:
        """Get entry from Redis."""
        r = await self._get_redis()
        data = await r.get(self._make_key(key))
        if data is None:
            return None

        try:
            parsed = json.loads(data)
            return BeadsEntry(
                key=parsed["key"],
                value=parsed["value"],
                created_at=datetime.fromisoformat(parsed["created_at"]),
                updated_at=datetime.fromisoformat(parsed["updated_at"]),
                expires_at=datetime.fromisoformat(parsed["expires_at"]) if parsed.get("expires_at") else None,
                version=parsed.get("version", 1),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupted entry at {key}: {e}")
            return None

    async def set(self, key: str, entry: BeadsEntry) -> None:
        """Set entry in Redis with optional TTL."""
        r = await self._get_redis()

        data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "version": entry.version,
        }

        redis_key = self._make_key(key)

        if entry.expires_at:
            ttl_seconds = int((entry.expires_at - datetime.utcnow()).total_seconds())
            if ttl_seconds > 0:
                await r.setex(redis_key, ttl_seconds, json.dumps(data, default=str))
            else:
                # TTL already passed - log warning and skip
                logger.warning(f"Skipping set for {key}: TTL already expired ({ttl_seconds}s)")
        else:
            await r.set(redis_key, json.dumps(data, default=str))

    async def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        r = await self._get_redis()
        result = await r.delete(self._make_key(key))
        return result > 0

    async def list_keys(self, prefix: str) -> list[str]:
        """List keys matching prefix using SCAN."""
        r = await self._get_redis()
        pattern = f"{self.prefix}{prefix}*"

        keys = []
        async for key in r.scan_iter(match=pattern, count=100):
            # Remove our prefix to get original key
            original_key = key.decode().removeprefix(self.prefix)
            keys.append(original_key)

        return sorted(keys)

    async def close(self) -> None:
        """Close Redis connections."""
        if self._redis:
            await self._redis.close()
```

## Configuration

```yaml
# beads_config.yaml
backend:
  type: "file"  # "file" or "redis"

  # File backend settings
  file:
    base_dir: ".beads/data"

  # Redis backend settings
  redis:
    url: "redis://localhost:6379"
    prefix: "beads:"
    pool_size: 10

defaults:
  ttl_seconds: null  # No default expiry

namespaces:
  # Per-namespace TTL overrides
  "system/seen-messages/": 3600      # 1 hour for dedup cache
  "system/health/": 60               # 1 minute for health checks
  "history/routines/": null          # Never expire routines
```

## Usage Examples

### Basic Operations

```python
async def basic_usage():
    beads = BeadsClient()

    # Simple get/set
    await beads.set("config/max-retries", 3)
    retries = await beads.get("config/max-retries")  # 3

    # With TTL (1 hour)
    await beads.set("cache/result", {"data": "value"}, ttl=3600)

    # Check existence
    if await beads.exists("cache/result"):
        print("Cache hit!")

    # Append to list
    await beads.append("logs/errors", {"error": "something failed"})
    await beads.append("logs/errors", {"error": "another failure"})
    errors = await beads.get("logs/errors")  # List of 2 errors

    # List keys in namespace
    keys = await beads.list_keys("logs/")  # ["logs/errors"]
```

### Tier Integration

```python
# In Abstract Architect
class ContextManager:
    def __init__(self, beads: BeadsClient):
        self.beads = beads

    async def initialize_context(self, context_key: str, task: ArchitectTask):
        """Initialize task context in Beads."""
        await self.beads.set(f"{context_key}/objective", {
            "description": task.objective,
            "constraints": task.constraints,
            "created_at": datetime.utcnow().isoformat(),
        })

        await self.beads.set(f"{context_key}/progress", {
            "status": "initialized",
            "steps_completed": 0,
            "revisions": 0,
        })

    async def update_progress(self, context_key: str, step: str, success: bool):
        """Update progress after step completion."""
        progress = await self.beads.get(f"{context_key}/progress") or {}
        progress["steps_completed"] = progress.get("steps_completed", 0) + 1
        progress["last_step"] = step
        progress["last_success"] = success
        await self.beads.set(f"{context_key}/progress", progress)


# In Routing Dispatcher
class GraphStateStore:
    def __init__(self, beads: BeadsClient):
        self.beads = beads

    async def save_graph(self, graph_id: str, graph_data: dict):
        """Save execution graph state."""
        await self.beads.set(f"graphs/{graph_id}", graph_data)

    async def record_routing_decision(self, task_embedding: list, specialist: str, success: bool):
        """Record routing decision for future training."""
        await self.beads.append("routing/training-data", {
            "embedding": task_embedding,
            "specialist": specialist,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        })
```

### Deduplication Pattern

```python
async def handle_message_with_dedup(
    beads: BeadsClient,
    tier: str,
    msg: Message,
    handler: Callable[[Message], Message],
) -> Message:
    """Handle message with deduplication."""
    cache_key = f"system/seen-messages/{tier}/{msg.id}"

    # Check for cached response
    cached = await beads.get(cache_key)
    if cached is not None:
        logger.info(f"Returning cached response for {msg.id}")
        return Message(**cached)

    # Process message
    response = await handler(msg)

    # Cache response (1 hour TTL)
    await beads.set(cache_key, response.to_dict(), ttl=3600)

    return response
```

### Routine Caching

```python
async def get_or_create_routine(
    beads: BeadsClient,
    task_signature: str,
    create_routine: Callable[[], dict],
) -> dict:
    """Get cached routine or create new one."""
    cache_key = f"history/routines/{task_signature}"

    # Check cache
    cached = await beads.get(cache_key)
    if cached is not None:
        logger.info(f"Found cached routine for {task_signature}")
        return cached

    # Create new routine (supports sync or async)
    if inspect.iscoroutinefunction(create_routine):
        routine = await create_routine()
    else:
        routine = create_routine()

    # Cache indefinitely (no TTL)
    await beads.set(cache_key, routine)

    return routine
```

## Testing

```python
import pytest


@pytest.fixture
async def beads():
    """Create test BeadsClient with in-memory backend."""
    client = BeadsClient(backend=MemoryBackend())
    yield client
    await client.backend.close()


class MemoryBackend(BeadsBackend):
    """In-memory backend for testing."""

    def __init__(self):
        self._store: dict[str, BeadsEntry] = {}

    async def get(self, key: str) -> BeadsEntry | None:
        return self._store.get(key)

    async def set(self, key: str, entry: BeadsEntry) -> None:
        self._store[key] = entry

    async def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    async def list_keys(self, prefix: str) -> list[str]:
        return sorted(k for k in self._store if k.startswith(prefix))

    async def close(self) -> None:
        self._store.clear()


@pytest.mark.asyncio
async def test_get_set(beads):
    await beads.set("test/key", {"value": 42})
    result = await beads.get("test/key")
    assert result == {"value": 42}


@pytest.mark.asyncio
async def test_get_nonexistent(beads):
    result = await beads.get("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_append(beads):
    await beads.append("test/list", "first")
    await beads.append("test/list", "second")
    result = await beads.get("test/list")
    assert result == ["first", "second"]


@pytest.mark.asyncio
async def test_ttl_expiry(beads):
    await beads.set("test/expiring", "value", ttl=1)

    # Should exist immediately
    assert await beads.exists("test/expiring")

    # Wait for expiry
    await asyncio.sleep(1.1)

    # Should be gone
    assert not await beads.exists("test/expiring")


@pytest.mark.asyncio
async def test_list_keys(beads):
    await beads.set("prefix/a", 1)
    await beads.set("prefix/b", 2)
    await beads.set("other/c", 3)

    keys = await beads.list_keys("prefix/")
    assert keys == ["prefix/a", "prefix/b"]


@pytest.mark.asyncio
async def test_increment(beads):
    await beads.set("counter", 0)

    result = await beads.increment("counter")
    assert result == 1

    result = await beads.increment("counter", delta=5)
    assert result == 6

    result = await beads.increment("counter", delta=-2)
    assert result == 4


@pytest.mark.asyncio
async def test_compare_and_set(beads):
    await beads.set("versioned", "v1")
    entry = await beads.get_entry("versioned")

    # Should succeed with correct version
    success = await beads.compare_and_set("versioned", entry.version, "v2")
    assert success

    # Should fail with stale version
    success = await beads.compare_and_set("versioned", entry.version, "v3")
    assert not success
```

## Migration Path

### Phase 1: File Backend (MVP)
- Use FileBackend for initial development
- Single-process, local storage
- Easy debugging via JSON files

### Phase 2: Redis Backend (Production)
- Switch to RedisBackend for multi-process
- Enable distributed access
- Leverage Redis TTL for automatic cleanup

### Phase 3: Optimizations
- Add connection pooling
- Implement batch operations
- Add metrics/observability
