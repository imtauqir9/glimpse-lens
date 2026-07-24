"""Shared Redis connection — the cluster-wide state backend (enterprise Phase 2).

Rate limiting and metrics counters are correct only if all replicas share state.
When REDIS_URL is set, `client()` returns a pooled Redis handle; otherwise it
returns None and the callers fall back to per-process in-memory state (fine for a
single replica / local dev). This keeps the app runnable with zero infra while
becoming multi-replica-correct the moment a Redis is attached.
"""
from __future__ import annotations

import os

REDIS_URL = os.getenv("REDIS_URL", "").strip()
_client = None
_tried = False


def client():
    """Return a shared Redis client, or None if REDIS_URL is unset/unreachable."""
    global _client, _tried
    if _client is not None:
        return _client
    if _tried or not REDIS_URL:
        return None
    _tried = True
    try:
        import redis  # lazy: only needed when REDIS_URL is set
        c = redis.Redis.from_url(REDIS_URL, socket_timeout=1.0,
                                 socket_connect_timeout=1.0, decode_responses=True)
        c.ping()
        _client = c
        print(f"[redis] connected ({REDIS_URL.split('@')[-1]})")
    except Exception as exc:  # noqa: BLE001
        print(f"[redis] unavailable ({type(exc).__name__}: {exc}) — using in-memory fallback")
        _client = None
    return _client


def available() -> bool:
    return client() is not None
