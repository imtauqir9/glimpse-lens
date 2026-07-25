"""Shared Redis connection — the cluster-wide state backend (enterprise Phase 2).

Rate limiting and metrics counters are correct only if all replicas share state.
When REDIS_URL is set, `client()` returns a pooled Redis handle; otherwise it
returns None and the callers fall back to per-process in-memory state (fine for a
single replica / local dev). This keeps the app runnable with zero infra while
becoming multi-replica-correct the moment a Redis is attached.
"""
from __future__ import annotations

import os
import time

REDIS_URL = os.getenv("REDIS_URL", "").strip()
# Managed Redis is not always a LAN-speed hop: Fly's Upstash instance answers the
# first PING in ~1.4s (same region), so the old hardcoded 1.0s timed out every
# time and the app ran degraded while looking configured. 5s is still far below
# any user-visible budget — a healthy round trip is single-digit ms.
TIMEOUT_S = float(os.getenv("REDIS_TIMEOUT_S", "5"))
# Don't latch OFF forever on one blip. A failed connect is retried, but at most
# once per cooldown, so a genuinely-down Redis costs one timeout per 30s rather
# than one per request.
RETRY_COOLDOWN_S = float(os.getenv("REDIS_RETRY_COOLDOWN_S", "30"))
_client = None
_next_try = 0.0


def client():
    """Return a shared Redis client, or None if REDIS_URL is unset/unreachable."""
    global _client, _next_try
    if _client is not None:
        return _client
    if not REDIS_URL or time.monotonic() < _next_try:
        return None
    _next_try = time.monotonic() + RETRY_COOLDOWN_S
    try:
        import redis  # lazy: only needed when REDIS_URL is set
        c = redis.Redis.from_url(REDIS_URL, socket_timeout=TIMEOUT_S,
                                 socket_connect_timeout=TIMEOUT_S, decode_responses=True)
        c.ping()
        _client = c
        print(f"[redis] connected ({REDIS_URL.split('@')[-1]})")
    except Exception as exc:  # noqa: BLE001
        print(f"[redis] unavailable ({type(exc).__name__}: {exc}) — using in-memory "
              f"fallback, retrying in {RETRY_COOLDOWN_S:.0f}s")
        _client = None
    return _client


def available() -> bool:
    return client() is not None
