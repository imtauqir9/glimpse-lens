"""Lightweight in-process guardrails — abuse/fairness (design §11) + cost
visibility (§18).

Per-user sliding-window rate limiting and a simple per-user LLM-call meter.
In-memory and single-process: correct for one API replica / a demo. For a
multi-replica production deployment, move the counters behind Redis (same
function signatures) so the limit is global rather than per-process.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from . import redis_client

# Env-tunable knobs.
RATE_MAX = int(os.getenv("RATE_LIMIT_PER_MIN", "30"))      # requests / minute / user
RATE_WINDOW = 60.0

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)   # user_id -> monotonic request times
_calls: dict[str, int] = defaultdict(int)      # user_id -> LLM calls served (session)


class RateLimited(Exception):
    """Raised when a user exceeds RATE_MAX requests in RATE_WINDOW seconds."""
    def __init__(self, retry_after: float):
        self.retry_after = max(1.0, retry_after)
        super().__init__("rate limit exceeded")


def check_rate(user_id: str) -> None:
    """Sliding-window per-user rate limit. Raises RateLimited when exceeded.

    Fail-closed on a missing tenant: an empty user_id is rejected outright so a
    caller can never dodge the limit (or tenant isolation) by omitting it."""
    if not user_id:
        raise RateLimited(retry_after=RATE_WINDOW)

    # Cluster-wide limit via Redis (correct across replicas) — fixed window.
    r = redis_client.client()
    if r is not None:
        try:
            bucket = int(time.time() // RATE_WINDOW)
            key = f"rl:{user_id}:{bucket}"
            n = r.incr(key)
            if n == 1:
                r.expire(key, int(RATE_WINDOW) + 1)
            if n > RATE_MAX:
                raise RateLimited(retry_after=RATE_WINDOW - (time.time() % RATE_WINDOW))
            return
        except RateLimited:
            raise
        except Exception:  # noqa: BLE001 — a Redis blip degrades to in-memory
            pass

    # In-memory fallback (single replica / no Redis): sliding window.
    now = time.monotonic()
    with _lock:
        dq = _hits[user_id]
        while dq and now - dq[0] > RATE_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_MAX:
            raise RateLimited(retry_after=RATE_WINDOW - (now - dq[0]))
        dq.append(now)


def meter_llm_call(user_id: str) -> int:
    """Count an LLM answer served for this user (cost visibility, §18). Returns
    the running count. Structured line so it's greppable in logs / scrapeable
    later; real token+$ metering threads usage out of src/llm.py as a follow-up."""
    with _lock:
        _calls[user_id] += 1
        n = _calls[user_id]
    print(f'[meter] event=llm_answer user_id="{user_id}" calls={n}')
    try:
        from . import metrics
        metrics.inc("glimpse_llm_answers_total")
    except Exception:
        pass
    return n
