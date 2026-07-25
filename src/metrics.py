"""Prometheus-style metrics (design §10 observability).

Counters + latency histograms rendered in the Prometheus text exposition format
at GET /metrics, and as a JSON rollup at GET /api/metrics.json.

Counters are cluster-wide when REDIS_URL is set: `inc()` writes through to a
Redis hash, and BOTH readers (`render()` and `snapshot()`) prefer it, so the two
endpoints always agree and the totals survive a machine restart. Without Redis
they fall back to this process's memory — fine for one replica / local dev.

Latency histograms are still per-replica in-process state; they reset when a
machine stops. Live ingest queue-depth gauges are computed from Postgres at
scrape time by the /metrics route (see api/search.py).
"""
from __future__ import annotations

import threading

from . import redis_client

_CTR_HASH = "glimpse:ctr"   # Redis hash holding cluster-wide counter totals

_lock = threading.Lock()
_counters: dict[tuple, float] = {}
_hsum: dict[tuple, float] = {}
_hcount: dict[tuple, float] = {}
_hbucket: dict[tuple, dict] = {}
_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)  # seconds


def _lbl(labels: dict | None) -> tuple:
    return tuple(sorted((labels or {}).items()))


def _field(name: str, labels: tuple) -> str:
    return name + "\x1f" + "&".join(f"{k}={v}" for k, v in labels)


def _parse_field(field: str):
    name, _, lbl = field.partition("\x1f")
    labels = {}
    if lbl:
        for pair in lbl.split("&"):
            k, _, v = pair.partition("=")
            labels[k] = v
    return name, labels


def inc(name: str, labels: dict | None = None, value: float = 1.0) -> None:
    lab = _lbl(labels)
    with _lock:
        _counters[(name, lab)] = _counters.get((name, lab), 0.0) + value
    r = redis_client.client()   # cluster-wide total (Phase 2)
    if r is not None:
        try:
            r.hincrbyfloat(_CTR_HASH, _field(name, lab), value)
        except Exception:  # noqa: BLE001
            pass


def observe(name: str, seconds: float, labels: dict | None = None) -> None:
    key = (name, _lbl(labels))
    with _lock:
        _hsum[key] = _hsum.get(key, 0.0) + seconds
        _hcount[key] = _hcount.get(key, 0.0) + 1
        b = _hbucket.setdefault(key, {})
        for le in _BUCKETS:
            if seconds <= le:
                b[le] = b.get(le, 0.0) + 1
        b["+Inf"] = b.get("+Inf", 0.0) + 1


def _fmt(labels: tuple, extra: dict | None = None) -> str:
    d = dict(labels)
    if extra:
        d.update(extra)
    if not d:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in d.items()) + "}"


def render() -> str:
    """Prometheus text exposition of all counters + histograms.

    COUNTERS are cluster-wide when Redis is available — the same source
    `snapshot()` uses, so /metrics and /api/metrics.json can't disagree. This
    matters more than it looks: the api process auto-stops when idle
    (min_machines_running = 0), so in-process counters reset to zero on every
    idle cycle and Prometheus sees a counter reset. The Redis totals are
    monotonic across restarts, which is what a counter is supposed to be.

    Assumes ONE scrape target in front of the cluster (the public hostname, as
    configured in monitoring/prometheus.yml). Scraping each machine separately
    would multiply the cluster total by the number of replicas.

    HISTOGRAMS stay per-replica: bucket state is in-process, so latency still
    resets when a machine stops. Fixing that needs Redis-backed buckets, which
    is a bigger change than this one.
    """
    out: list[str] = []
    with _lock:
        counters = list(_counters.items())
        hkeys = list(_hcount.keys())
        hsum = dict(_hsum)
        hcount = dict(_hcount)
        hbucket = {k: dict(v) for k, v in _hbucket.items()}

    r = redis_client.client()
    if r is not None:
        try:
            counters = [(_parse_field(f), float(v))
                        for f, v in r.hgetall(_CTR_HASH).items()]
        except Exception:  # noqa: BLE001
            pass            # Redis blipped — this replica's numbers still render

    # hgetall returns arbitrary order; keep each metric family contiguous so the
    # single "# TYPE" line still precedes all of its samples.
    counters.sort(key=lambda kv: (kv[0][0], sorted(dict(kv[0][1]).items())))

    seen = set()
    for (name, labels), val in counters:
        if name not in seen:
            out.append(f"# TYPE {name} counter"); seen.add(name)
        out.append(f"{name}{_fmt(labels)} {val:g}")

    for (name, labels) in hkeys:
        if name not in seen:
            out.append(f"# TYPE {name} histogram"); seen.add(name)
        b = hbucket.get((name, labels), {})
        cum = 0.0
        for le in _BUCKETS:
            cum = b.get(le, cum)
            out.append(f'{name}_bucket{_fmt(labels, {"le": le})} {cum:g}')
        out.append(f'{name}_bucket{_fmt(labels, {"le": "+Inf"})} {b.get("+Inf", 0.0):g}')
        out.append(f"{name}_sum{_fmt(labels)} {hsum.get((name, labels), 0.0):g}")
        out.append(f"{name}_count{_fmt(labels)} {hcount.get((name, labels), 0.0):g}")

    return "\n".join(out) + "\n"


# ── Token + cost metering (design §18) ───────────────────────────────────────
_PRICE_PER_M = {  # ($ per 1M input tokens, $ per 1M output tokens)
    "claude-opus-4-8": (5.0, 25.0),   "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),   "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4o-mini": (0.15, 0.60),      "gpt-4o": (2.5, 10.0),
}


def record_tokens(model: str, in_tok: int, out_tok: int) -> None:
    """Record actual LLM token usage + $ cost, labelled by model."""
    inc("glimpse_llm_input_tokens_total", {"model": model}, in_tok)
    inc("glimpse_llm_output_tokens_total", {"model": model}, out_tok)
    pin, pout = _PRICE_PER_M.get(model, (0.0, 0.0))
    inc("glimpse_llm_cost_usd_total", {"model": model},
        in_tok / 1e6 * pin + out_tok / 1e6 * pout)


def _pct(bucket: dict, count: float, q: float) -> float:
    if count <= 0:
        return 0.0
    target = q * count
    for le in _BUCKETS:
        if bucket.get(le, 0.0) >= target:
            return le
    return _BUCKETS[-1]


def snapshot() -> dict:
    """JSON-friendly rollup for the dashboard: totals, per-route latency
    percentiles, request status breakdown, token + cost totals."""
    with _lock:
        mem = [((n, dict(l)), v) for (n, l), v in _counters.items()]
        hcount = dict(_hcount)
        hsum = dict(_hsum)
        hbucket = {k: dict(v) for k, v in _hbucket.items()}

    # Counters: cluster-wide from Redis when available, else this replica's.
    clist = None
    r = redis_client.client()
    if r is not None:
        try:
            clist = [(_parse_field(f), float(v)) for f, v in r.hgetall(_CTR_HASH).items()]
        except Exception:  # noqa: BLE001
            clist = None
    scope = "cluster (redis)" if clist is not None else "this replica"
    if clist is None:
        clist = mem

    def total(name):
        return sum(v for (n, _), v in clist if n == name)

    # Latency is per-replica (Prometheus aggregates across replicas at scrape).
    latency = {}
    for (name, labels), cnt in hcount.items():
        if name != "glimpse_request_duration_seconds":
            continue
        route = dict(labels).get("route", "?")
        b = hbucket.get((name, labels), {})
        latency[route] = {
            "count": cnt,
            "avg_ms": round(hsum.get((name, labels), 0.0) / cnt * 1000, 1) if cnt else 0,
            "p50_ms": round(_pct(b, cnt, 0.5) * 1000, 1),
            "p95_ms": round(_pct(b, cnt, 0.95) * 1000, 1),
        }

    by_status = {}
    for (name, labels), v in clist:
        if name == "glimpse_requests_total":
            st = str(labels.get("status", "?"))
            by_status[st] = by_status.get(st, 0) + v

    return {
        "scope": scope,
        "requests_total": total("glimpse_requests_total"),
        "llm_answers_total": total("glimpse_llm_answers_total"),
        "rate_limited_total": total("glimpse_rate_limited_total"),
        "input_tokens": total("glimpse_llm_input_tokens_total"),
        "output_tokens": total("glimpse_llm_output_tokens_total"),
        "cost_usd": round(total("glimpse_llm_cost_usd_total"), 4),
        "latency": latency,
        "latency_scope": "this replica",
        "by_status": by_status,
    }
