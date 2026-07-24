"""In-process Prometheus-style metrics (design §10 observability).

Single-process counters + latency histograms rendered in the Prometheus text
exposition format at GET /metrics. Correct for one API replica / a demo; for a
multi-replica deployment use a real Prometheus client and aggregate across
replicas. Live ingest queue-depth gauges are computed from Postgres at scrape
time by the /metrics route (see api/search.py).
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_counters: dict[tuple, float] = {}
_hsum: dict[tuple, float] = {}
_hcount: dict[tuple, float] = {}
_hbucket: dict[tuple, dict] = {}
_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)  # seconds


def _lbl(labels: dict | None) -> tuple:
    return tuple(sorted((labels or {}).items()))


def inc(name: str, labels: dict | None = None, value: float = 1.0) -> None:
    key = (name, _lbl(labels))
    with _lock:
        _counters[key] = _counters.get(key, 0.0) + value


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
    """Prometheus text exposition of all counters + histograms."""
    out: list[str] = []
    with _lock:
        counters = list(_counters.items())
        hkeys = list(_hcount.keys())
        hsum = dict(_hsum)
        hcount = dict(_hcount)
        hbucket = {k: dict(v) for k, v in _hbucket.items()}

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
