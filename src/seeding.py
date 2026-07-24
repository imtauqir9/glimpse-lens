"""Sample-corpus seeding — run to completion BEFORE the app serves.

This is the startup gate: seed.py runs it as a one-shot container that must
exit 0 before api/worker start (docker-compose depends_on ...
service_completed_successfully), so the UI is never reachable with a
half-indexed corpus. Idempotent and durable (Qdrant Cloud): once the four
talks are indexed they stay indexed, so every later start finishes in seconds.
"""
from __future__ import annotations

import time
import urllib.request

from . import config, db
from .ingest.pipeline import ingest_video
from .rag import vector_store
from .samples import SAMPLE_VIDEOS, sample_video_id

_MAX_PASSES = 3  # re-attempt videos that fail (e.g. a transient YouTube hiccup)


def wait_for_clip(timeout: int = 600) -> None:
    """Block until the CLIP service answers /healthz — first boot downloads the
    model (~600MB). No-op when embedding is in-process (CLIP_SERVICE_URL unset)."""
    if not config.CLIP_SERVICE_URL:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(config.CLIP_SERVICE_URL + "/healthz", timeout=5) as r:
                if r.status == 200:
                    print("[seed] CLIP service ready", flush=True)
                    return
        except Exception:
            pass
        print("[seed] waiting for CLIP service to warm up…", flush=True)
        time.sleep(5)
    print("[seed] CLIP service not ready in time — attempting anyway", flush=True)


def _not_indexed() -> list[dict]:
    out = []
    for v in SAMPLE_VIDEOS:
        row = db.get_video(sample_video_id(v["url"]))
        if (row or {}).get("status") != "indexed":
            out.append(v)
    return out


def seed_to_completion() -> bool:
    """Index every sample, retrying failures. Returns True iff all four end up
    indexed. Blocking — the caller (seed.py) gates the app on this."""
    if not config.SEED_SAMPLE_VIDEOS:
        print("[seed] SEED_SAMPLE_VIDEOS=false — skipping", flush=True)
        return True

    db.init_schema()
    vector_store.ensure_collection()

    # Light sampling for the demo corpus so all four finish in ~2 min on CPU
    # (the Karpathy talk is 1h). User uploads run in separate Prefect
    # subprocesses that re-read config, so they keep full quality.
    config.MAX_FRAMES = min(config.MAX_FRAMES, 60)
    config.FRAME_INTERVAL_SEC = max(config.FRAME_INTERVAL_SEC, 5.0)

    wait_for_clip()

    pending = _not_indexed()
    if not pending:
        print("[seed] all four samples already indexed — ready", flush=True)
        return True

    for attempt in range(1, _MAX_PASSES + 1):
        todo = _not_indexed()
        if not todo:
            break
        print(f"[seed] pass {attempt}/{_MAX_PASSES}: indexing {len(todo)} sample(s)", flush=True)
        for v in todo:
            vid = sample_video_id(v["url"])
            print(f"[seed] -> {vid}: {v['title']}", flush=True)
            db.upsert_pending({"id": vid, "user_id": config.DEFAULT_USER_ID,
                               "source": "youtube", "url": v["url"],
                               "storage_key": None, "source_hash": vid,
                               "title": v["title"]})
            try:
                ingest_video(video_id=vid, user_id=config.DEFAULT_USER_ID)
            except Exception as exc:
                print(f"[seed] {vid} failed ({type(exc).__name__}: {exc})", flush=True)

    remaining = _not_indexed()
    if remaining:
        names = ", ".join(sample_video_id(v["url"]) for v in remaining)
        print(f"[seed] STILL not indexed after {_MAX_PASSES} passes: {names}", flush=True)
        return False
    print("[seed] sample corpus complete — all four indexed", flush=True)
    return True
