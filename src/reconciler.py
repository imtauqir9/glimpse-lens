"""Stale-run reconciler — the redelivery half of no-loss (design §9).

A worker killed mid-ingest leaves its source in a NON-terminal status with no
running flow: Prefect marks the crashed run "Crashed" but does not reschedule it.
Idempotent point ids + delete-before-upsert already make *re-running* safe — but
something must *re-trigger* it. This background sweep (running in the always-up
API process) finds sources stuck past RECONCILE_STALE_S and re-enqueues them, so
a live worker picks them up and drives them to `indexed`. This is what turns
"crash-safe code" into an actual no-loss guarantee.

Env: RECONCILE_STALE_S (default 60), RECONCILE_INTERVAL_S (default 20).
"""
from __future__ import annotations

import os
import threading
import time

from . import db, jobs, jobs_documents

STALE_S = int(os.getenv("RECONCILE_STALE_S", "60"))
INTERVAL_S = int(os.getenv("RECONCILE_INTERVAL_S", "20"))
_TERMINAL = ("indexed", "failed", "skipped")


def reap_once() -> int:
    """Re-enqueue every source stuck in a non-terminal status past STALE_S.
    Returns how many were re-enqueued."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, user_id, kind, url, storage_key, title FROM ms_videos "
            "WHERE NOT (status = ANY(%s)) "
            "AND updated_at < now() - (%s * interval '1 second')",
            (list(_TERMINAL), STALE_S),
        ).fetchall()
    n = 0
    for r in rows:
        kind = (r.get("kind") or "video")
        try:
            if kind in ("paper", "deck"):
                jobs_documents.enqueue_document(
                    source_id=r["id"], uri=r.get("url"),
                    title=r.get("title") or r["id"], kind=kind,
                    user_id=r["user_id"], storage_key=r.get("storage_key"))
            else:
                jobs.enqueue_video(r["id"], r["user_id"])
            # Touch updated_at so the same row isn't re-reaped on the next sweep
            # before its re-run has a chance to move it forward.
            db.set_status(r["id"], "queued")
            n += 1
            print(f'[reconciler] re-enqueued stuck source id="{r["id"]}" kind={kind}')
        except Exception as exc:  # noqa: BLE001
            print(f'[reconciler] re-enqueue failed id="{r["id"]}": {exc}')
    return n


def start_in_background() -> None:
    """Daemon sweep loop. Started once from the API lifespan."""
    def _loop():
        print(f"[reconciler] started (stale>{STALE_S}s, every {INTERVAL_S}s)")
        while True:
            try:
                reap_once()
            except Exception as exc:  # noqa: BLE001
                print(f"[reconciler] sweep error: {exc}")
            time.sleep(INTERVAL_S)

    threading.Thread(target=_loop, daemon=True).start()
