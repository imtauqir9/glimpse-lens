"""Stale-run reconciler — the redelivery half of no-loss (design §9).

A worker killed mid-ingest leaves its source in a NON-terminal status with no
running flow: Prefect marks the crashed run "Crashed" but does not reschedule it.
Idempotent point ids + delete-before-upsert already make *re-running* safe — but
something must *re-trigger* it. This background sweep (running in the always-up
API process) finds sources stuck past RECONCILE_STALE_S and re-enqueues them, so
a live worker picks them up and drives them to `indexed`. This is what turns
"crash-safe code" into an actual no-loss guarantee.

Two thresholds, because "stuck" and "waiting" look identical in one column:

  RUNNING (fetching/sampling/parsing/chunking/embedding) — a worker HAD this row
    and stopped touching it. Past RECONCILE_STALE_S (60s) that is a crash, and
    re-enqueueing is the whole point of this module.

  QUEUED — handed to Prefect, not yet picked up. Under a real backfill this is
    the NORMAL state of everything behind the worker pool, so the running
    threshold would re-enqueue the entire waiting line every sweep: duplicate
    flow runs, each re-parsing and re-embedding the same document, stealing the
    capacity the queue was waiting for. Only a much longer silence
    (RECONCILE_QUEUE_STALE_S, 300s) means the run itself is gone.

  PENDING splits by kind:
    • videos are never reaped — they belong to the fair dispatcher
      (src/dispatcher.py), which admits them round-robin across users.
      Re-enqueuing one here would call jobs.enqueue_video directly and walk
      straight past WFQ, which is the starvation the dispatcher exists to stop.
    • documents ARE reaped, on the queued threshold. They skip the dispatcher
      (wfq_claim takes kind='video' only) and are flipped to `queued` the moment
      their Prefect run exists, so a document still `pending` minutes later means
      the scheduling call never landed — the one crash window /admin/documents
      has, since it schedules in a BackgroundTask after the 202 is already sent.

Env: RECONCILE_STALE_S (60), RECONCILE_QUEUE_STALE_S (300),
     RECONCILE_INTERVAL_S (20).
"""
from __future__ import annotations

import os
import threading
import time

from . import config, db, jobs, jobs_documents

STALE_S = int(os.getenv("RECONCILE_STALE_S", "60"))
QUEUE_STALE_S = int(os.getenv("RECONCILE_QUEUE_STALE_S", "300"))
INTERVAL_S = int(os.getenv("RECONCILE_INTERVAL_S", "20"))
_QUEUE_STATUSES = ("queued",)
_DOCUMENT_KINDS = ("paper", "deck")


def reap_once() -> int:
    """Re-enqueue sources a worker abandoned. Returns how many were re-enqueued."""
    with db.pool().connection() as conn:
        rows = conn.execute(
            "SELECT id, user_id, kind, url, storage_key, title, attempts FROM ms_videos "
            "WHERE (status = ANY(%s) "
            "       AND updated_at < now() - (%s * interval '1 second')) "
            "   OR (status = ANY(%s) "
            "       AND updated_at < now() - (%s * interval '1 second')) "
            "   OR (status = 'pending' AND kind = ANY(%s) "
            "       AND updated_at < now() - (%s * interval '1 second'))",
            (list(config.RUNNING_STATUSES), STALE_S,
             list(_QUEUE_STATUSES), QUEUE_STALE_S,
             list(_DOCUMENT_KINDS), QUEUE_STALE_S),
        ).fetchall()
    n = 0
    for r in rows:
        kind = (r.get("kind") or "video")
        # DEAD-LETTER (design §7, §20). A source that raises gets terminal
        # `failed` from its own flow and never reaches this sweep. The dangerous
        # one is a source that KILLS the worker — an OOM on a pathological PDF is
        # not an exception, so nothing marks it failed, and re-enqueueing it just
        # kills the next worker. Without a cap that is an infinite loop that eats
        # the pool. `attempts` was already counted on every run and only ever
        # printed; this is what makes it mean something.
        attempts = int(r.get("attempts") or 0)
        if attempts >= config.MAX_INGEST_ATTEMPTS:
            db.set_status(r["id"], "failed",
                          error=f"dead-lettered after {attempts} attempts — "
                                f"the source keeps stranding its worker. Inspect "
                                f"it and POST a retry once fixed.")
            print(f'[reconciler] dead-lettered id="{r["id"]}" after {attempts} attempts')
            continue
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
