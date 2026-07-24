"""
src/db_documents.py — document (paper/deck) rows, reusing the ms_videos table.

We chose to REUSE `ms_videos` (add a `kind` column) rather than a separate
`ms_documents` table — fewer moving parts, and the provided `db.set_status`,
`db.bump_attempts`, `db.get_video`, `db.videos_by_ids` already operate on any
row in it, so the ingest modules and read path need no DB changes.

This module is ADDITIVE — it does not modify the provided `db.py`. It reuses
`db.pool()` for connections. Import it once at startup (alongside
`db.init_schema()`) to run the idempotent column migration.

Column semantics for a document row in ms_videos:
    id           pp_<hex> (paper) | dk_<hex> (deck)
    source       'paper' | 'deck'         (existing rows stay 'youtube'|'upload')
    kind         'paper' | 'deck'         (existing rows default to 'video')
    url          the document uri (arXiv URL, etc.)   — OR —
    storage_key  bucket key if uploaded via presign (like video uploads)
    title        display title
    status       pending → parsing → chunking → embedding → indexed | failed
    frame_count  reused as the CHUNK count for documents
    progress     0..1 within the current stage        (→ 'pct' in the API)
"""
from __future__ import annotations

import uuid
from typing import Any

from .db import pool

_KINDS = ("paper", "deck")


def add_kind_column() -> None:
    """Idempotent migration: add `kind` to ms_videos, defaulting existing rows
    (all videos) to 'video'. Safe to call on every boot."""
    with pool().connection() as conn:
        conn.execute(
            "ALTER TABLE ms_videos ADD COLUMN IF NOT EXISTS kind "
            "TEXT NOT NULL DEFAULT 'video'"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS ms_videos_kind_idx ON ms_videos (user_id, kind)"
        )


def new_document_id(kind: str) -> str:
    prefix = "pp" if kind == "paper" else "dk"
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def upsert_pending_document(
    *, source_id: str, user_id: str, kind: str, uri: str | None,
    storage_key: str | None, title: str | None,
) -> dict:
    """Insert a document as `pending` (re-submitting the same id resets it),
    mirroring db.upsert_pending's shape for videos. Returns the row.

    `source` and `kind` both carry the document kind so a document is
    self-describing whether code keys off `source` (like the video flow) or the
    new `kind` column.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")
    with pool().connection() as conn:
        row = conn.execute(
            """
            INSERT INTO ms_videos
                (id, user_id, source, kind, url, storage_key, title, status)
            VALUES
                (%(id)s, %(user_id)s, %(kind)s, %(kind)s, %(url)s,
                 %(storage_key)s, %(title)s, 'pending')
            ON CONFLICT (id) DO UPDATE SET
                url = COALESCE(EXCLUDED.url, ms_videos.url),
                storage_key = COALESCE(EXCLUDED.storage_key, ms_videos.storage_key),
                title = COALESCE(EXCLUDED.title, ms_videos.title),
                kind = EXCLUDED.kind,
                status = 'pending', error = NULL, progress = NULL, updated_at = now()
            RETURNING *
            """,
            {"id": source_id, "user_id": user_id, "kind": kind,
             "url": uri, "storage_key": storage_key, "title": title},
        ).fetchone()
    return row


def list_sources(user_id: str, kind: str | None = None) -> list[dict[str, Any]]:
    """Unified video + document listing for GET /admin/sources.

    Returns every source (video, paper, deck) for the user, newest first, with
    `kind` and `pct` (0..100, from the 0..1 `progress`). One query — because
    documents live in the same table as videos.
    """
    q = "SELECT * FROM ms_videos WHERE user_id = %s"
    params: list = [user_id]
    if kind:
        q += " AND kind = %s"
        params.append(kind)
    q += " ORDER BY created_at DESC"
    with pool().connection() as conn:
        rows = conn.execute(q, tuple(params)).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "kind": r.get("kind") or "video",
            "title": r.get("title"),
            "status": r["status"],
            "pct": round((r.get("progress") or 0.0) * 100),   # 0..1 → 0..100
            "error": r.get("error"),
            "chunk_count": r.get("frame_count"),   # frame_count reused as chunks for docs
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
        })
    return out
