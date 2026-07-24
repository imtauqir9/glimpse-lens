"""Audit log — who did what, when (enterprise Phase 3).

Every mutating action (ingest, delete, retry, key mint) appends an immutable row
to `ms_audit`: actor (user_id) + role, action, target, source IP, metadata, and a
server timestamp. This is the security-review + compliance trail; it is
append-only (no update/delete path from the app).
"""
from __future__ import annotations

import json

from .db import pool

SCHEMA = """
CREATE TABLE IF NOT EXISTS ms_audit (
    id       BIGSERIAL PRIMARY KEY,
    ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id  TEXT,
    role     TEXT,
    action   TEXT NOT NULL,
    target   TEXT,
    ip       TEXT,
    meta     JSONB
);
CREATE INDEX IF NOT EXISTS ms_audit_ts_idx   ON ms_audit (ts DESC);
CREATE INDEX IF NOT EXISTS ms_audit_user_idx ON ms_audit (user_id, ts DESC);
"""


def init_schema() -> None:
    with pool().connection() as c:
        c.execute(SCHEMA)


def record(user_id: str | None, role: str | None, action: str,
           target: str | None = None, ip: str | None = None,
           meta: dict | None = None) -> None:
    """Append one audit event. Never raises into the request path."""
    try:
        with pool().connection() as c:
            c.execute(
                "INSERT INTO ms_audit (user_id, role, action, target, ip, meta) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, role, action, target, ip,
                 json.dumps(meta) if meta is not None else None),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[audit] write failed ({type(exc).__name__}: {exc})")


def recent(limit: int = 100, user_id: str | None = None) -> list[dict]:
    q = "SELECT ts, user_id, role, action, target, ip, meta FROM ms_audit"
    params: list = []
    if user_id:
        q += " WHERE user_id = %s"
        params.append(user_id)
    q += " ORDER BY ts DESC LIMIT %s"
    params.append(limit)
    with pool().connection() as c:
        return c.execute(q, tuple(params)).fetchall()
