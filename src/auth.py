"""API-key auth + RBAC — enterprise Phase 1.

Bearer keys look like `glk_...`; only their SHA-256 hash is stored (`ms_api_keys`),
so a database leak never exposes a usable key. Each key maps to a `user_id`
(tenant) and a `role`:
  • admin  — read + write (ingest, delete, mint keys, view metrics)
  • viewer — read-only (search)

The KEY determines the tenant — not the old, spoofable `X-User-Id` header — so
tenant isolation can no longer be bypassed by changing a header.

Auth is ACTIVE when `ADMIN_TOKEN` is set OR at least one key exists; otherwise the
app stays open (dev convenience). `ADMIN_TOKEN` also works as a master admin key,
which is how you bootstrap the first real key.
"""
from __future__ import annotations

import hashlib
import re
import secrets

from fastapi import Header, HTTPException

from .config import ADMIN_TOKEN, DEFAULT_USER_ID
from .db import pool

_ROLE_RANK = {"viewer": 1, "admin": 2}
_USER_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ms_api_keys (
    key_hash    TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'viewer',
    label       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked     BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS ms_api_keys_user_idx ON ms_api_keys (user_id);
"""


def init_schema() -> None:
    with pool().connection() as c:
        c.execute(SCHEMA)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def mint_key(user_id: str, role: str = "viewer", label: str | None = None) -> str:
    """Create a key for a tenant + role. Returns the PLAINTEXT key (shown once)."""
    if role not in _ROLE_RANK:
        raise ValueError("role must be 'admin' or 'viewer'")
    if not _USER_RE.match(user_id or ""):
        raise ValueError("invalid user_id")
    token = "glk_" + secrets.token_urlsafe(32)
    with pool().connection() as c:
        c.execute("INSERT INTO ms_api_keys (key_hash, user_id, role, label) "
                  "VALUES (%s, %s, %s, %s)", (_hash(token), user_id, role, label))
    return token


def list_keys() -> list[dict]:
    with pool().connection() as c:
        return c.execute("SELECT user_id, role, label, created_at, revoked "
                         "FROM ms_api_keys ORDER BY created_at DESC").fetchall()


def revoke_key(token: str) -> None:
    with pool().connection() as c:
        c.execute("UPDATE ms_api_keys SET revoked = true WHERE key_hash = %s", (_hash(token),))


def _keys_exist() -> bool:
    with pool().connection() as c:
        return c.execute("SELECT 1 FROM ms_api_keys WHERE NOT revoked LIMIT 1").fetchone() is not None


def auth_active() -> bool:
    """Auth enforced once a master token or any key exists; else open (dev)."""
    return bool(ADMIN_TOKEN) or _keys_exist()


def _bearer(authorization: str | None, x_api_key: str | None) -> str | None:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return (x_api_key or "").strip() or None


def resolve(token: str | None) -> tuple[str, str] | None:
    """token -> (user_id, role), or None if unknown/revoked."""
    if not token:
        return None
    if ADMIN_TOKEN and token == ADMIN_TOKEN:      # master admin (bootstrap)
        return (DEFAULT_USER_ID, "admin")
    with pool().connection() as c:
        r = c.execute("SELECT user_id, role FROM ms_api_keys "
                      "WHERE key_hash = %s AND NOT revoked", (_hash(token),)).fetchone()
    return (r["user_id"], r["role"]) if r else None


# ── request helpers (called from the route dependencies) ─────────────────────

def resolve_identity(authorization: str | None, x_api_key: str | None,
                     x_user_id: str | None) -> tuple[str, str]:
    """(user_id, role). Fails CLOSED with 401 when auth is active and the key is
    missing/invalid. When auth is inactive (dev), honors X-User-Id as admin."""
    token = _bearer(authorization, x_api_key)
    if not auth_active():
        uid = (x_user_id or DEFAULT_USER_ID).strip()
        if not _USER_RE.match(uid):
            raise HTTPException(400, "Invalid X-User-Id.")
        return (uid, "admin")
    who = resolve(token)
    if who is None:
        raise HTTPException(401, "Missing or invalid API key.",
                            headers={"WWW-Authenticate": "Bearer"})
    return who


def check_admin(authorization: str | None, x_api_key: str | None,
                x_user_id: str | None) -> tuple[str, str]:
    uid, role = resolve_identity(authorization, x_api_key, x_user_id)
    if _ROLE_RANK.get(role, 0) < _ROLE_RANK["admin"]:
        raise HTTPException(403, "Admin role required.")
    return (uid, role)


# ── FastAPI dependencies ─────────────────────────────────────────────────────

def identity_dep(authorization: str | None = Header(default=None),
                 x_api_key: str | None = Header(default=None),
                 x_user_id: str | None = Header(default=None)) -> tuple[str, str]:
    return resolve_identity(authorization, x_api_key, x_user_id)


def admin_dep(authorization: str | None = Header(default=None),
              x_api_key: str | None = Header(default=None),
              x_user_id: str | None = Header(default=None)) -> tuple[str, str]:
    return check_admin(authorization, x_api_key, x_user_id)
