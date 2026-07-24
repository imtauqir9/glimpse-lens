"""MomentSearch — unified API (one service, one port).

Two routers on one FastAPI app (:8000):
  - src/api/videos.py  /api/videos/*  — presigned uploads + registration +
                                        ingest status (Bearer auth)
  - src/api/search.py  public         — / (web UI), /api/ask, /api/config,
                                        local-dev media, /api/health

Heavy processing never happens here — the videos router only schedules Prefect
flow runs; worker.py (separate process, same image) executes the ingest
pipeline. Every durable byte lives in object storage, Qdrant, or Postgres, so
this process is stateless and disposable.

Run:
    uvicorn src.app:app --port 8000
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from . import config, db, db_documents, metrics
from .api.admin import router as admin_router
from .api.search import router as search_router
from .api.videos import router as videos_router
from .rag import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    db_documents.add_kind_column()  # idempotent: add `kind` to ms_videos for paper/deck
    # Create the Qdrant collection up front (known CLIP dims resolve without
    # loading the model) so a question before the first ingest returns
    # "no moments" instead of a 500. Qdrant being down must not block boot.
    try:
        vector_store.ensure_collection()          # visual (CLIP frames)
        if config.ENABLE_TRANSCRIPT:
            vector_store.ensure_text_collection()  # transcript (bge text)
    except Exception as exc:
        print(f"[startup] Qdrant not ready ({exc!r}) — search degrades to empty results")
    # Redelivery half of no-loss: re-enqueue sources stuck after a worker crash
    # (idempotent re-runs; design §9). Runs in this always-up process.
    from . import reconciler
    reconciler.start_in_background()
    yield


app = FastAPI(title="Glimpse", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Structured JSON access log (correlation id) + request metrics (§10)."""
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - t0
    p = request.url.path
    # Collapse high-cardinality media paths so the metric labels stay bounded.
    route = "/api/media" if (p.startswith("/api/frame") or
                             (p.startswith("/api/video") and p != "/api/videos")) else p
    metrics.inc("glimpse_requests_total",
                {"route": route, "method": request.method, "status": response.status_code})
    metrics.observe("glimpse_request_duration_seconds", dt, {"route": route})
    print(json.dumps({"request_id": rid, "method": request.method, "path": p,
                      "status": response.status_code, "ms": round(dt * 1000, 1)}))
    response.headers["x-request-id"] = rid
    return response


app.include_router(videos_router)
app.include_router(search_router)
app.include_router(admin_router)  # /admin/documents + /admin/sources (paper/deck)
