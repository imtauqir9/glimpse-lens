"""
src/ingest/flows.py — Prefect flows for document ingest (paper + deck).

This is the per-stage version of the monolithic `ingest_paper` / `ingest_deck`
bodies in paper.py / deck.py. Splitting each stage into its own @task — exactly
like the provided video pipeline (src/ingest/pipeline.py) — buys the real
resilience win: **a failed stage retries WITHOUT redoing the finished ones.**
If embedding hits a rate limit, Prefect retries just `embed-index`; the parsed
pages / chunks from the earlier tasks are reused, not recomputed.

Task results (PageText / Slide / Chunk dataclasses) pass between tasks the same
way the video flow passes `list[Frame]` — in-memory for the default runner,
picklable if you move to a distributed one.

Crash-safety is unchanged from the module bodies:
  • status set to `indexed` ONLY after the Qdrant upsert returns (commit-then-
    complete),
  • `delete_video` clears a dead attempt's partial points before re-upsert,
  • deterministic uuid5 point ids (in vector_store.upsert_chunks) make a full
    re-run idempotent.

Register these two flows in worker.py (see WIRING.md); jobs_documents.py
schedules them by the deployment names "ms-ingest-paper/ingest" and
"ms-ingest-deck/ingest".
"""
from __future__ import annotations

from prefect import flow, task

# Absolute imports (not `from ..`): Prefect loads this entry-point file BY PATH
# when running a flow — relative imports raise "beyond top-level package".
# Absolute `from src...` works because /app is on sys.path.
from src import db
from src.config import TEXT_EMBED_VERSION
from src.rag import vector_store
from src.rag.embeddings import embed_docs
from src.ingest import deck as deck_mod
from src.ingest import paper as paper_mod

# Retry policy mirrors the video flow: the flaky I/O stages (download, embed)
# retry; pure-CPU chunking does not need to.
_FETCH_RETRY = dict(retries=2, retry_delay_seconds=[30, 120])
_EMBED_RETRY = dict(retries=2, retry_delay_seconds=60)


# ── Shared tail: embed → index (identical for paper and deck) ─────────────────
@task(name="embed-index", **_EMBED_RETRY)
def _embed_index(source_id: str, user_id: str, chunks: list) -> int:
    """Embed the chunks and idempotently upsert into the shared TEXT collection.
    Sets `indexed` ONLY after the upsert succeeds. Retrying this task alone (e.g.
    on an embedding rate-limit) does not re-parse or re-chunk."""
    if not chunks:
        # A source that yielded no text is NOT a success. Marking it `indexed`
        # made a scanned PDF or an uncaptioned image-only deck look identical to
        # a fully-retrievable one in /admin/sources, while contributing nothing
        # to retrieval — the failure mode that shows up later as unexplained
        # recall misses. Terminal `failed`, not raised: re-parsing deterministic
        # bytes yields the same nothing, so a Prefect retry would only burn a
        # worker slot (design §20, dead-letter the poison document).
        db.set_status(source_id, "failed", frame_count=0,
                      error="no extractable text — scanned or image-only source; "
                            "captioning is required for it to be searchable")
        return 0
    db.set_status(source_id, "embedding", progress=0.0)
    vectors = embed_docs([c.text for c in chunks])
    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, source_id)     # drop stale points from a prior run
    vector_store.upsert_chunks(
        user_id, source_id, vectors, payloads=[c.payload for c in chunks])
    db.set_status(source_id, "indexed", frame_count=len(chunks),
                  embed_version=TEXT_EMBED_VERSION, progress=1.0)
    return len(chunks)


# ── Paper flow ────────────────────────────────────────────────────────────────
@task(name="paper-parse", **_FETCH_RETRY)
def _paper_parse(source_id: str, uri: str, storage_key: str | None) -> list:
    db.set_status(source_id, "parsing")
    return paper_mod.parse_paper(paper_mod._load_bytes(uri, storage_key))


@task(name="paper-chunk")
def _paper_chunk(source_id: str, title: str, pages: list, user_id: str) -> list:
    db.set_status(source_id, "chunking")
    return paper_mod.chunk_paper(source_id, title, pages, user_id)


@flow(name="ms-ingest-paper", log_prints=True, timeout_seconds=3600)
def ingest_paper_flow(source_id: str, uri: str, title: str, user_id: str,
                      storage_key: str | None = None) -> dict:
    db.bump_attempts(source_id)
    try:
        pages = _paper_parse(source_id, uri, storage_key)
        chunks = _paper_chunk(source_id, title, pages, user_id)
        n = _embed_index(source_id, user_id, chunks)
        print(f"[ingest_paper] {source_id} indexed {n} chunks")
        return {"source_id": source_id, "chunks": n}
    except Exception as exc:  # noqa: BLE001
        db.set_status(source_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise  # Prefect marks the run Failed; visible + retryable in the Cloud UI


# ── Deck flow ─────────────────────────────────────────────────────────────────
@task(name="deck-parse", **_FETCH_RETRY)
def _deck_parse(source_id: str, uri: str, storage_key: str | None,
                filename: str) -> list:
    db.set_status(source_id, "parsing")
    data = paper_mod._load_bytes(uri, storage_key)
    return deck_mod.parse_deck(data, filename or uri)


@task(name="deck-chunk")
def _deck_chunk(source_id: str, title: str, slides: list, user_id: str) -> list:
    db.set_status(source_id, "chunking")
    return deck_mod.chunk_deck(source_id, title, slides, user_id)


@flow(name="ms-ingest-deck", log_prints=True, timeout_seconds=3600)
def ingest_deck_flow(source_id: str, uri: str, title: str, user_id: str,
                     storage_key: str | None = None, filename: str = "") -> dict:
    db.bump_attempts(source_id)
    try:
        slides = _deck_parse(source_id, uri, storage_key, filename or uri)
        chunks = _deck_chunk(source_id, title, slides, user_id)
        n = _embed_index(source_id, user_id, chunks)
        print(f"[ingest_deck] {source_id} indexed {n} chunks")
        return {"source_id": source_id, "chunks": n}
    except Exception as exc:  # noqa: BLE001
        db.set_status(source_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


# ── Serve (register the deployments) ──────────────────────────────────────────
# In worker.py, beside the video deployment. Each served with name="ingest" so
# the deployment strings match jobs_documents.py:
#     ms-ingest-paper/ingest , ms-ingest-deck/ingest
#
#   from prefect import serve
#   from .ingest.flows import ingest_paper_flow, ingest_deck_flow
#   serve(
#       ingest_paper_flow.to_deployment(name="ingest"),
#       ingest_deck_flow.to_deployment(name="ingest"),
#       # ...plus the existing video deployment...
#   )
#
# Note: `caption_image`/captioning for image-only deck slides needs the multimodal
# LLM configured on the WORKER (llm.env_config()); otherwise those slides index on
# their sparse text only (best-effort, never fatal — see deck.py :: _caption).
