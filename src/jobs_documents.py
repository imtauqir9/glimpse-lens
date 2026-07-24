"""
src/jobs_documents.py — Prefect trigger for document ingest (mirrors jobs.py).

The API schedules a run and returns immediately; a worker picks it up. Like
jobs.enqueue_video, this NEVER imports the pipeline or its heavy deps (pymupdf,
python-pptx) — it just asks Prefect Cloud to schedule the right deployment.

The two deployments (`ms-ingest-paper/ingest`, `ms-ingest-deck/ingest`) are
registered by worker.py's flow.serve() — see WIRING.md step 3. Until they exist,
this raises at schedule time and /admin/documents returns 502 (upstream), which
is the correct behavior to surface.
"""
from __future__ import annotations

from prefect.deployments import run_deployment

_DEPLOYMENT = {
    "paper": "ms-ingest-paper/ingest",
    "deck": "ms-ingest-deck/ingest",
}


def enqueue_document(source_id: str, uri: str, title: str, kind: str,
                     user_id: str, storage_key: str | None = None) -> str:
    """Schedule the paper/deck ingest flow for one document. Returns the
    Prefect flow-run id. Parameters match the flow signature in
    src/ingest/paper.py :: ingest_paper_flow (and deck.py)."""
    name = _DEPLOYMENT.get(kind)
    if name is None:
        raise ValueError(f"no ingest deployment for kind={kind!r}")
    flow_run = run_deployment(
        name=name,
        parameters={"source_id": source_id, "uri": uri, "title": title,
                    "user_id": user_id, "storage_key": storage_key},
        timeout=0,  # fire-and-forget: don't block the API on the run
        flow_run_name=f"ingest-{source_id}",
    )
    return str(flow_run.id)
