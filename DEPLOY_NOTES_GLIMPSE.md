# Deploy notes — Glimpse (paper/deck) additions

The base `DEPLOYMENT.md` is complete and **unchanged** — follow it as-is. The
document-ingest extension needs **no new infrastructure and no new secrets**:

- `pymupdf` + `python-pptx` are in `requirements.txt`, so the Docker image builds
  with them automatically (no Dockerfile change).
- `POST /admin/documents` + `GET /admin/sources` ride the existing `api` process;
  the paper/deck Prefect flows ride the existing `worker` process (`src/worker.py`
  now serves all three deployments in one worker).
- The `kind` column migration runs at API startup (`app.py` lifespan,
  idempotent) — nothing to run by hand.
- Deck image-only-slide captioning uses the **same** `LLM_API_KEY` already set for
  answer synthesis. No key → those slides index on sparse text only (never fatal).

## After `fly deploy` — verify the new path

```bash
APP=https://<your-app>.fly.dev
TOKEN=<ADMIN_TOKEN>

# 202 immediately (async accept)
curl -si $APP/admin/documents -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"uri":"https://arxiv.org/pdf/2312.10997","kind":"paper","title":"RAG Survey"}' | head -1

# walks to indexed, shows kind + pct
curl -s $APP/admin/sources -H "X-User-Id: default" | python -m json.tool

# a query cites a paper PAGE
curl -s $APP/api/ask -H 'content-type: application/json' \
  -d '{"question":"what is hybrid retrieval","top_k":10}' | python -m json.tool | grep -A2 '"page"'
```

## Grading hygiene

- Remove the `benchmark/` line from `.gitignore` so `bench.py` is committed.
- Run the SLA gate against the deployed app: `BASE_URL=$APP python benchmark/bench.py`
  (and `--resilience` with `BENCH_KILL_CMD="fly machine stop <worker-id>"`).
- Do **not** commit `.env` (it holds every credential + `FLY_IO_TOKEN`).
