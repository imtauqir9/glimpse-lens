# `paper.py` / `deck.py` — reconciled to the real `momentsearch` repo

The Appendix-B scaffolds in `glimpse-design.md` were written against **guessed**
module names (`# ADAPT`). I cloned the real repo
([traversaal-ai/momentsearch](https://github.com/traversaal-ai/momentsearch))
and rewrote both modules against the **actual** APIs. This file records what
changed and the three things still on you (they live outside these two files).

## What the scaffold guessed vs. what the repo actually has

| Scaffold guess | Reality in the repo | What I used |
|---|---|---|
| `src.rag.chunk.semantic_chunks` | **does not exist** — only time-based `transcript.chunk_cues` | a small word-window splitter (`_window_chunks` in `paper.py`) |
| `src.ingest.enrich.enrich_chunks` | **does not exist** (no separate enrich stage) | dropped; deck captioning is done inline during parse |
| `src.rag.embed.embed_chunks` | `src.rag.embeddings.embed_docs(list[str])` | `embed_docs` (papers/decks are text) |
| `src.rag.index.upsert_chunks(collection=…, chunks=…, vectors=…)` | `vector_store.upsert_chunks(user_id, video_id, vectors, payloads)` — collection is hardcoded to `TEXT_COLLECTION`, point IDs are `uuid5(f"{video_id}:text:{i}")` internally | `vector_store.upsert_chunks(user_id, source_id, vecs, payloads)` |
| `src.storage.fetch_bytes(uri)` | **does not exist** — only `fetch.fetch_upload(key)` + yt-dlp | `_load_bytes(uri, storage_key)`: bucket key → `storage.get_bytes`, else http(s) download |
| `src.db.set_status(id, status, pct=…)` | `db.set_status(id, status, *, frame_count, progress, embed_version, …)` — no `pct` | `progress` (0..1) + `frame_count` for chunk count |
| `src.config.QDRANT_COLLECTION` for text | that's the **image/CLIP** collection; text lives in `TEXT_COLLECTION` | text branch → `ensure_text_collection()` + `TEXT_COLLECTION` |
| `src.llm.caption_image(bytes)` | **does not exist** — `llm.answer(question, moments, cfg)` | `_caption` wraps `llm.answer` with a one-image moment |

## Design decisions baked in (so retrieval + crash-safety actually work)

- **Text branch, not a new collection.** Papers/decks are text, so they upsert
  into the same `TEXT_COLLECTION` the video transcript uses → one query fuses
  video + paper + deck for free. `kind` in the payload distinguishes them.
- **`video_id = source_id` in every payload.** The read path (`src/rag/search.py`)
  fuses and filters on `video_id` and joins metadata via `db.videos_by_ids`.
  Reusing that key means retrieval works unchanged; `kind` + `page`/`slide` ride
  alongside for citation rendering.
- **Idempotent re-runs.** `upsert_chunks` derives deterministic uuid5 IDs from
  `source_id + index`; `delete_video(user_id, source_id)` clears partial points
  from a crashed attempt first. `indexed` is set **only after** the upsert
  returns — this is what makes `bench.py --resilience` pass.

- **`user_id` MUST be in every chunk payload.** `vector_store` filters every
  search on `user_id` (`_user_filter`), and `upsert_chunks` does NOT inject it —
  the caller's payload must carry it (the video transcript payload does). A
  local end-to-end run caught this: without `user_id`, `search_text` returns zero
  hits and documents are invisible to retrieval (recall@10 = 0). `chunk_paper` /
  `chunk_deck` therefore take `user_id` and put it in the payload.

## Still on you (outside these two files)

1. **DB: give documents a home.** `db.py`'s `ms_videos` has a fixed column set
   (no `kind`, `page`, `slide`) and ids are `yt_`/`up_`. Either add a `kind`
   column to `ms_videos` and store docs as rows (`source='paper'|'deck'`, id
   `pp_…`/`dk_…`), or add an `ms_documents` table. These modules call
   `db.set_status`/`db.bump_attempts`, which work on any existing `ms_videos`
   row — so the simplest path is: add `kind`, insert the pending row in
   `/admin/documents`, done.
2. **Admin API** (`src/api/admin.py`): `POST /admin/documents` → validate
   `{uri, kind, title}`, insert a `pending` row, **schedule the Prefect flow**,
   return `202 {id, status:"pending", kind}` (no parsing in the request path).
   `GET /admin/sources` → unified video+doc list with `kind` + `pct` (map
   `progress`→`pct`). Errors: `400` bad input · `401` bad token · `502` upstream.
   Model it on `src/api/videos.py :: register`.
3. **Prefect wiring** (`src/jobs.py` + `src/worker.py`): register
   `ms-ingest-paper` / `ms-ingest-deck` deployments beside `ms-ingest-video`
   (same per-task retries) and add `enqueue_paper`/`enqueue_deck` mirroring
   `enqueue_video`. Flow/task sketch is in each module's footer.
4. **Read-path citations** (`src/rag/search.py`, `ui/`): carry `kind` + the
   locator into the citation payload and render per kind — video → seek
   `start_ms`, paper → open `page`, deck → show `slide`.

## Dependencies to add

```
pip install pymupdf python-pptx      # add to requirements.txt
# optional: LibreOffice (headless) if you rasterize PPTX slides for captioning
```

## Verified here

`parse_paper` + `chunk_paper` run against a real generated PDF: page locators
are preserved, blank/image-only pages are dropped, and the payload contract
(`kind`, `video_id==source_id`, `modality`, `text`, integer `page`) holds. The
`db`/`qdrant`/`embeddings`/`llm` calls are import-reconciled to the real repo but
need the repo installed + services running to exercise end-to-end.
