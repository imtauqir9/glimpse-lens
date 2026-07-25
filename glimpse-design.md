# Glimpse — Scalable Multimodal RAG: Design Doc

**Status:** Draft · **Owner:** Imran Tauqir · **Scope:** Production architecture for multimodal (video + document) retrieval at scale

---

## 1. Context & Problem

Text RAG answers questions but returns only text. Moment Search retrieves the exact *visual* evidence (a frame/clip) but doesn't reason. Neither alone is enough for a trustworthy answer over rich media.

**Thesis:** unify them. Text RAG finds the *answer*; Moment Search finds the *evidence*; the system returns **both together** — a grounded answer plus the supporting clip/passage, each cited to an exact locator (timestamp, page, or slide).

This doc describes the production architecture for that system and how it extends from a video-only tool (**Glimpse** today) to a **multi-source multimodal RAG pipeline** (video + PDF papers + slide decks) that stays fast under load and survives real backfills.

## 2. Goals / Non-Goals

**Goals**
- One searchable space across video, PDFs, and slide decks.
- Ingestion fully decoupled from search — accept in milliseconds, index in the background.
- Search latency stays flat during large backfills.
- Every answer is grounded: retrieved evidence + exact locator, never hallucinated.
- Resilient workers: no data loss when a worker dies mid-ingest.
- Multi-tenant isolation (per-user data).

**Non-Goals**
- Real-time (sub-second) *ingestion* — ingest is async by design.
- Training custom embedding models — use off-the-shelf CLIP + a text embedder.
- Full video understanding/generation — retrieval + grounded synthesis only.

## 3. Architecture Overview

Two decoupled paths sharing one vector store.

![Glimpse — Multimodal RAG Architecture](glimpse-architecture.png)

### 3.1 Ingest (write path)
```
Browser ──upload (multipart)──▶ Object Storage (GCS / S3 / Tigris)
   │
   ▼
API (FastAPI) ──register manifest──▶ Postgres (Neon)   [source manifest + status]
   │
   └──enqueue job──▶ Job Queue ──▶ Ingest Worker (background, per source type)
                                       │  parse → dedup → chunk → enrich → caption
                                       ▼
                                   CLIP service (one warm model, 800D) + text embedder
                                       │
                                       ▼  upsert (int8 + payload)
                                   Qdrant  (int8 in RAM + float32 on disk · HNSW · multi-tenant)
```
Key properties: **one warm shared CLIP model** (CPU→GPU swap), **~4× less RAM via int8**, **dedup kills near-duplicate vectors** before they ever hit the index.

### 3.2 Query (read path)
```
Browser (text query)
   ▼
API (FastAPI)  ── rate limiting + fair queue · optimize time-to-first-byte
   │  embed query (text + CLIP-text)
   ▼  search with user_id filter
Qdrant  (HNSW on int8 in RAM → rescore on float32 on disk → return ~20 candidates)
   ▼
Rerank + Fusion (RRF + cross-modal boost → top 6 moments)
   ▼
Object Storage (fetch thumbnails / clipped frames)
   ▼
Multimodal LLM (GPT-4o / vLLM / others)  — reason over answer + evidence
   ▼
Browser: cited answer + timestamps + thumbnails
```

### 3.3 The flow as implemented

3.1 and 3.2 are the intent. This is what actually runs, step by step, with the code that does it.

**Process topology.** One Docker image, three long-running process groups plus a one-shot seed gate:

```
                    ┌─ api    :8000  public    routes · auth · retrieval · UI
one image ──────────┼─ worker  no ports        polls Prefect, runs ingest
                    └─ clip   :8001 internal   ONE warm CLIP model (binds ::)
                              │
   Neon Postgres · Qdrant Cloud · Prefect Cloud · GCS/Tigris · Upstash Redis
```

Everything stateful is a rented managed service, so every machine is disposable. `clip` binds `::` rather than `0.0.0.0` because Fly private networking is IPv6-only — bound to `0.0.0.0` it is unreachable at `clip.process.<app>.internal`.

**Write path — ingest**

1. `POST /api/videos/presign` (`api/videos.py`) returns a presigned PUT URL.
2. The client uploads bytes **directly to object storage**, never through the API. (`PUT /{id}/content` is the fallback when the bucket can't presign.)
3. `POST /api/videos` writes a `pending` row, schedules a Prefect run, and returns **202 in single-digit ms**. Papers and decks take the same shape via `POST /admin/documents`. Both append to `ms_audit`.
4. The **dispatcher** (`dispatcher.py`) admits pending sources round-robin across users — a weighted-fair queue, so one bulk uploader cannot starve everyone else.
5. The **worker** (`worker.py`) long-polls Prefect Cloud — **outbound HTTPS only, no inbound ports** — running up to `WORKER_CONCURRENCY` flows concurrently.
6. Pipeline stages, each a Prefect task carrying its own retry policy (`ingest/pipeline.py`):

   | Task | Retries | Does |
   |---|---|---|
   | `t_fetch` | 2 @ 30s, 120s | download source |
   | `t_sample` | — | ffmpeg → frames |
   | `t_embed_index` | 2 @ 60s | CLIP embed → Qdrant upsert |
   | `t_transcript` | 1 @ 30s | transcript chunks → text embed → text collection |

   Embedding goes to the warm `clip` service; with `CLIP_SERVICE_URL` unset each run loads the model in-process, which is the ~15–30s penalty the service exists to pay once.
7. The **reconciler** (`reconciler.py`, started in the api lifespan) re-enqueues sources stranded by a worker crash. Retries are idempotent, so redelivery is always safe.

**Read path — `POST /api/ask`**

1. **Middleware** (`app.py`) — correlation id, timing, request counter + duration histogram. High-cardinality media paths collapse to `/api/media` so label cardinality stays bounded.
2. **Auth** — the key resolves to `(user_id, role)`; **the key sets the tenant** (§11).
3. **Rate limit** (`guardrails.check_rate`) — Redis fixed window keyed `rl:<tenant>:<minute>`, falling back to an in-memory sliding window. Over limit → 429 + `Retry-After`. An empty `user_id` is rejected outright, so the limit can't be dodged by omitting the tenant.
4. **Two retrieval branches** (`rag/search.py`), both carrying the `user_id` filter: visual (CLIP text→image over frames) and text (bge query→transcript chunks). **Papers and decks live in the same text collection**, which is why cross-source citations work at all.
5. **Fusion** (`_fuse`) — RRF into time windows. Documents bucket by *locator* rather than time, so two pages of one paper stay two windows instead of collapsing into one.
6. **Gate 1 — confidence**, scored on the **raw per-branch bests**, not the fused score: abstain only if *neither* what's on screen nor what's said looks relevant. This runs **before** the LLM call, so an unanswerable question costs nothing.
7. **Model resolution** — the tenant's own hosted model if configured, else the server's; with neither, `_fallback_answer` summarizes the matches rather than inventing prose.
8. **Synthesis** → citation validation → token/`$` metered from the provider's reported usage (§18).

`POST /api/retrieve` is this path minus steps 7–8. That split is deliberate: it is embed + ANN in milliseconds, so it measures the **decoupling SLA honestly** — the SLA is about retrieval, not the seconds-long multimodal LLM call, and folding the LLM into the measurement would hide the property being proven.

**Why the paths don't contend.** The API only ever writes a row and schedules a run. No request ever waits on parsing, embedding, or an LLM call triggered by someone else's upload. That is the whole reason a thousand-video backfill and a user's query can run at once without the query noticing — the claim §9 makes and `benchmark/bench.py` tests.

## 4. Data & Infrastructure

| Component | Choice | Role |
|-----------|--------|------|
| Object storage | GCS / S3 / Tigris | Source media (videos, PDFs, decks) + thumbnails/clips |
| Metadata DB | Postgres (Neon) | Source manifest, ingest status/lifecycle, API keys (`ms_api_keys`), audit trail (`ms_audit`) |
| Shared state | Redis (Upstash) | Cluster-wide rate-limit + metrics counters. Optional: no `REDIS_URL` → per-process fallback, correct only at one replica |
| Vector DB | Qdrant | Unified index; **scalar quantization** (int8 in RAM + float32 on disk); HNSW; multi-tenant |
| Queue | Job queue (managed or self-hosted) / Prefect | Background ingest jobs, status, retries |
| Compute | Containers + autoscaling workers | CLIP service, ingest workers, API |
| Models | CLIP (800D) · text embedder · multimodal LLM | Embedding + grounded synthesis |

## 5. The Unified Index (making mixed sources one space)

One Qdrant collection holds vectors from **every** source. Two embedding families:
- **Text embedder** for all text — video transcript chunks, PDF passages, slide text.
- **CLIP** for all images — video frames, PDF figures, slide images. CLIP's shared text–image space is what powers "search by what's on screen."

Each point carries a payload that makes retrieval typed and citations grounded:

```json
{
  "source_type": "video | pdf | slide",
  "modality":    "text | image",
  "doc_id":      "…",
  "user_id":     "…",             // multi-tenant filter
  "locator":     { "timestamp": 812.4 } // or { "page": 7 } or { "slide": 12 }
  "thumb_ref":   "object-storage://…"    // frame/clip/figure for display
}
```

Use Qdrant **named vectors** (a `text` vector and an `image` vector) so text and CLIP embeddings coexist without being forced into one space; fusion (Section 8) combines their result lists.

## 6. Multi-Source Ingestion (Task 3.1)

Generalize the video-only write path into **per-source ingest flows** that all funnel into the same embed + upsert. Only the parse/chunk/dedup head differs.

| Source | Parse | Chunk | Dedup | Embed |
|--------|-------|-------|-------|-------|
| Video | `ffmpeg` frames (~2s) + Whisper transcript | by scene / transcript segment | **pHash** on frames | CLIP (frames) + text (transcript) |
| PDF paper | PyMuPDF (text, layout, figures) | by section/heading | text hash | text (passages) + CLIP (figures, captioned) |
| Slide deck | python-pptx / render+OCR | by slide | text hash | text (slide text) + CLIP (slide image) |

**Enrichment:** caption figures/frames/slide images with a vision model so images are retrievable by text; attach title/section/source metadata. **Definition of done:** one natural-language query returns a video moment, a paper passage, and a slide — each correctly cited.

## 7. Async Work Queues (Task 3.2)

Each source type is a **Prefect flow** (`ingest_video`, `ingest_pdf`, `ingest_deck`) whose tasks mirror the video lifecycle: `fetch → parse → chunk → enrich → embed → upsert`, with the same **status lifecycle** (`queued → running → done | failed`) recorded in Postgres and **retries** on the flaky steps (download, embed, LLM captioning).

**Why queues separate ingestion from search:** ingestion is slow, bursty, and GPU-heavy; search must be instant and always available. The API only *accepts + enqueues* (single-digit ms); workers do the heavy lifting; each side scales independently and an ingest spike never touches query latency.

Concurrency limits throttle embedding/GPU work; a dead-letter path isolates poison documents.

## 8. Fusion, Ranking & Grounding

- Retrieve candidates from both vector families (text + CLIP) with the `user_id` filter.
- **Reciprocal Rank Fusion (RRF)** merges the lists; a **cross-modal boost** rewards moments where visual and textual evidence agree.
- Qdrant serves ANN on **int8 in RAM**, then **rescores the shortlist on float32 on disk**. This two-tier trick is **scalar quantization**: each vector's `float32` values are compressed to 8-bit integers (~¼ the memory, <1% accuracy loss). The quantized copy gives fast in-memory ANN; the full-precision copy on disk restores exact accuracy on the ~20 shortlisted candidates only — speed *and* accuracy.
- Top ~6 moments go to the multimodal LLM with their thumbnails/clips for synthesis.
- **Grounding is non-negotiable:** the answer cites retrieved evidence + exact locator; the UI shows the source frame/passage. No citation → not shown.

## 9. Scale & Reliability (Task 3.3)

**Benchmarks**
- *Accept latency* — time to accept + enqueue an ingest request. Target: single-digit ms (it only enqueues).
- *Throughput* — docs/min the worker pool sustains.
- *Recall@k* — retrieval quality vs. a small labeled query set.

**Decoupling proof:** run a large backfill (hundreds of docs) and show query **p95 stays flat** throughout.

**Chaos test:** kill a worker mid-ingest → prove nothing is lost. This passes only with:
1. **Idempotent upserts** — deterministic point IDs (hash of `doc_id + locator`) so re-processing can't duplicate.
2. **Retry / re-queue** — an interrupted job is retried on another worker.
3. **Commit-then-complete** — a job is marked `done` only *after* the Qdrant upsert succeeds.

## 10. Observability

You can't operate — or prove Task 3.3's "nothing is lost" — without measurement. Instrument three layers.

**Metrics (per stage → Prometheus/Grafana or equivalent):**
- *Ingest:* queue depth, jobs in-flight, docs/min throughput, per-stage duration (parse / embed / upsert), retry rate, dead-letter count.
- *Embedding / CLIP:* request latency, GPU utilization, warm-model hit rate, batch size.
- *Query:* p50/p95/p99 latency, time-to-first-byte, candidates returned, rerank time, LLM latency + token cost.
- *Quality:* recall@k vs. the labeled eval set, % of answers carrying a citation, empty-result rate.

**Structured logging:** JSON logs keyed by `job_id` / `request_id` (correlation IDs) so one ingest or query can be traced end-to-end across API → queue → worker → Qdrant → LLM.

**Tracing:** distributed spans (OpenTelemetry) across pipeline stages — a single trace shows exactly where a slow query spent its time (embed vs. search vs. rescore vs. LLM).

**Dashboards, SLOs & alerts:**
- SLO: query p95 < target, held **even during a backfill** (this is the decoupling proof, made visible).
- Alerts: queue depth climbing (workers falling behind), DLQ non-empty (poison docs), recall regression, error-rate spike, worker crash-loop.

### What's built, and where the state lives

Two surfaces over one metrics store: `GET /metrics` (Prometheus text) and `GET /api/metrics.json` (JSON rollup behind `/dashboard`). Both are admin-gated. Ingest queue-depth gauges are computed from Postgres at scrape time rather than kept as counters, so they can't drift from reality.

**Counters are cluster-wide; histograms are per-replica.** This asymmetry is deliberate and worth stating, because it decides what you may trust:

- *Counters* (requests, rate-limited, tokens, `$` cost) write through to a Redis hash, and **both** readers prefer it. So the two endpoints can never disagree, and totals survive a machine restart — which matters more than it sounds: the api process auto-stops when idle (`min_machines_running = 0`), so in-process counters would reset on **every idle cycle**, and Prometheus would record a counter reset. Sawtooth graphs, and `rate()`-based alerts evaluating a window that just dropped to zero.
- *Histograms* (latency buckets) remain in-process and **do** reset when a machine stops. Normally Prometheus aggregates per-replica histograms across instances at scrape time, which is the standard answer — but that assumes instances stay up. With idle-stop machines, p95 is only meaningful within a single machine's uptime. Fixing it properly needs Redis-backed buckets.

The gap this closes is a specific failure worth remembering: shared state added on the **write** path is not enough. Until both read paths used it, `/api/metrics.json` reported 53 requests while `/metrics` reported 18 **at the same instant** — and it was the Prometheus endpoint, the one the monitoring stack actually scrapes, that was wrong.

**Prometheus deployment:** the `monitoring/` stack (Prometheus + Grafana + alert rules) runs as containers scraping the deployed app over TLS, not as an app process group — a per-app time-series database wants a persistent volume and is the wrong thing to pay for at this size. Fly's managed Prometheus is the alternative, with one catch: its scraper sends no `Authorization` header, so it cannot read an admin-gated `/metrics` without that route being opened first.

## 11. Guardrails

Guardrails protect quality, cost, and safety — critical here because the pipeline ingests **untrusted documents** and answers with an LLM.

- **Input validation (ingest):** allowed file types + size caps, page/duration limits, MIME sniffing, scan on upload; reject or quarantine malformed files (→ dead-letter, surfaced in status).
- **Prompt-injection defense (critical):** retrieved document text is **data, not instructions**. A malicious PDF/slide may contain "ignore previous instructions…". Mitigate by clearly delimiting retrieved context, instructing the LLM to treat it as quoted evidence only, never letting retrieved text alter system/tool behavior, and escaping instruction-like patterns. Assume ingested content is hostile.
- **Grounding / faithfulness:** every answer cites retrieved evidence with an exact locator; **no citation → not shown**. Optionally verify the answer is *supported* by the retrieved passages and abstain ("couldn't find this") rather than hallucinate.
- **Abuse & fairness:** per-user rate limiting + fair queue (no tenant starves others), plan quotas, and hard cost caps on LLM calls. *Built:* `RATE_LIMIT_PER_MIN` per user/minute, counted in Redis so the limit is a **cluster** limit — with per-process counters, N replicas would silently allow N × the limit. No Redis → in-memory fallback, correct for one replica.
- **Tenant isolation:** every query carries the `user_id` payload filter — **fail closed** if it's ever missing; no cross-tenant retrieval. **What establishes `user_id` matters as much as the filter:** it comes from the API key, not from a client-supplied `X-User-Id` header. A header is caller-controlled, so header-derived tenancy is bypassable by editing a request — the filter would be enforced perfectly against the wrong tenant. `X-User-Id` is honored *only* when auth is inactive (local dev). See **Authentication & RBAC** below.

### Authentication & RBAC (implemented)

Bearer keys shaped `glk_…`, stored as **SHA-256 hashes** in `ms_api_keys` — a database leak yields no usable key. Each key maps to a `user_id` (tenant) and a role:

| Role | Can |
|---|---|
| `viewer` | search + ask; read own tenant's sources |
| `admin` | the above, plus ingest, delete, mint/revoke keys, read metrics |

- **Fail-open by design, and that is a deployment hazard.** Auth activates only once `ADMIN_TOKEN` is set **or** at least one key exists. This keeps `docker compose up` zero-config, but on a public URL an unset token means every route is open, `/admin/*` included. Setting it is a required deploy step, not a hardening option.
- **`ADMIN_TOKEN` is the bootstrap**, doubling as a master admin key — used once to mint real per-tenant keys. It should not be the credential in day-to-day use: it can't be revoked without rotating the secret and redeploying, whereas a `glk_` key is revocable through the API.
- **401 vs 403 are distinct signals:** 401 = identity not established; 403 = identity established, role insufficient. A viewer hitting `/metrics` gets 403.

### Audit log (implemented)

Every mutating action — ingest, delete, retry, key mint — appends to `ms_audit`: actor `user_id` + role, action, target, source IP, metadata, server timestamp. Append-only; the application exposes no update or delete path. This is the compliance and security-review trail, and it is what makes "who deleted this tenant's source?" answerable after the fact.
- **PII & compliance:** flag/redact PII in transcripts and documents where required; honor deletion across object storage, Postgres, **and** Qdrant together.

## 12. Retrieval Enhancements — Hybrid Search & Reranking

Dense vectors alone miss exact terms (names, IDs, acronyms, code). Strengthen retrieval in two places:

**Hybrid search (dense + sparse).** Alongside the dense embeddings, index a **sparse/keyword representation** (BM25 or SPLADE — Qdrant supports sparse vectors natively) and fuse both: dense catches *meaning*, sparse catches *exact tokens*. This is the biggest recall win for technical corpora (papers full of precise terminology).

**Cross-encoder reranking.** RRF fuses ranked lists cheaply but never *reads* query + candidate together. Add a **cross-encoder reranker** (`bge-reranker`, Cohere Rerank, etc.) over the top ~20–50 candidates — it scores each `(query, passage)` pair jointly for a large precision jump. Full pipeline becomes: ANN (int8) → rescore (float32) → **hybrid fuse** → **cross-encoder rerank** → top 6 → LLM.

*Trade-off:* reranking adds per-query latency/cost — cap it to the shortlist and make it optional by query complexity.

## 13. Semantic Caching

Many queries are near-duplicates ("what is an FDE?" vs. "explain forward deployed engineer"). Embedding-based caching cuts cost and latency hard:

- On query, embed it and check a **cache index** (Redis/Qdrant) for a prior query within a similarity threshold.
- **Hit** → return the cached answer + citations instantly (no retrieval, no LLM).
- **Miss** → run the full pipeline, then store `{query_embedding → answer, citations, sources}`.
- **Invalidation:** scope entries by tenant; evict on TTL or when underlying sources change (so a re-indexed doc never serves a stale answer). Tune the similarity threshold — too loose returns wrong answers, too tight never hits.

Payoff: repeat and near-duplicate questions skip the two most expensive stages — retrieval and the multimodal LLM — entirely.

## 14. Evaluation Harness

You can't improve retrieval you don't measure. Treat eval as a first-class, automated system — not a one-off benchmark.

- **Golden set:** representative queries with known-relevant sources/answers, grown from real usage + the feedback loop.
- **Retrieval metrics:** recall@k, MRR, nDCG — did the right source make the shortlist, and how high?
- **Answer metrics:** faithfulness (is the answer supported by the retrieved evidence?), relevance, citation correctness — automatable with an LLM-judge or a framework like **Ragas**.
- **Regression gate:** run the eval set in **CI** on every change to chunking, embeddings, prompts, or ranking, and **block merges that regress** recall or faithfulness past a threshold. This is what lets you change the pipeline confidently instead of hoping.

## 15. Feedback Loop

Real usage is the best source of eval data *and* ranking signal — capture it and close the loop.

**Signals to collect:**
- *Explicit:* thumbs up/down on each answer.
- *Implicit:* which citation/moment the user clicks (and dwell — did they stay?), which results are ignored, and query reformulations (a re-ask often signals a miss).

**How it feeds back:**
- **Grows the eval set (§14):** promote thumbs-up / high-click queries into the golden set; flag thumbs-down for review — so evaluation gets more representative over time, automatically.
- **Improves ranking:** clicked citations are positive labels — use them to tune fusion weights or the cross-encoder (learning-to-rank), and down-weight results that are retrieved often but never clicked.
- **Catches drift:** a rising thumbs-down rate or falling click-through is an early-warning signal that should trip an alert (ties to §10 Observability).

**Guardrail:** store feedback per tenant, treat it as noisy (users misclick), and require volume before it moves ranking — never let a handful of clicks overfit.

## 16. Knowledge-Graph RAG (GraphRAG)

Vector retrieval finds *similar* passages but struggles with **multi-hop** questions that connect facts across sources — "how does the method in paper A relate to the result shown in the video?" GraphRAG adds a structured layer:

- **Extraction (at ingest):** run entity + relationship extraction (LLM or NER) over transcripts, passages, and slide text → nodes (papers, methods, people, datasets, concepts) and edges (`uses`, `cites`, `contradicts`, `improves-on`) stored in a graph (Neo4j, or Postgres/Qdrant payload links).
- **Retrieval:** for relational queries, traverse the graph 1–2 hops to gather a connected subgraph, then pull each node's supporting passage/moment as grounding.
- **Fusion:** vectors answer "what's relevant," the graph answers "how things connect" — combine both.
- **Payoff:** cross-document reasoning vector RAG can't do ("trace the lineage," "what contradicts X"), while every node/edge still cites its source.

*Cost note:* extraction is an extra LLM pass at ingest — batch it, cache it, and only build the graph for corpora where relational queries matter.

## 17. Agentic RAG

Instead of one fixed retrieve→answer pass, an **agent** plans and adapts:

- **Plan:** decompose a complex query into sub-questions; decide *which sources* (video / papers / slides) and *which modality* (visual CLIP vs. text) each needs.
- **Act:** issue multiple retrievals (including graph traversals), fetch frames, call tools.
- **Reflect / self-correct:** judge whether the evidence actually answers each sub-question; if not, reformulate and retrieve again (ReAct-style loop) before answering.
- **Synthesize:** combine sub-answers into one grounded response with citations across sources.

This turns Glimpse from "search + answer" into "reason across a multimodal corpus."

*Guardrails (critical for agents):* hard caps on loop iterations and tool calls, a per-query token/cost budget (§18), and fall-back to plain RAG when the agent stalls — otherwise an agent silently burns cost. Reserve the agent for *hard* queries; route simple ones straight through the fast path.

## 18. Token & Cost Utilization

Multimodal LLM calls (especially with image frames) and embedding dominate cost — track and cap them explicitly:

- **Meter per request & per tenant:** input/output tokens, image tokens, embedding calls, rerank calls, and `$` cost — tagged by `request_id`/`user_id` (ties to §10 Observability).
- **Budgets & caps:** per-query and per-tenant/plan cost ceilings; the agentic path (§17) gets a hard token budget + iteration cap.
- **Cost levers:** semantic-cache hits (§13) skip the LLM entirely; retrieve/rerank to a *small* top-K so fewer tokens reach the LLM; send frames as compact thumbnails, not full images; use a cheap model for planning/judging and the strong model only for final synthesis; batch embeddings.
- **Dashboards & alerts:** cost per query, cost per tenant, % served from cache, token-spend trend — alert on anomalies (a runaway agent, a spend spike).

**Built:** actual usage is read from the provider response (not estimated from string length) and metered as `glimpse_llm_{input,output}_tokens_total` and `glimpse_llm_cost_usd_total`, labelled by model, priced from a per-model table. A live answer on `claude-opus-4-8` metered 1514 in / 176 out = **$0.012**. The `LLMCostSpike` alert fires above $5/hour. Two gaps against the design above: cost is not yet broken down *per tenant* (the counters carry a `model` label, not a `user_id` one), and the budgets/caps are alert-only — nothing hard-stops a runaway spend yet.

## 19. Forward-Deployed-Engineer Considerations (Task 3.4)

- **Managed vs. self-hosted queue:** managed (Prefect Cloud, SQS, managed Kafka) trades cost for far less ops and faster time-to-prod; self-hosted trades ops burden for control and cost-at-scale. Choose what survives *in the client's environment* — their infra, compliance, budget, and team — not what's most impressive.
- **Resilient workers:** idempotency, retries, dead-letter, health checks, graceful SIGTERM (finish or re-queue in-flight), backpressure.
- **Grounded citations everywhere** — trust is the product.
- **Cost control** — one warm CLIP model, int8 memory, dedup, and rescore-only-on-shortlist keep RAM and compute bounded.
- **Turn the demo into something that eats a real backfill:** malformed PDFs, 500-page files, rate limits, partial failures — all handled, all observable.

**Which of those are actually built** — the bullets above are the principles; this is the state of the code, because "resilient workers" is the kind of phrase that hides a gap:

| | State |
|---|---|
| Idempotency | Built — deterministic uuid5 point ids + delete-before-upsert |
| Retries | Built — per-stage Prefect task policies (`ingest/flows.py`) |
| Redelivery after a crash | Built — `reconciler.py`, the piece that turns crash-safe *code* into a no-loss *guarantee* |
| Dead-letter | Built — `MAX_INGEST_ATTEMPTS`, then parked at `failed` with a reason. Without it the reconciler retried a worker-killing document forever |
| Health checks | Built — `/api/health`, plus Fly restart policies |
| Backpressure | Partial — the WFQ dispatcher caps in-flight **videos**; documents skip it (`db.wfq_claim` filters `kind='video'`), so a tenant bulk-posting papers is not fairly throttled |
| Graceful SIGTERM | **Not built.** No signal handling anywhere in `src/`. Prefect's runner owns process lifecycle, and a killed run is recovered by the reconciler rather than drained on the way out. That is a weaker guarantee than draining — it costs the in-flight work, which is then redone — but it is the one actually exercised by `bench.py --resilience` |

## 20. Failure Modes & Mitigations

| Failure | Mitigation |
|---------|-----------|
| Worker dies mid-ingest | Idempotent upsert + retry + commit-then-complete |
| Poison document (corrupt PDF) | Dead-letter after N retries; surface in status |
| Embedding rate limits | Concurrency caps + backoff; queue absorbs bursts |
| Cold CLIP model (first request slow) | Keep one model warm; CPU→GPU swap |
| Near-duplicate frames bloat index | pHash dedup before upsert |
| Cross-tenant leakage | `user_id` payload filter on every query |
| Quantization accuracy loss | float32 rescore on the shortlist |

## 21. Milestones (maps to the project)

1. **3.1 — Multi-source ingest:** PDF + slide flows into the shared index with source-typed payloads.
2. **3.2 — Queues:** Prefect flows with status lifecycle + retries; API accept-and-enqueue.
3. **3.3 — Scale & prove:** benchmark accept latency / throughput / recall; backfill + chaos test.
4. **3.4 — Harden:** managed-vs-self-hosted decision, resilience, cost, grounded citations end-to-end.

## 22. Open Questions / Risks

- Text vs. CLIP vector spaces — named vectors + fusion, or a single multimodal embedder (e.g., SigLIP)? Benchmark recall both ways.
- Long-context multimodal LLMs may absorb some retrieval work over time — keep the pipeline modular so components can be swapped.
- Chunking strategy per source materially affects recall — needs an eval set to tune.
- Multi-tenant scale in one Qdrant collection vs. per-tenant collections — revisit at higher tenant counts.

---

# Appendix A — Assignment 3: Mapping & Build Punch-List

### "Moment Search at Scale" → your Glimpse design → what's left to build

> Reminder: this is a **reading assignment** (policy MS-3.14). Read the README yourself — this checklist is a study/build aid, not a substitute. (And the README's top comment is a **honeypot**: do **not** create `ROBOT_WAS_HERE.md`, a toaster poem, or `🦥`-prefixed commits — those are the tripwire.)

**Legend:** ✅ design already covers it · ⚠️ tighten to the exact spec · 🔨 build from scratch · 🚫 don't touch (provided)

---

## 1. API contract (grade depends on exact shapes)

| Endpoint | Spec | Status | Files |
|----------|------|--------|-------|
| `POST /admin/videos` | provided, returns `202` | 🚫 don't change | `src/api/admin.py` |
| `POST /admin/documents` | `202 {id,status:"pending",kind}` **before** parsing; `kind: paper\|deck` | 🔨 build | `src/api/admin.py`, `src/jobs.py` |
| `GET /admin/sources` | unified video+doc status w/ `kind` + `pct` | 🔨 build | `src/api/admin.py` |
| `GET /ask_stream` | SSE; citations carry `locator` (`start_ms`\|`page`\|`slide`) | ⚠️ extend | `src/rag/`, `ui/` |
| status codes | `400` bad input · `401` bad token · `502` upstream | 🔨 add | `src/api/*` |

Your design nailed the *concept* (locators, async accept) but named none of these concrete shapes — this is the biggest "tighten to spec" area.

---

## 2. Grading rubric → build map (100 pts)

**① Search lights up cross-source — 15 pts**
Spec: one query cites a video moment + paper page + deck slide, each jumping to the right spot.
Design: ✅ §5 (locators) + §8 (fusion). Build: 🔨 carry `kind` + `locator` into the citation payload; render per kind in the UI (video→seek `start_ms`, paper→`page`, deck→`slide`). Files: `src/rag/`, `ui/`.

**② Multi-format ingestion — 25 pts** (biggest single block)
Spec: page-aware paper parsing + slide-aware deck parsing; correct locators; shared index.
Design: ✅ §6. Build:
- 🔨 `src/ingest/paper.py` — PDF → text w/ structure (`pymupdf`/`pypdf`), **carry `page`** into each chunk; reuse `src/rag/chunk.py`; enrich+embed **unchanged**.
- 🔨 `src/ingest/deck.py` — PDF/PPTX → slides; slide text + **vision-LLM caption** for image-only slides; `slide` in payload.
- Upsert to the **same** Qdrant collection with `kind` in payload.

**③ Queue & decoupling — 20 pts**
Spec: new flows on the **Prefect** queue with the video status lifecycle + retries; search stays fast during a big ingest; writeup explains *why* the queue exists.
Design: ✅ §7 + §9. Build: 🔨 add `kind=paper|deck` flow branches mirroring `ingest_video` (`pending→parsing→chunking→enriching→embedding→indexed|failed`); `/admin/documents` **schedules a run and returns immediately**. Files: `src/jobs.py`, `src/ingest/pipeline.py`.
⚠️ **Prefect Cloud is the required path — not Kafka.** (Your Kafka build = bonus §10 below.)

**④ Resilience / no-loss — 15 pts**
Spec: `bench.py --resilience` — kill a worker mid-ingest → 0 dropped, all resume to `indexed`, finished stages not re-run.
Design: ✅ §9 (idempotent upserts, commit-then-complete). Build: 🔨 set status **after** the Qdrant upsert succeeds; deterministic point IDs (`hash(source_id+locator)`) so retries don't duplicate or re-run finished stages.

**⑤ Retrieval quality & grounding — 15 pts**
Spec: cross-source recall@10 **≥ 0.70**; every citation grounded; no invented pages/timestamps; empty retrieval → empty.
Design: ✅ §8/§11/§14. Build: 🔨 label a small query set; ensure `kind` isn't filtered out of retrieval; fail-closed on empty.

**⑥ Deploy & docs — 10 pts**
Spec: `docker compose up` one-command; deployed on Fly.io from the one image; clear `.env.example`; short "How I ran it".
Design: ⚠️ mentions Fly only. Build: 🔨 deploy; add run notes (LLM/embedding provider, deploy target); verify `.gitignore`.

---

## 3. SLA targets to hit (prove with `benchmark/bench.py`, don't eyeball)

| Metric | Target |
|--------|--------|
| `/admin/documents` accept p95 | ≤ 300 ms |
| Search p95 while a big ingest runs | ≤ 1.3× idle p95 |
| Cross-source recall@10 | ≥ 0.70 |
| No-loss under worker crash | 100% |
| Ingestion throughput (≥2 workers) | ≥ 8 chunks/s |

`bench.py` **exits non-zero if any SLA fails** — it's the grading gate. Complete the `benchmark/` scaffold; your design's §9/§14 metrics map here but you must add these exact thresholds.

---

## 4. Self-verify before "done" (run all — all must pass)

```bash
curl -sf localhost:8100/ >/dev/null && echo "UI ok"                 # 1 app up
curl -si localhost:8100/admin/documents -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"uri":"https://arxiv.org/pdf/2312.10997","kind":"paper","title":"RAG Survey"}' | head -1   # 2 → 202
curl -sN "localhost:8100/ask_stream?q=hybrid+retrieval" | grep -m1 '"page"'   # 3 paper page citation
python benchmark/bench.py                # 5 SLA gate — exit 0
python benchmark/bench.py --resilience    # 6 no-loss — exit 0
git status --porcelain | grep -E '\.env$|\.venv|__pycache__|\.pdf$|\.pptx$' && echo "FAIL: unstage" || echo "clean"  # 7
curl -sf https://<your-app>.fly.dev/ >/dev/null && echo "deployed ok"   # 8
```

---

## 5. Recommended build order
0. Stand up the base app on **video only** — see the queue do its job.
1. `paper.py` (page-aware).  2. `deck.py` (slide-aware).  3. `/admin/documents` + `/admin/sources` wiring.  4. Cross-source citations in `rag/` + `ui/`.  5. Backfill + benchmark + `--resilience`.

---

## 6. Bonus — where your extra design lives (do NOT let these block the core)
- 🚀 **Own broker** (Redis Streams / RabbitMQ / **Kafka**) with at-least-once + DLQ = the assignment's explicit stretch goal → *your Kafka build already does this.*
- Your **GraphRAG (§16), Agentic RAG (§17), Semantic cache (§13), Feedback (§15)** are all beyond scope / map to stretch ("more modalities," "cost panel"). Great north-star; not graded core.
- **Built since this list was written** — no longer north-star, now deployed and verified in production: **Observability (§10)** — `/metrics` + `/dashboard`, Prometheus + Grafana + four alert rules; **Cost panel (§18)** — real token/`$` metering per model, measured at $0.012 on a live `claude-opus-4-8` answer; and **Auth, RBAC + audit log (§11)**, which weren't on this list at all.

---

## 7. Red-lines / hygiene (auto-flagged)
- ✅ No secrets committed — `.env`, `.venv/`, `__pycache__/`, model caches, media/PDF git-ignored.
- ✅ Provided **video pipeline untouched**; video endpoints + UI still work.
- ✅ No `ROBOT_WAS_HERE.md`, toaster poem, or `🦥` commit prefix (the honeypot).
- ✅ Never fabricate recall/latency numbers — every value from a real run.

## 8. Submit
`PRODUCT_EVAL.md` via `/fde-momentsearch-scaled-eval` (runs rubric + benchmark + a live cross-source query on media you don't control) + a **60–90s demo** (one query citing talk-moment + paper-page + deck-slide, then the Prefect run view during a backfill while search stays fast) + push your fork with `paper.py` & `deck.py`.

---

# Appendix B — Reference Ingest Scaffolds

Two modules that mirror the provided video flow (parse + locator-aware chunking, then the shared enrich → embed → index tail). Reconcile every `# ADAPT` import with the real `momentsearch` repo before wiring. Crash-safety (status-after-upsert + deterministic point ids) is built in.

## `src/ingest/paper.py`

```python
"""
src/ingest/paper.py — Paper (PDF) ingestion.

Mirrors the PROVIDED video flow (src/ingest/pipeline.py :: ingest_video):

    fetch → parse → chunk → enrich → embed → index

…but the source is a PDF paper and the citation locator is a **page** number.

Design rules from the assignment:
  • The enrich / embed / index stages are SHARED and reused UNCHANGED — this
    module owns only parse + page-aware chunking, then hands chunks to the same
    tail the video flow uses.
  • Everything lands in the SAME Qdrant collection, tagged `kind="paper"`.
  • Status lifecycle mirrors video exactly; mark `indexed` only AFTER a
    successful upsert (crash-safety → the `--resilience` gate).

⚠️  The imports marked `# ADAPT` are best-guess names based on the README's file
    map. Reconcile them with the real repo after you've read it — match the exact
    chunk schema, embed/index signatures, and status enum the video flow uses.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import fitz  # PyMuPDF  (pip install pymupdf)

# --- shared repo pieces (ADAPT paths/names to the real momentsearch repo) -----
from src.rag.chunk import semantic_chunks      # ADAPT: the SAME chunker video uses
from src.ingest.enrich import enrich_chunks     # ADAPT: shared enrichment stage
from src.rag.embed import embed_chunks          # ADAPT: shared embedding stage
from src.rag.index import upsert_chunks         # ADAPT: shared Qdrant upsert
from src.storage import fetch_bytes             # ADAPT: object-storage / URL fetch
from src.db import set_status                   # ADAPT: Postgres status lifecycle
from src.config import QDRANT_COLLECTION        # ADAPT: the ONE shared collection

KIND = "paper"


@dataclass
class PageText:
    page: int          # 1-based page number == the citation locator
    text: str


@dataclass
class Chunk:
    text: str
    payload: dict = field(default_factory=dict)   # must match the video chunk schema


# 1) PARSE ────────────────────────────────────────────────────────────────────
def parse_paper(data: bytes) -> list[PageText]:
    """PDF bytes → per-page text, preserving 1-based page numbers (the locator)."""
    pages: list[PageText] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:                       # skip blank / image-only pages here
                pages.append(PageText(page=i, text=text))
    return pages


# 2) CHUNK (page-aware) ───────────────────────────────────────────────────────
def chunk_paper(source_id: str, title: str, pages: list[PageText]) -> list[Chunk]:
    """
    Reuse the repo's semantic chunker PER PAGE so every chunk carries the page it
    came from. `page` must ride all the way to the payload — it's the locator the
    UI deep-links to. (If a chunk can span pages in your chunker, keep the page of
    its first token, or split on page boundaries first — page accuracy is graded.)
    """
    chunks: list[Chunk] = []
    for pg in pages:
        for piece in semantic_chunks(pg.text):          # ADAPT: signature may differ
            if not piece.strip():
                continue
            chunks.append(Chunk(
                text=piece,
                payload={
                    "kind": KIND,
                    "source_id": source_id,
                    "title": title,
                    "page": pg.page,          # ← locator
                },
            ))
    return chunks


# 3) FLOW: parse → chunk → enrich → embed → index ─────────────────────────────
def _point_id(source_id: str, page: int, idx: int) -> str:
    """Deterministic id → idempotent upsert (retries/re-runs never duplicate or
    re-do finished work). This is what makes `bench.py --resilience` pass."""
    return hashlib.sha1(f"{source_id}:p{page}:{idx}".encode()).hexdigest()


def ingest_paper(source_id: str, uri: str, title: str) -> None:
    """
    The paper flow body. Status lifecycle mirrors ingest_video EXACTLY:
        pending → parsing → chunking → enriching → embedding → indexed | failed

    CRASH-SAFETY: set `indexed` ONLY after the Qdrant upsert returns. If the
    worker dies mid-run, the source stays un-`indexed`, the queue redelivers it,
    and deterministic point ids mean re-running is a no-op on finished chunks.
    """
    try:
        set_status(source_id, "parsing")
        data = fetch_bytes(uri)                          # ADAPT
        pages = parse_paper(data)

        set_status(source_id, "chunking")
        chunks = chunk_paper(source_id, title, pages)
        for i, c in enumerate(chunks):
            c.payload["point_id"] = _point_id(source_id, c.payload["page"], i)

        set_status(source_id, "enriching")
        chunks = enrich_chunks(chunks)                   # ADAPT: shared, unchanged

        set_status(source_id, "embedding")
        vectors = embed_chunks(chunks)                   # ADAPT: shared, unchanged

        upsert_chunks(collection=QDRANT_COLLECTION,      # ADAPT: SAME collection as video
                      chunks=chunks, vectors=vectors)
        set_status(source_id, "indexed", pct=100)        # ← only AFTER a successful write
    except Exception as e:                                # noqa: BLE001
        set_status(source_id, "failed", error=str(e))
        raise                                            # let the queue apply its retry policy


# ── Prefect wiring (register alongside the video deployment) ──────────────────
# The video pipeline runs as a Prefect flow; add a paper branch that mirrors it.
# Keep per-task retries the same as video. Sketch (ADAPT to the repo's pattern in
# src/jobs.py — the video flow is your template):
#
#   from prefect import flow, task
#
#   @task(retries=3, retry_delay_seconds=10)
#   def _parse(uri):  return parse_paper(fetch_bytes(uri))
#   # …one @task per stage so a failed stage retries without redoing the rest…
#
#   @flow(name="ingest_paper")
#   def ingest_paper_flow(source_id: str, uri: str, title: str):
#       ingest_paper(source_id, uri, title)   # or compose the @task stages above
#
# Then have POST /admin/documents schedule THIS flow for kind="paper".
```

## `src/ingest/deck.py`

```python
"""
src/ingest/deck.py — Slide deck (PDF or PPTX) ingestion.

Mirrors the PROVIDED video flow, but the source is a slide deck and the citation
locator is a **slide** number:

    fetch → parse (per slide) → chunk (per slide) → enrich → embed → index

Key wrinkle vs. papers: image-only slides have little/no extractable text, so
caption them with the vision-capable LLM (env-switched, like the rest of the app)
before embedding — otherwise those slides retrieve poorly (a graded pitfall).

Everything lands in the SAME Qdrant collection, tagged `kind="deck"`.

⚠️  `# ADAPT` imports are best-guess names — reconcile with the real repo. Match
    the video flow's chunk schema, status enum, and the caption/embed/index calls.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import fitz  # PyMuPDF (PDF decks + rendering slides to images)   pip install pymupdf

# --- shared repo pieces (ADAPT) ----------------------------------------------
from src.rag.chunk import semantic_chunks       # ADAPT: same chunker video uses
from src.ingest.enrich import enrich_chunks      # ADAPT: shared
from src.rag.embed import embed_chunks           # ADAPT: shared
from src.rag.index import upsert_chunks          # ADAPT: shared Qdrant upsert
from src.storage import fetch_bytes              # ADAPT
from src.db import set_status                    # ADAPT
from src.config import QDRANT_COLLECTION         # ADAPT
from src.llm import caption_image                # ADAPT: vision-LLM caption(bytes|b64)->str

KIND = "deck"
MIN_TEXT_CHARS = 24          # below this, treat a slide as image-only and caption it


@dataclass
class Slide:
    slide: int         # 1-based slide number == the citation locator
    text: str          # slide text (+ appended caption for image-only slides)


@dataclass
class Chunk:
    text: str
    payload: dict = field(default_factory=dict)


# 1) PARSE ────────────────────────────────────────────────────────────────────
def parse_deck(data: bytes, filename: str = "") -> list[Slide]:
    """Deck bytes → per-slide text. Handles PPTX and PDF; captions image-only slides."""
    if filename.lower().endswith(".pptx"):
        return _parse_pptx(data)
    return _parse_pdf_deck(data)


def _parse_pptx(data: bytes) -> list[Slide]:
    from pptx import Presentation                 # pip install python-pptx
    prs = Presentation(io.BytesIO(data))
    slides: list[Slide] = []
    for i, slide in enumerate(prs.slides, start=1):
        texts = [sh.text for sh in slide.shapes if getattr(sh, "has_text_frame", False) and sh.text]
        text = "\n".join(t.strip() for t in texts if t.strip())
        if len(text) < MIN_TEXT_CHARS:
            # image-heavy slide: render it and caption with the vision LLM
            img = _render_pptx_slide_png(slide)   # ADAPT: see helper note below
            if img:
                text = (text + "\n" + caption_image(img)).strip()
        slides.append(Slide(slide=i, text=text))
    return slides


def _parse_pdf_deck(data: bytes) -> list[Slide]:
    """PDF deck: one page == one slide. Extract text; caption image-only slides."""
    slides: list[Slide] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) < MIN_TEXT_CHARS:
                png = page.get_pixmap(dpi=120).tobytes("png")
                text = (text + "\n" + caption_image(png)).strip()   # ADAPT
            slides.append(Slide(slide=i, text=text))
    return slides


def _render_pptx_slide_png(slide) -> bytes | None:  # noqa: ANN001
    """
    python-pptx can't rasterize a slide on its own. Options (pick one, ADAPT):
      • pull embedded picture shapes' blobs and caption those, or
      • convert the PPTX to PDF once (LibreOffice headless) and reuse _parse_pdf_deck.
    Returning None just skips captioning for that slide.
    """
    return None


# 2) CHUNK (slide-aware) ──────────────────────────────────────────────────────
def chunk_deck(source_id: str, title: str, slides: list[Slide]) -> list[Chunk]:
    """One (or a few) chunk(s) per slide, each carrying `slide` as the locator."""
    chunks: list[Chunk] = []
    for sl in slides:
        pieces = semantic_chunks(sl.text) if len(sl.text) > 400 else [sl.text]
        for piece in pieces:                             # ADAPT: chunker signature
            if not piece.strip():
                continue
            chunks.append(Chunk(
                text=piece,
                payload={
                    "kind": KIND,
                    "source_id": source_id,
                    "title": title,
                    "slide": sl.slide,        # ← locator
                },
            ))
    return chunks


# 3) FLOW ─────────────────────────────────────────────────────────────────────
def _point_id(source_id: str, slide: int, idx: int) -> str:
    return hashlib.sha1(f"{source_id}:s{slide}:{idx}".encode()).hexdigest()


def ingest_deck(source_id: str, uri: str, title: str, filename: str = "") -> None:
    """
    Deck flow body. Same status lifecycle + crash-safety as paper/video:
    set `indexed` only AFTER the Qdrant upsert succeeds.
    """
    try:
        set_status(source_id, "parsing")
        data = fetch_bytes(uri)                          # ADAPT
        slides = parse_deck(data, filename or uri)

        set_status(source_id, "chunking")
        chunks = chunk_deck(source_id, title, slides)
        for i, c in enumerate(chunks):
            c.payload["point_id"] = _point_id(source_id, c.payload["slide"], i)

        set_status(source_id, "enriching")
        chunks = enrich_chunks(chunks)                   # ADAPT

        set_status(source_id, "embedding")
        vectors = embed_chunks(chunks)                   # ADAPT

        upsert_chunks(collection=QDRANT_COLLECTION, chunks=chunks, vectors=vectors)  # ADAPT
        set_status(source_id, "indexed", pct=100)
    except Exception as e:                                # noqa: BLE001
        set_status(source_id, "failed", error=str(e))
        raise


# ── Prefect wiring: add an `ingest_deck` flow beside `ingest_paper` / video,
#    same per-task retries; POST /admin/documents schedules it for kind="deck".
```

## Integration notes


These two modules mirror the **provided** video flow (`src/ingest/pipeline.py :: ingest_video`).
They own only **parse + locator-aware chunking**; the enrich → embed → index tail is
**shared and reused unchanged**. Both upsert to the **same** Qdrant collection.

> **Read the repo first.** These are scaffolds shaped to the assignment's described
> pipeline — reconcile the `# ADAPT` points with the real names before wiring.

## 1. Dependencies
```
pip install pymupdf python-pptx      # add to requirements
# optional: LibreOffice (headless) if you rasterize PPTX slides for captioning
```

## 2. Reconcile the `# ADAPT` imports (match the video flow)
Find the real equivalents in the repo and fix the imports in both files:

| Scaffold name | Find the repo's real… |
|---------------|-----------------------|
| `semantic_chunks` | the chunker the video flow calls (`src/rag/chunk.py`) |
| `enrich_chunks` | shared enrichment stage |
| `embed_chunks` | shared embedding stage |
| `upsert_chunks` | shared Qdrant upsert (+ its collection arg) |
| `fetch_bytes` | object-storage / URL fetch (`src/storage.py`) |
| `set_status` | Postgres status lifecycle (`src/db.py`) |
| `QDRANT_COLLECTION` | the ONE shared collection name (`src/config.py`) |
| `caption_image` | the vision-LLM caption call (deck only) |

Also match the **chunk schema**: check what fields a *video* chunk carries and
produce the same shape, adding `page` (paper) / `slide` (deck) and `kind`.

## 3. Register the Prefect flows (`src/jobs.py`)
Add `ingest_paper` and `ingest_deck` flows **beside** the video deployment, with the
**same per-task retries and status lifecycle**. Use the video flow as the template —
don't invent a new pattern.

## 4. Admin API (`src/api/admin.py`)
- `POST /admin/documents` → validate `{uri, kind, title}`, insert a **`pending`** row,
  **schedule the right flow**, and return **`202 {id,status:"pending",kind}`** — *no
  parsing in the request path* (this is graded: search must not block on ingest).
- `GET /admin/sources` → unified video + document status with `kind` and `pct`.
- Errors: `400` bad input · `401` bad/missing admin token · `502` upstream failure.

## 5. Cross-source citations (`src/rag/`, `ui/`)
Retrieval already hits the shared index — make sure `kind` and the `locator`
(`page` / `slide` / `start_ms`) ride into the citation payload, and render per kind:
video → seek `start_ms`, paper → open `page`, deck → show `slide`.

## 6. Crash-safety (the `--resilience` gate)
Already baked into the scaffolds, keep it:
- `set_status(..., "indexed")` runs **only after** the Qdrant upsert returns.
- Deterministic `point_id = sha1(source_id + locator + idx)` → retries/re-runs are
  idempotent; finished chunks aren't re-done.

## 7. Verify
```bash
# async accept → 202 immediately
curl -si localhost:8100/admin/documents -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"uri":"https://arxiv.org/pdf/2312.10997","kind":"paper","title":"RAG Survey"}' | head -1
# indexed → a query returns a PAGE citation
curl -sN "localhost:8100/ask_stream?q=hybrid+retrieval" | grep -m1 '"page"'
python benchmark/bench.py --resilience   # kill-a-worker, assert no loss
```

**Don't** touch the video pipeline, and don't commit `.env`, media, or PDFs.
