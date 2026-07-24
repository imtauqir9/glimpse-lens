# benchmark/ — the SLA gate

`bench.py` proves the five assignment SLAs against a **running** deployment and
**exits non-zero** if any is missed (it's the grading gate — never eyeball).

| Metric | Target | How it's measured |
|---|---|---|
| `/admin/documents` accept p95 | ≤ 300 ms | time each POST; p95 over the golden docs |
| Search p95 during a big ingest | ≤ 1.3× idle p95 | sample `/api/ask` idle, then again while a backfill floods the queue |
| Cross-source recall@10 | ≥ 0.70 | each golden query's expected source must appear in top-10 citations |
| No-loss under worker crash | 100% | `--resilience`: kill a worker mid-ingest, assert all resume to `indexed` |
| Ingestion throughput (≥2 workers) | ≥ 8 chunks/s | total chunks indexed ÷ wall-clock |

## Run

```bash
export BASE_URL=http://localhost:8100
export ADMIN_TOKEN=...          # if the server sets one
export BENCH_USER=bench         # isolate benchmark data in its own tenant
cp benchmark/golden.example.json benchmark/golden.json   # then fill with REAL queries

python benchmark/bench.py                 # accept + throughput + recall + decoupling
python benchmark/bench.py --resilience     # kill-a-worker no-loss
```

## The golden set

`golden.json` drives three checks at once. `documents` are ingested by the
harness (feeding accept-latency + throughput); each `query` names the one source
that must show up in its top-10 citations (recall). Include a **paper, a deck,
and a video** so recall proves *cross-source* retrieval, not just one modality.

**Never fabricate numbers** — every value bench prints comes from a real run
against your deployment (a red-line in the assignment).

## Resilience kill hook

The crash is pluggable so CI can run it unattended:

```bash
export BENCH_KILL_CMD="docker kill ms-worker-1"     # or: pkill -9 -f worker.py
python benchmark/bench.py --resilience
```

Without `BENCH_KILL_CMD`, `--resilience` pauses and asks you to kill a worker by
hand. No-loss passes only because ingest is **commit-then-complete** with
**deterministic point IDs** (see `src/ingest/RECONCILED.md`).

## Endpoint names

`bench.py` targets `/admin/documents`, `/admin/sources`, `/api/ask` (the base
repo's real ask path). If your assignment grades a `/ask_stream` SSE variant or
`/api/documents`, change the `*_PATH` constants at the top of `bench.py`.
```
