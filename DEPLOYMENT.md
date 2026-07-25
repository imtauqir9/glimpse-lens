# Deploying MomentSearch to Fly.io

MomentSearch ships as **one Docker image** that runs as **three long-running
process groups** (`api`, `worker`, `clip`) plus a **one-shot seed gate** — each
on its own Fly machine, each scaled by its own bottleneck. This is the whole
point of the architecture: the pieces scale in different directions, so they
live on different machines.

```
                 ┌────────────── one image, three process groups ──────────────┐
 users ──HTTPS──►│  api      (:8000, public)   presign · register · search · UI │
                 │  worker   (no ports)        pulls ingest runs from Prefect    │
                 │  clip     (:8001, internal) ONE warm CLIP model behind a URL  │
                 └───────┬──────────────┬───────────────────┬───────────────────┘
                         ▼              ▼                    ▼
                   Neon Postgres   Prefect Cloud        Qdrant Cloud
                   (manifest)      (work queue)         (vectors)
                         ▲              ▲                    ▲
                         └───────  GCS bucket (videos + frame thumbnails)  ──────┘
```

## Why three separate services (not one box)

| Service | Scales on | Machine | Why separate |
|---|---|---|---|
| **api** | request concurrency | tiny, auto-stops when idle | stateless HTTP; must answer `202` instantly and never block on heavy work |
| **worker** | ingest throughput | cheap CPU, scale to N | download + ffmpeg per video; add replicas for a backfill, remove them after |
| **clip** | embedding FLOPs | one warm model (→ GPU later) | loading CLIP costs ~15–30s; doing it once in a shared service, not per-video, is the difference between fast and unusable |

If these were one process, you'd pay for a GPU on every web box, or reload the
model on every video, or block uploads behind embedding. Splitting them lets
each grow (and cost) independently: `fly scale count worker=5` for a big import,
or point `CLIP_SERVICE_URL` at a GPU machine when embedding is the wall — with
**zero code changes**.

Everything stateful is a rented managed service (Neon, Prefect Cloud, Qdrant
Cloud, GCS), so every Fly machine is disposable — "nothing on local."

## Prerequisites

You already have these wired in `.env` (they're external, so the same accounts
work from Fly):

- **Neon Postgres** — `DATABASE_URL`
- **Prefect Cloud** — `PREFECT_API_URL`, `PREFECT_API_KEY`
- **Qdrant Cloud** — `QDRANT_URL`, `QDRANT_API_KEY`
- **Object storage** — `STORAGE_PROVIDER=gcp_native` + the `GOOGLE_CLOUD_*` keys
  (bucket `momentsearch-media`)
- **LLM** — `LLM_API_KEY`
- A **Fly.io account** + the `flyctl` CLI installed.

Two more get created during the deploy rather than beforehand — `REDIS_URL`
(step 3a) and `ADMIN_TOKEN` (step 3b).

> **The sample corpus is already indexed** in your shared Qdrant/Neon from local
> runs, so the deploy's seed gate finds them done and skips re-downloading — the
> deploy won't be blocked by YouTube.

## Deploy — step by step

All commands assume you're in the repo root. On Windows use PowerShell.

### 1. Authenticate

`flyctl` reads the `FLY_API_TOKEN` env var. The token lives in `.env` as
`FLY_IO_TOKEN` — load it into the session (this also works headless/CI, no
browser login needed):

```powershell
$env:FLY_API_TOKEN = ((Select-String '^FLY_IO_TOKEN=' .env).Line -replace '^FLY_IO_TOKEN=','').Trim().Trim('"')
fly auth whoami        # confirm it's your account
```

Bash equivalent:

```bash
export FLY_API_TOKEN="$(grep '^FLY_IO_TOKEN=' .env | cut -d= -f2- | tr -d '\r\"')"
fly auth whoami
```

### 2. Create the app (once)

```powershell
fly apps create momentsearch --org personal
```

If the name is taken, pick another (e.g. `momentsearch-<you>`) and update **two
places** in `fly.toml`: the `app = '…'` line and the `CLIP_SERVICE_URL`
internal-DNS host (`clip.process.<app-name>.internal`).

### 3. Push secrets (once, and whenever they change)

Import everything from `.env` except the local-only bits, then add the YouTube
cookies as a base64 secret (there's no `./data` mount on Fly, so the file path
won't work there — the worker decodes the secret to a temp file at runtime):

```powershell
# import .env (skip FLY_ and the local cookie FILE path)
Get-Content .env |
  Where-Object { $_ -match '^[A-Z_]+=.+' -and $_ -notmatch '^FLY_' -and $_ -notmatch '^YT_COOKIES_FILE=' } |
  fly secrets import

# YouTube cookies as a secret (needed because Fly's datacenter IP is bot-checked)
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("data/cookies.txt"))
fly secrets set YT_COOKIES_B64="$b64"
```

### 3a. Attach Redis (cluster-wide rate limits + metrics)

Rate limiting and the metrics counters are only correct if **every replica shares
one counter store**. With no `REDIS_URL`, `src/redis_client.py` falls back to
per-process memory: a `RATE_LIMIT_PER_MIN=30` limit silently becomes 30 × the
number of api machines, and `/metrics` reports whichever machine answered the
scrape. Single replica → harmless. The moment you `fly scale count api=2` → wrong.

```powershell
fly redis create                 # Upstash; pick the same region as the app (iad)
fly secrets set REDIS_URL='redis://default:<password>@<name>.upstash.io:6379'
```

The URL resolves over Fly's private network — no public egress, no extra cost at
this volume. `redis>=5.0` is already in `requirements.txt`, so the image picks it
up on the next build; the client is lazy-imported and only used when the URL is
set. Confirm after deploy — the api logs print one line at first use:

```
[redis] connected (<name>.upstash.io:6379)
```

If you see `[redis] unavailable (…) — using in-memory fallback` instead, the app
is still up and serving; it just isn't multi-replica-correct. Fix the URL.

### 3b. Turn on auth (and mint the first key)

Auth **fails open by design**: `src/auth.py` enforces keys only once an
`ADMIN_TOKEN` is set or at least one key exists — that's what keeps `docker
compose up` zero-config for local dev. On a public Fly URL that default means
`/admin/*` (ingest, delete, key minting) and `/metrics` are wide open. Set the
token as part of the first deploy, not after:

```powershell
fly secrets set ADMIN_TOKEN="$([guid]::NewGuid().Guid)"   # or any long random string
```

`ADMIN_TOKEN` doubles as a master admin key — that's the bootstrap. Use it once to
mint real per-tenant keys, then use those:

```powershell
$APP = 'https://glimpse-lens.fly.dev'
curl -s "$APP/admin/keys" -H "Authorization: Bearer $ADMIN_TOKEN" `
  -H 'content-type: application/json' `
  -d '{"user_id":"acme","role":"viewer","label":"demo read-only"}'
```

The plaintext key (`glk_…`) is returned **once** — only its SHA-256 hash is
stored, so a database leak never yields a usable key. The key also determines the
tenant, which is why `X-User-Id` stops being honored once auth is active: tenant
isolation can no longer be bypassed by editing a header.

Two tables back this — `ms_api_keys` (Phase 1) and `ms_audit` (Phase 3, the
append-only who-did-what trail). Both are `CREATE TABLE IF NOT EXISTS` run in the
API's startup lifespan, so they appear in Neon on the first boot of the new
version. **Nothing to migrate by hand.**

### 4. Deploy

```powershell
fly deploy --ha=false
```

> **If the build fails with a 403** — e.g. `error building: ... (status 403):
> Your account has been marked as high risk`, or the remote builder is otherwise
> refused/unavailable — build the image **locally** instead (needs Docker Desktop
> running) so it never touches Fly's remote builder:
>
> ```powershell
> fly deploy --ha=false --local-only
> ```
>
> This builds with your local Docker daemon and pushes the finished image to
> `registry.fly.io`. Alternatively, verify the account at
> <https://fly.io/high-risk-unlock> to use the remote builder.

On deploy, fly.toml's `release_command` runs the **seed gate** first
(`python -m src.seed`). Because the samples are already indexed in your shared
Qdrant/Neon, it exits in seconds and the app goes live. If it can't verify the
samples it aborts and the previous version keeps serving — you never get a
half-indexed app.

### 5. Open it

```powershell
fly open           # -> https://momentsearch.fly.dev/
fly logs           # tail all processes
```

## Scaling knobs

```powershell
fly scale count worker=3          # more ingest throughput (concurrent videos)
fly scale count worker=0 clip=0   # between ingest sessions — queued runs just wait
fly secrets set WORKER_CONCURRENCY=3   # more videos per worker machine
```

The `api` machine auto-stops when idle and auto-starts on the next request
(`min_machines_running = 0` in fly.toml), so it costs almost nothing at rest.

> Past `api=1`, set `REDIS_URL` first (step 3a) — otherwise the rate limiter and
> the metrics counters fragment per machine.

## Monitoring the deployed app

The app exposes three things, all admin-gated:

| Endpoint | What |
|---|---|
| `/metrics` | Prometheus text — request latency histograms, counters, live ingest queue depth (computed from Postgres at scrape time) |
| `/api/metrics.json` | the same rollup as JSON, plus token/cost metering |
| `/dashboard` | self-contained observability UI that polls the JSON |

`/dashboard` needs nothing beyond the deploy — open `$APP/dashboard` and paste an
admin key. For history and alerting you need a Prometheus, and there are two ways:

**Option 1 — run the compose stack against Fly (works today).** `monitoring/` is
a full Prometheus + Grafana + alert-rules stack. Point it at the deployed app by
editing the target in `monitoring/prometheus.yml`:

```yaml
    static_configs:
      - targets: ['glimpse-lens.fly.dev']
        labels: { service: glimpse }
    scheme: https          # add this line — the Fly app is HTTPS-only
```

then `docker compose up -d prometheus grafana` (Grafana on `:3000`, admin/admin;
the Glimpse dashboard is auto-provisioned). The scrape authenticates with
`ADMIN_TOKEN` from `.env` — the compose entrypoint writes it to
`/tmp/prom_token`, which `prometheus.yml` reads via `credentials_file`. So the
`.env` token must match the Fly secret.

**Option 2 — Fly's managed Prometheus.** Uncomment the `[[metrics]]` block in
`fly.toml` and metrics land in `fly-metrics.net` (hosted Grafana, no containers
to run). **Caveat:** Fly's scraper sends no `Authorization` header, so it gets
401 against the admin-gated `/metrics` as written — you'd have to drop the auth
dependency on that one route first. That's defensible (Fly scrapes over the
private network, never the public edge) but it's a real decision, which is why
the block ships commented out.

`monitoring/alerts.yml` carries the rules either way: `HighErrorRate` (>5% 5xx),
`SearchLatencyP95High` (retrieve p95 > 1.5s — the decoupling SLA), `RateLimitSpike`
(>50 throttled in 5m, possible abuse), and `LLMCostSpike` (>$5/hour, runaway spend).

Prometheus is deliberately **not** a Fly process group: its TSDB wants a
persistent volume, and paying for a per-app time-series database is the wrong
trade when option 1 and option 2 both exist.

## CI/CD (optional)

`.github/workflows/fly-deploy.yml` redeploys automatically on **every push to
`dev`** (it runs `flyctl deploy --remote-only`). One-time setup — add a deploy
token as the `FLY_API_TOKEN` repo secret:

```powershell
fly tokens create deploy -x 999999h
# GitHub → Settings → Secrets and variables → Actions → New repository secret
```

> CI uses Fly's **remote** builder, so if the account is flagged "high risk"
> (see Troubleshooting) CI deploys fail there too — unlock the account, or deploy
> manually with `fly deploy --local-only` from a machine that has Docker until
> it's cleared.

## Cost (rough)

| Piece | At rest | Active |
|---|---|---|
| api (auto-stop) | ~$0 | ~$2–6/mo |
| worker | scale to 0 between sessions | ~$2–5/mo up |
| clip | scale to 0 between sessions | ~$2–5/mo (CPU) |
| Neon / Prefect / Qdrant | free tiers | — |
| GCS | ~$1–2/mo per 50 GB | — |
| LLM | — | ~$0.005–0.01 per question |

Everything-on ≈ **$40/mo**; idle-scaled with free tiers ≈ **$5–10/mo**. GPU (for
the clip service) is a burst cost only — rent it for a big backfill, kill it after.

## Troubleshooting

- **Build fails with 403 / "high risk account" / remote builder error** → Fly's
  shared remote builder refused the build. Build locally instead:
  `fly deploy --ha=false --local-only` (needs Docker Desktop running), or unlock
  the account at <https://fly.io/high-risk-unlock>. This is a builder/account
  issue, not a code issue — the same image builds fine locally.
- **Deploy aborts on release_command** → the seed gate couldn't verify samples.
  Check `fly logs`; usually a bad `DATABASE_URL`/`QDRANT_URL` secret. Set
  `SEED_SAMPLE_VIDEOS=false` to skip the gate if you need to deploy anyway.
- **YouTube ingest fails on Fly** → datacenter IP is blocked; make sure
  `YT_COOKIES_B64` is set (step 3). Cookies expire in ~2–3 weeks; re-run the
  `fly secrets set YT_COOKIES_B64=…` command to refresh. Uploads are unaffected.
- **Browser uploads fail** → the GCS bucket needs a CORS rule allowing `PUT`
  from your site's origin (see `.env.example`).
- **`clip` unreachable** → confirm `CLIP_SERVICE_URL` in fly.toml matches the
  app name (`clip.process.<app>.internal:8001`).
- **`401 Missing or invalid API key` everywhere after setting `ADMIN_TOKEN`** →
  expected: auth activates the moment the token exists. Send
  `Authorization: Bearer <ADMIN_TOKEN>`, or mint a key (step 3b). The UI and
  `/dashboard` both prompt for one.
- **`/admin/*` still open on the public URL** → `ADMIN_TOKEN` isn't set and no key
  has been minted, so auth is inactive (fail-open dev default). `fly secrets list`
  to confirm, then step 3b.
- **Rate limit seems too generous, or `/metrics` numbers jump around** → more than
  one api machine with no `REDIS_URL`; each keeps its own counters. Step 3a.
- **`[redis] unavailable` in the logs** → wrong URL, or the Upstash instance is in
  another region/org. The app keeps serving on the in-memory fallback, so this
  never pages — but it won't be multi-replica-correct until it's fixed.
- **Prometheus target shows `401`** → the scrape token doesn't match. The compose
  stack reads `ADMIN_TOKEN` from `.env`; make sure it equals the Fly secret.
