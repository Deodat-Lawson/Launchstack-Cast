# Cast — Implementation Plan

Status: draft v0.1 — awaiting maintainer sign-off before any scaffolding.

---

## 1. Repo layout

```
cast/
├── cmd/
│   ├── cast-api/           # HTTP server entrypoint
│   └── cast-worker/        # Background job runner (pulls jobs from DB queue)
├── internal/
│   ├── api/                # HTTP handlers, routing, request/response types
│   ├── config/             # Env-driven config loading
│   ├── db/                 # pgx pool, query helpers, sqlc-generated code
│   │   └── queries/        # .sql files consumed by sqlc
│   ├── ingest/             # ffmpeg + scenedetect orchestration
│   ├── pyworker/           # Go ↔ Python subprocess contract
│   ├── perception/         # Calls pyworker for whisper/clip/insightface
│   ├── identity/           # Face-embedding clustering, entity resolution
│   ├── graph/              # Entity/clip/state graph writes + reads
│   ├── llm/                # Anthropic + OpenAI clients for caption/state extraction
│   ├── retrieval/          # Hybrid scoring, RRF, query execution
│   └── pipeline/           # End-to-end job orchestration (ingest → ... → graph)
├── python/
│   ├── workers/
│   │   ├── transcribe.py   # Whisper
│   │   ├── visual_embed.py # OpenCLIP
│   │   └── faces.py        # InsightFace detect + ArcFace embed
│   ├── common/             # Shared stdin/stdout JSON helpers
│   ├── pyproject.toml      # uv-managed
│   └── README.md
├── migrations/             # goose SQL migrations, numbered
├── docs/
│   ├── PLAN.md             # this file
│   ├── ARCHITECTURE.md     # deferred — written after v1 lands
│   └── api.md              # generated from handler godoc, deferred
├── scripts/                # dev helpers (seed, reset-db, etc.)
├── web/                    # placeholder for Next.js frontend (empty)
├── testdata/               # small fixture videos + golden outputs
├── docker-compose.yml      # local postgres+pgvector
├── Makefile
├── go.mod
├── .env.example
├── LICENSE
└── README.md
```

**Rationale**

- `cmd/` holds only `main.go` — wiring, nothing else. One binary per process.
- `internal/` because Cast is an application, not a library. No external imports.
- Package names are single words (`identity`, `graph`, `retrieval`) so call sites read `identity.Resolve(...)`, `graph.WriteClip(...)`.
- `pyworker` is the *contract* (Go side); `python/workers/` is the *implementation*. Kept separate so the Go side has one place to change when the wire format evolves.
- `sqlc` generates Go from `.sql` — we write SQL, not an ORM DSL. Keeps "one way to do each thing".

---

## 2. Postgres schema

Single database, `cast`. Extensions: `pgvector`, `pg_trgm` (for fuzzy name match on entity labels), `pgcrypto` (UUIDv4).

### Core tables

```sql
-- A submitted video. Row created the moment a job is accepted.
videos (
  id             uuid primary key default gen_random_uuid(),
  source_url     text,                    -- nullable if uploaded
  source_path    text not null,           -- local/object-store path after ingest
  duration_sec   double precision,
  status         text not null,           -- pending|ingesting|perceiving|resolving|extracting|ready|failed
  error          text,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
)

-- Detected scenes. One row per PySceneDetect cut.
clips (
  id             uuid primary key default gen_random_uuid(),
  video_id       uuid not null references videos(id) on delete cascade,
  idx            int not null,            -- ordinal within video
  start_sec      double precision not null,
  end_sec        double precision not null,
  keyframe_path  text,                    -- jpg on disk/S3
  caption        text,                    -- VLM-produced, nullable until extraction step
  transcript     text,                    -- concatenated whisper segments
  transcript_tsv tsvector generated always as (to_tsvector('english', coalesce(transcript,''))) stored,
  visual_embed   vector(512),             -- OpenCLIP ViT-B/32
  transcript_embed vector(1024),          -- optional dense text embed; nullable v1
  unique (video_id, idx)
)
create index on clips using ivfflat (visual_embed vector_cosine_ops) with (lists=100);
create index on clips using gin (transcript_tsv);

-- Per-clip whisper segments (timed). Needed for "who-said-what-when".
transcript_segments (
  id         bigserial primary key,
  clip_id    uuid not null references clips(id) on delete cascade,
  start_sec  double precision not null,
  end_sec    double precision not null,
  text       text not null
)
create index on transcript_segments(clip_id, start_sec);

-- Every detected face instance in a keyframe, BEFORE clustering.
-- This is the raw perception output; entity_id is filled in by identity resolver.
face_detections (
  id         bigserial primary key,
  clip_id    uuid not null references clips(id) on delete cascade,
  frame_sec  double precision not null,          -- timestamp within clip
  bbox       int4[] not null,                    -- [x,y,w,h]
  embedding  vector(512) not null,               -- ArcFace
  det_score  real not null,
  entity_id  uuid references entities(id)        -- null until clustered
)
create index on face_detections using ivfflat (embedding vector_cosine_ops) with (lists=200);
create index on face_detections(entity_id) where entity_id is not null;
create index on face_detections(clip_id);

-- Stable identity across a video (v1: per-video; v2: cross-video).
entities (
  id             uuid primary key default gen_random_uuid(),
  video_id       uuid not null references videos(id) on delete cascade,
  label          text,                           -- human-assigned, nullable
  centroid_embed vector(512),                    -- mean of cluster, refreshed on merge
  n_detections   int not null default 0,
  created_at     timestamptz not null default now()
)
create index on entities(video_id);
create index on entities using gin (label gin_trgm_ops);

-- The *graph* — state snapshots per (entity, clip). This is the "temporal state graph".
-- We treat it as a join table; dimensions are columns, not a JSONB blob, so we can
-- index and query them directly. Schema evolves; migrations add columns.
entity_states (
  id              bigserial primary key,
  entity_id       uuid not null references entities(id) on delete cascade,
  clip_id         uuid not null references clips(id) on delete cascade,
  emotional_state text,          -- e.g. "resolute", "afraid"
  knowledge       text,          -- what they now know
  allegiance      text,          -- who they serve / side
  goal            text,          -- active objective
  raw             jsonb not null, -- full LLM output for forensics
  unique (entity_id, clip_id)
)
create index on entity_states(clip_id);
create index on entity_states(entity_id);

-- Edges between entities, if the LLM extracts relationships.
entity_relations (
  id           bigserial primary key,
  video_id     uuid not null references videos(id) on delete cascade,
  subject_id   uuid not null references entities(id) on delete cascade,
  object_id    uuid not null references entities(id) on delete cascade,
  predicate    text not null,         -- "trusts", "betrays", "leads"
  clip_id      uuid references clips(id) on delete set null,  -- when this was established
  raw          jsonb not null
)
create index on entity_relations(subject_id);
create index on entity_relations(object_id);

-- Background job queue. Simple SELECT ... FOR UPDATE SKIP LOCKED pattern;
-- avoids introducing Redis/NATS at this stage.
jobs (
  id          bigserial primary key,
  video_id    uuid not null references videos(id) on delete cascade,
  kind        text not null,           -- ingest|perceive_clip|resolve_identity|extract_state|...
  payload     jsonb not null default '{}'::jsonb,
  status      text not null default 'pending',  -- pending|running|done|failed
  attempts    int not null default 0,
  last_error  text,
  locked_by   text,
  locked_at   timestamptz,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
)
create index on jobs(status, kind) where status = 'pending';
```

### Notes

- **Vector dimensions**: OpenCLIP ViT-B/32 → 512, InsightFace buffalo_l → 512. If we swap models we add a migration + re-embed pass.
- **Index choice**: `ivfflat` over HNSW for v1 — smaller build time, acceptable recall at our scale. Revisit when we hit ~1M face detections.
- **BM25**: v1 uses built-in `tsvector` + `ts_rank_cd`. Good enough, zero extra extensions. If ranking quality is poor, install `paradedb`'s `pg_search`; schema shouldn't need to change (new GIN index on same column).
- **Graph as join tables**: `entity_states` and `entity_relations` *are* the graph. No graph DB. Traversal is SQL joins. Documented in `docs/ARCHITECTURE.md` later.
- **Job queue in Postgres**: `SELECT ... FOR UPDATE SKIP LOCKED` gives us at-least-once without a broker. When throughput demands more we can move to NATS/Redis.

---

## 3. Go package boundaries

| Package | Depends on | Responsibilities |
|---|---|---|
| `config` | stdlib | Load env into typed struct. No side effects. |
| `db` | pgx, sqlc output | Pool, transactions, generated query funcs. No business logic. |
| `pyworker` | stdlib `os/exec` | Generic `Run(ctx, script, stdin) (stdout []byte, err error)`. Owns timeouts, stderr capture, exit-code mapping. |
| `perception` | `pyworker`, `db` | Thin adapters: `Transcribe(clip)`, `VisualEmbed(clip)`, `DetectFaces(clip)`. Parses JSON from pyworker; writes rows via `db`. |
| `ingest` | `os/exec`, `db` | ffmpeg audio extract, PySceneDetect invocation, keyframe extraction. Creates `clips` rows. |
| `identity` | `db` | Pulls face embeddings for a video, clusters them, writes `entities` + backfills `face_detections.entity_id`. v1 = threshold-based agglomerative; HDBSCAN deferred. |
| `llm` | anthropic-sdk, openai-go | Clients + the one struct describing the caption+state prompt. Retries, token accounting. Provider chosen by config, no interface layer yet (YAGNI — add when second real use site exists). |
| `graph` | `db`, `llm` | Per-clip: build prompt from (transcript, caption, entities present), call LLM, write `entity_states` + `entity_relations`. |
| `retrieval` | `db` | Hybrid query: (a) vector search on `clips.visual_embed`, (b) tsvector on transcript, (c) optional dense transcript, (d) graph predicate filter. RRF combine. |
| `pipeline` | all of the above | The orchestrator. Consumes jobs from `jobs` table, advances a video through stages, transitions `videos.status`. |
| `api` | `db`, `pipeline`, `retrieval` | HTTP handlers. Zero business logic — delegates to the above. |

Dependency rule: `api` and `pipeline` may import anything; lower-level packages (`db`, `pyworker`, `config`) import nothing from Cast. Enforced by convention, not tooling.

---

## 4. Go ↔ Python worker contract

### Invocation

```
python -u python/workers/transcribe.py < input.json > output.json
```

- Go side uses `os/exec.CommandContext`. Python interpreter path and script root come from config (`CAST_PYTHON_BIN`, `CAST_PYTHON_ROOT`).
- Payload is a single JSON object on stdin. Response is a single JSON object on stdout.
- **Logs and progress go to stderr, always.** Stdout is reserved for the JSON response.
- Timeouts live on the Go side via `context.WithTimeout`. On timeout Go sends SIGTERM, then SIGKILL after 5s grace. No Python-side watchdog.

### Request schema (Go → Python)

```json
{
  "request_id": "uuid",
  "inputs": { /* script-specific */ },
  "options": { /* script-specific */ }
}
```

### Response schema (Python → Go)

```json
{
  "request_id": "uuid",
  "ok": true,
  "data": { /* script-specific */ },
  "error": null,
  "meta": { "duration_ms": 1234, "model": "whisper-large-v3" }
}
```

On failure: `ok: false`, `error: { "code": "...", "message": "..." }`, non-zero exit code. Go side: if exit code is zero, trust `ok`; if non-zero, synthesize an error from stderr tail.

### Per-worker schemas (sketch — detailed in each script's docstring)

- `transcribe.py`: in `{audio_path, language?}` → out `{segments: [{start, end, text}], language}`
- `visual_embed.py`: in `{image_paths: [...]}` → out `{embeddings: [[float]...]}`
- `faces.py`: in `{image_path}` → out `{detections: [{bbox, score, embedding}]}`

### Error handling

- Python uncaught exception → `sys.exit(1)` with JSON error on stdout *and* traceback on stderr (via a `common/run.py` decorator).
- Missing model files, bad audio, etc. → structured `error.code` so Go can distinguish transient (retry) from permanent (fail job).

### Why not gRPC / a long-lived Python server?

Subprocess-per-call is simpler, crash-isolates ML code, and lets the OS reclaim GPU memory between jobs. Model load time is amortized across one clip's worth of calls per worker — acceptable for v1. Revisit if load time dominates.

---

## 5. Migrations

- **Tool**: `goose`. Single Go binary, supports SQL-only migrations, plays well with `make`, no Ruby/Node dependency.
- **Naming**: `migrations/NNNN_short_snake_case.sql` with `-- +goose Up` / `-- +goose Down` markers. Numbers are zero-padded 4 digits.
- **Rollback policy**: every migration has a working `Down`. For destructive ops (drop column, drop table) the `Down` recreates the structure but not the data — documented in a comment. No production rollback is expected; `Down` exists for local dev.
- **Shadow/prod parity**: `make migrate-up` runs against `DATABASE_URL`. CI runs migrations against a fresh Postgres in docker before tests.

---

## 6. Config + secrets

- Single struct in `internal/config/config.go`, populated via `caarlos0/env/v11` from env vars.
- `.env.example` checked in; `.env` gitignored.
- Precedence: env → `.env` file (loaded only when `CAST_ENV=dev`) → defaults in the struct.
- Secrets (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`) never logged; config struct has a `String()` that redacts.
- No config file format (no yaml, no toml). Env only. "One way to do each thing."

---

## 7. Testing strategy

| Layer | Approach |
|---|---|
| Pure Go units (`identity` clustering math, `retrieval` RRF) | Table-driven tests, stdlib `testing`. |
| `db` package | Integration tests against a real Postgres spun up by `docker-compose` (or `testcontainers-go` in CI). No mocks. |
| `api` handlers | `httptest.Server` + real DB. Assert on JSON bodies. |
| `pyworker` contract | Go test invokes a fixture Python script (`testdata/workers/echo.py`) and asserts the protocol. Doesn't require ML models. |
| ML workers themselves | Python `pytest` with a 5-second clip fixture checked into `testdata/`. Run in a separate CI job, skippable locally via `make test-go` vs `make test-all`. |
| End-to-end | One `make e2e` target that ingests a 10-second sample video and asserts final state. Skipped in normal CI (too slow); run nightly. |

No mock library. Use interfaces sparingly; when a fake is needed, handwrite it next to the test.

---

## 8. HTTP API surface (v1)

Framework: **stdlib `net/http` + Go 1.22 ServeMux**. No chi, no echo, no gin. Middleware is plain `func(http.Handler) http.Handler`.

```
POST   /v1/videos                 -> { source_url | upload } → 202 { video_id, status }
GET    /v1/videos/{id}            -> video + stage status
GET    /v1/videos/{id}/clips      -> list clips with captions
GET    /v1/videos/{id}/entities   -> list entities with centroid thumbnails
GET    /v1/entities/{id}          -> entity + state timeline
POST   /v1/entities/{id}/label    -> { label: "Alice" } → 200
POST   /v1/query                  -> hybrid retrieval
         request:  { video_id?, text?, entity_id?, k?: 10 }
         response: { results: [{ clip_id, score, why: [...breakdown] }] }
GET    /healthz                   -> 200 ok
GET    /readyz                    -> DB ping
```

Conventions:
- JSON only. `application/json; charset=utf-8`.
- Errors: `{ "error": { "code": "...", "message": "..." } }` + appropriate HTTP status.
- UUIDs in paths, never sequential IDs.
- No pagination in v1 retrieval response (k capped at 100).

---

## 9. Observability

**In scope for v1:**
- Structured logging via `log/slog` (stdlib), JSON handler in prod, text in dev.
- Request logging middleware: method, path, status, duration, request_id.
- `request_id` propagated to pyworker via the JSON payload; pyworker echoes it back and includes it in stderr lines.
- `/healthz` (liveness), `/readyz` (DB ping).

**Deferred:**
- OpenTelemetry tracing. Plumbing is invasive; add once we have a real deployment target.
- Metrics (Prometheus). Add with OTel.
- Error tracking (Sentry). Add when we have users.

---

## 10. Deferred to v2

- Frontend (Next.js). `web/` stays empty with a README stub.
- Auth, multi-tenancy, API keys.
- Cross-video entity resolution (v1 is per-video only).
- Graph traversal queries beyond simple filters (e.g. "find clips where A's allegiance changed while B was present").
- HDBSCAN clustering (v1 uses threshold agglomerative).
- Dense transcript embeddings (`transcript_embed` column exists, stays null v1).
- ParadeDB / advanced BM25.
- Worker autoscaling, distributed job queue.
- OpenTelemetry.
- yt-dlp ingestion of URLs (v1 accepts uploaded files + pre-resolved URLs only).

---

## 11. Open questions for the maintainer

1. **Module path**: `github.com/launchstack/cast`? Or different org?
2. **License**: MIT, Apache-2.0, or AGPL? Affects what we can vendor.
3. **GPU expectation**: do we assume CUDA is available for Python workers in dev? Affects the docker-compose story and the `faces.py` default backend (InsightFace CPU is ~10x slower).
4. **Video storage**: local disk under `CAST_DATA_DIR` (simple) or S3/R2-compatible object store from day one (more work, but production-realistic)?
5. **Ingest surface**: does `source_url` need to support arbitrary web URLs (yt-dlp) in v1, or only direct video file URLs? yt-dlp adds a subprocess dep and legal surface area.
6. **Anthropic vs OpenAI**: pick one for v1 or support both behind a `CAST_LLM_PROVIDER` env var? I'd lean "Anthropic only, add OpenAI when there's a reason" — the provider interface is the kind of abstraction we said we'd avoid until needed.
7. **Python tooling**: `uv` (fast, modern) vs `pip + requirements.txt` (boring, universal)? I'd pick `uv`.
8. **Worker process model**: confirmed one subprocess per call? Alternative is a persistent Python process per worker type with a line-delimited JSON protocol (faster, but adds lifecycle management). v1 = per-call; flag if you want me to reconsider.
9. **`cmd/cast-worker` vs in-process worker**: separate binary (cleaner scaling) or same process as `cast-api` with a goroutine pool (simpler dev)? I lean separate binary; docker-compose runs both.
