# Cast

Cast is a multimodal video retrieval system that combines face-cluster identity resolution with LLM-extracted temporal state graphs. Built for video corpora where *who* matters as much as *what* — narrative film, training videos, recorded meetings, instructional content. Powered by Whisper, InsightFace, OpenCLIP, and Postgres + pgvector.

## Architecture

Cast ingests video, splits it into scenes, runs ML perception (transcription, visual embedding, face detection), clusters faces into stable entity identities, extracts per-entity temporal state via LLM, and serves hybrid retrieval queries across all of it.

- **Go** — orchestration, identity resolution, graph construction, retrieval API, all business logic
- **Python** — narrow ML inference workers (Whisper, InsightFace, OpenCLIP), invoked as subprocesses
- **Postgres 16 + pgvector** — vectors, full-text search, entity/state graph, job queue

See [docs/PLAN.md](docs/PLAN.md) for the full design.

## Prerequisites

- Go 1.23+
- Python 3.11+
- Docker and Docker Compose
- ffmpeg
- [goose](https://github.com/pressly/goose) (migration tool)

For ML workers:
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

## Getting started

```bash
# 1. Clone
git clone https://github.com/launchstack/cast.git
cd cast

# 2. Copy env and edit as needed
cp .env.example .env

# 3. Start Postgres
make db-up

# 4. Run migrations
make migrate-up

# 5. Install Python dependencies (for ML workers)
cd python
uv venv
uv pip install -e ".[dev]"
cd ..

# 6. Run the API server
make run-api

# 7. Open the frontend prototype in a browser
#    http://localhost:8080/

# 8. (separate terminal) Submit a video
curl -s -X POST http://localhost:8080/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"source_url": "https://example.com/video.mp4"}' | jq .

# 9. Run tests
make test-go
```

The API also serves a single-file React prototype of the UI at `/` (redirects to `/Cast.html`). See [web/README.md](web/README.md) for what's there and how it maps to future backend endpoints.

## Project structure

```
cmd/cast-api/        API server entrypoint
cmd/cast-worker/     Background job runner
internal/
  api/               HTTP handlers + middleware
  config/            Env-driven configuration
  db/                Postgres pool + queries
  ingest/            ffmpeg + scene detection
  pyworker/          Go ↔ Python subprocess contract
  perception/        ML worker orchestration
  identity/          Face-embedding clustering
  graph/             Entity state extraction + writes
  llm/               LLM API clients
  retrieval/         Hybrid search + RRF
  pipeline/          End-to-end job orchestration
python/workers/      ML inference scripts
migrations/          SQL migrations (goose)
```

## License

MIT
