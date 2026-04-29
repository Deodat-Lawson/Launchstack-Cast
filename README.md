# DroneSearch + RescueRank

A multimodal person retrieval and triage engine over aerial drone video.
Final project for **JHU EN.601.466/666 — Information Retrieval and Web Agents (Spring 2026)**, Option 9 (IR over a non-text modality).

DroneSearch treats every detected person in a drone's video stream as a *document* and builds an IR system over that corpus: visual feature extraction (the tokenizer analog), an inverted index over discrete attributes, a dense FAISS index over CLIP embeddings, TF-IDF weighting, hybrid Boolean+ranked retrieval, Rocchio relevance feedback, DBSCAN dedup, and a Bayesian triage classifier. A standing-query watcher (RescueRank) fires alerts when a query target reappears in the stream — the autonomous-decision layer.

See [docs/PROPOSAL.md](docs/PROPOSAL.md) for the full proposal and [docs/PLAN.md](docs/PLAN.md) for architecture.

## Quickstart

Requires Python 3.11 and [uv](https://github.com/astral-sh/uv).

```bash
make install                                  # uv venv + uv pip install -e ".[dev]"
make ingest VIDEO=data/raw/sample.mp4         # YOLOv8 + ByteTrack + CLIP -> Documents parquet
make app                                      # Streamlit UI
make test                                     # pytest
```

The first `make ingest` will download YOLOv8 weights and CLIP ViT-B/32 weights to your local cache.

## Layout

```
src/drone_search/
  ingest.py       YOLOv8 + ByteTrack person detection
  embed.py        CLIP image + text encoder
  document.py     Per-detection Document record
  index/          dense (FAISS), inverted (TF-IDF), identity (centroids)
  retrieve.py     hybrid scoring + Rocchio RF
  cluster.py      DBSCAN dedup
  triage.py       Bayesian RescueRank
  agent.py        standing-query watcher
  eval/metrics.py P@k, MAP, NDCG, bootstrap CIs
app/streamlit_app.py
tests/
docs/PROPOSAL.md  full course proposal
```

Cast (the previous incarnation of this repo, a Go+Postgres video retrieval system) is preserved on the `cast-archive` branch.

## License

MIT
