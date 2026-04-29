# DroneSearch — Architecture

This is a one-page overview. See [PROPOSAL.md](PROPOSAL.md) for full project framing.

## Stack

- Python 3.11, single `pyproject.toml`.
- `ultralytics` (YOLOv8 + ByteTrack), `open-clip-torch`, `faiss-cpu`, `scikit-learn`, `streamlit`, `whoosh`, `typer`.
- Single workstation, one consumer GPU. No Postgres, no cloud.

## Layered design (mirrors proposal §3)

```
Layer 4 — Agent       retrieve.py + agent.py + triage.py
Layer 3 — Index       index/dense.py + index/inverted.py + index/identity.py
Layer 2 — Embed/Tag   embed.py + (attribute tagger; week 5)
Layer 1 — Ingest      ingest.py (YOLOv8 + ByteTrack)
```

Data flows top to bottom; agent decisions flow back up.

## Rubric mapping (course-grader-facing)

| Rubric concept                | File                                |
|-------------------------------|-------------------------------------|
| Tokenization                  | `ingest.py` (detector as tokenizer) |
| Inverted index, IDF           | `index/inverted.py`                 |
| Vector model + centroid       | `index/identity.py`                 |
| Cosine similarity, ranked     | `index/dense.py` (FAISS)            |
| Boolean + ranked hybrid       | `retrieve.py:hybrid_score`          |
| Rocchio relevance feedback    | `retrieve.py:rocchio`               |
| Bayesian classifier           | `triage.py`                         |
| Document clustering           | `cluster.py` (DBSCAN)               |
| Cross-modal IR                | `embed.py` (text → image via CLIP)  |
| Routing/filtering             | `agent.py` (standing-query watcher) |
| Evaluation                    | `eval/metrics.py` + `eval/queries/` |

## Week-1 scope

Working: `ingest.py`, `embed.py`, `document.py`, the CLI in `__main__.py`, the Streamlit feature browser. Everything else is stubbed (raises `NotImplementedError`) so call sites planned for later weeks can compile-import without blocking.
