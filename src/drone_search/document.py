"""Per-detection IR Document record + parquet I/O.

A Document is one detected person at one frame. It carries everything the
retrieval and triage layers need: bbox, embedding, attribute tags, and the
metadata that lets us reconstruct identity (track_id) and timing (frame_id, t).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Document:
    frame_id: int
    t: float
    bbox: tuple[int, int, int, int]    # x, y, w, h
    track_id: int
    det_conf: float
    embedding: np.ndarray | None = None     # (512,) float32, L2-normalized
    tags: list[str] = field(default_factory=list)
    region_weights: dict[str, float] = field(default_factory=dict)


def to_parquet(docs: list[Document], path: str | Path) -> None:
    """Persist Documents as a parquet table for caching feature extraction."""
    import pandas as pd

    rows = []
    for d in docs:
        rows.append(
            {
                "frame_id": d.frame_id,
                "t": d.t,
                "bbox_x": d.bbox[0],
                "bbox_y": d.bbox[1],
                "bbox_w": d.bbox[2],
                "bbox_h": d.bbox[3],
                "track_id": d.track_id,
                "det_conf": d.det_conf,
                "embedding": (
                    d.embedding.astype(np.float32).tolist() if d.embedding is not None else None
                ),
                "tags": list(d.tags),
                "region_weights": dict(d.region_weights),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(Path(path), index=False)


def from_parquet(path: str | Path) -> list[Document]:
    import pandas as pd

    df = pd.read_parquet(Path(path))
    docs: list[Document] = []
    for _, r in df.iterrows():
        emb = r["embedding"]
        if emb is None:
            embedding = None
        else:
            embedding = np.asarray(emb, dtype=np.float32)
        docs.append(
            Document(
                frame_id=int(r["frame_id"]),
                t=float(r["t"]),
                bbox=(int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"])),
                track_id=int(r["track_id"]),
                det_conf=float(r["det_conf"]),
                embedding=embedding,
                tags=list(r["tags"]) if r["tags"] is not None else [],
                region_weights=dict(r["region_weights"]) if r["region_weights"] is not None else {},
            )
        )
    return docs
