"""Per-detection IR Document record + per-frame scene captions + parquet I/O.

A Document is one detected person at one frame. It carries everything the
retrieval and triage layers need: bbox, embedding, attribute tags, an optional
Gemini description, and the metadata that lets us reconstruct identity
(track_id) and timing (frame_id, t).

A FrameCaption is one sampled frame's scene-level summary. The Watch & Ask
tab uses these for moment-centric Q&A; the FAISS scene-embedding index uses
them for "similar moments" search.
"""
from __future__ import annotations

import json
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
    description: str = ""                   # Gemini per-detection caption


@dataclass(slots=True)
class FrameCaption:
    frame_id: int
    t: float
    scene_description: str = ""
    scene_tags: list[str] = field(default_factory=list)
    scene_embedding: np.ndarray | None = None    # (512,) full-frame CLIP


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
                "region_weights": json.dumps(d.region_weights),
                "description": d.description,
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
        embedding = np.asarray(emb, dtype=np.float32) if emb is not None else None
        # `description` was added after week 1; old parquets may not have it.
        description = str(r["description"]) if "description" in r.index and r["description"] is not None else ""
        docs.append(
            Document(
                frame_id=int(r["frame_id"]),
                t=float(r["t"]),
                bbox=(int(r["bbox_x"]), int(r["bbox_y"]), int(r["bbox_w"]), int(r["bbox_h"])),
                track_id=int(r["track_id"]),
                det_conf=float(r["det_conf"]),
                embedding=embedding,
                tags=list(r["tags"]) if r["tags"] is not None else [],
                region_weights=json.loads(r["region_weights"]) if r["region_weights"] else {},
                description=description,
            )
        )
    return docs


def frames_to_parquet(frames: list[FrameCaption], path: str | Path) -> None:
    import pandas as pd

    rows = []
    for f in frames:
        rows.append(
            {
                "frame_id": f.frame_id,
                "t": f.t,
                "scene_description": f.scene_description,
                "scene_tags": list(f.scene_tags),
                "scene_embedding": (
                    f.scene_embedding.astype(np.float32).tolist()
                    if f.scene_embedding is not None
                    else None
                ),
            }
        )
    df = pd.DataFrame(rows)
    df.to_parquet(Path(path), index=False)


def frames_from_parquet(path: str | Path) -> list[FrameCaption]:
    import pandas as pd

    df = pd.read_parquet(Path(path))
    frames: list[FrameCaption] = []
    for _, r in df.iterrows():
        emb = r["scene_embedding"]
        embedding = np.asarray(emb, dtype=np.float32) if emb is not None else None
        frames.append(
            FrameCaption(
                frame_id=int(r["frame_id"]),
                t=float(r["t"]),
                scene_description=str(r["scene_description"]) if r["scene_description"] is not None else "",
                scene_tags=list(r["scene_tags"]) if r["scene_tags"] is not None else [],
                scene_embedding=embedding,
            )
        )
    return frames
