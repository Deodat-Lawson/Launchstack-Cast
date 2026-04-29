"""Document + FrameCaption parquet roundtrip."""
from __future__ import annotations

import numpy as np

from drone_search.document import (
    Document,
    FrameCaption,
    frames_from_parquet,
    frames_to_parquet,
    from_parquet,
    to_parquet,
)


def test_document_roundtrip(tmp_path) -> None:
    emb = np.random.default_rng(0).standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    docs = [
        Document(
            frame_id=10,
            t=0.5,
            bbox=(1, 2, 30, 40),
            track_id=4,
            det_conf=0.9,
            embedding=emb,
            tags=["red_jacket", "walking"],
            region_weights={"head": 1.0, "torso": 0.8},
            description="a person in a red jacket walking",
        ),
        Document(frame_id=20, t=1.5, bbox=(0, 0, 10, 10), track_id=7, det_conf=0.7),
    ]
    p = tmp_path / "docs.parquet"
    to_parquet(docs, p)
    back = from_parquet(p)

    assert len(back) == 2
    assert back[0].frame_id == 10
    assert back[0].tags == ["red_jacket", "walking"]
    assert back[0].description == "a person in a red jacket walking"
    assert back[0].region_weights == {"head": 1.0, "torso": 0.8}
    assert back[0].embedding is not None
    np.testing.assert_allclose(back[0].embedding, emb, rtol=1e-5)
    assert back[1].embedding is None
    assert back[1].description == ""


def test_frame_caption_roundtrip(tmp_path) -> None:
    emb = np.random.default_rng(1).standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    frames = [
        FrameCaption(
            frame_id=0,
            t=0.0,
            scene_description="two people walking on a street",
            scene_tags=["urban_street", "two_people"],
            scene_embedding=emb,
        ),
        FrameCaption(frame_id=30, t=1.0),
    ]
    p = tmp_path / "frames.parquet"
    frames_to_parquet(frames, p)
    back = frames_from_parquet(p)

    assert len(back) == 2
    assert back[0].scene_description == "two people walking on a street"
    assert back[0].scene_tags == ["urban_street", "two_people"]
    assert back[0].scene_embedding is not None
    np.testing.assert_allclose(back[0].scene_embedding, emb, rtol=1e-5)
    assert back[1].scene_embedding is None
    assert back[1].scene_description == ""
