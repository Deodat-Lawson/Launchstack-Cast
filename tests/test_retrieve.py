"""hybrid_score + rocchio unit tests."""
from __future__ import annotations

import numpy as np

from drone_search import retrieve
from drone_search.document import Document
from drone_search.index import dense as dense_idx
from drone_search.index.inverted import build_index as build_inverted


def _docs_and_index(n: int = 20, d: int = 16, seed: int = 0) -> tuple[list[Document], np.ndarray]:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, d)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    docs: list[Document] = []
    for i in range(n):
        tags = ["walking"] if i % 2 == 0 else ["running"]
        if i == 5:
            tags = ["red_jacket", "walking"]
        docs.append(
            Document(
                frame_id=i,
                t=float(i),
                bbox=(0, 0, 1, 1),
                track_id=i,
                det_conf=1.0,
                embedding=embs[i],
                tags=tags,
            )
        )
    return docs, embs


def test_hybrid_score_dense_only_recovers_self() -> None:
    docs, embs = _docs_and_index(n=12, d=8)
    di = dense_idx.build_index(embs)
    inv = build_inverted(docs)

    hits = retrieve.hybrid_score(
        embs[3], [], docs, di, inv, alpha=1.0, beta=0.0, k=3,
    )
    assert hits[0].doc_idx == 3
    assert hits[0].dense_score > 0.99


def test_hybrid_score_tag_boost_changes_order() -> None:
    docs, embs = _docs_and_index(n=12, d=8)
    di = dense_idx.build_index(embs)
    inv = build_inverted(docs)

    # Use a query that's far from doc 5 so dense ranking won't put it first;
    # then a strong tag match for `red_jacket` (only doc 5) should pull it up.
    q = embs[0].copy()
    no_tag_hits = retrieve.hybrid_score(q, [], docs, di, inv, alpha=1.0, beta=0.0, k=12)
    tag_hits = retrieve.hybrid_score(q, ["red_jacket"], docs, di, inv, alpha=0.1, beta=10.0, k=12)

    no_tag_top = [h.doc_idx for h in no_tag_hits[:3]]
    tag_top = [h.doc_idx for h in tag_hits[:3]]
    assert 5 in tag_top
    assert tag_top != no_tag_top


def test_rocchio_moves_query_toward_relevant() -> None:
    rng = np.random.default_rng(0)
    q = rng.standard_normal(8).astype(np.float32)
    q /= np.linalg.norm(q)

    relevant = rng.standard_normal((3, 8)).astype(np.float32)
    relevant /= np.linalg.norm(relevant, axis=1, keepdims=True)
    irrelevant = -relevant.copy()

    new_q = retrieve.rocchio(q, relevant, irrelevant)

    # New query should be more similar to the relevant centroid than the old query was.
    rel_centroid = relevant.mean(axis=0)
    rel_centroid /= np.linalg.norm(rel_centroid)
    old_sim = float(q @ rel_centroid)
    new_sim = float(new_q @ rel_centroid)
    assert new_sim > old_sim
    np.testing.assert_allclose(np.linalg.norm(new_q), 1.0, atol=1e-5)


def test_rocchio_handles_empty_feedback() -> None:
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    new_q = retrieve.rocchio(q, np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32))
    np.testing.assert_allclose(new_q, q, atol=1e-6)
