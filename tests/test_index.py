"""Dense FAISS + inverted index unit tests."""
from __future__ import annotations

import math

import numpy as np

from drone_search.document import Document
from drone_search.index import dense as dense_idx
from drone_search.index.inverted import (
    boolean_search,
    build_index as build_inverted,
    score_tags,
)


def _make_embeddings(n: int, d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((n, d)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def test_dense_recall_self() -> None:
    embs = _make_embeddings(50, d=8)
    idx = dense_idx.build_index(embs)
    scores, ids = dense_idx.search(idx, embs, k=1)

    # Each embedding's top-1 should be itself with score ≈ 1.
    assert ids.shape == (50, 1)
    np.testing.assert_array_equal(ids[:, 0], np.arange(50))
    np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-3)


def test_dense_empty_index() -> None:
    idx = dense_idx.build_index(np.zeros((0, 8), dtype=np.float32))
    q = _make_embeddings(2, d=8)
    scores, ids = dense_idx.search(idx, q, k=5)
    assert scores.shape == (2, 5)
    assert (ids == -1).all()


def _doc(idx: int, tags: list[str]) -> Document:
    return Document(frame_id=idx, t=float(idx), bbox=(0, 0, 1, 1), track_id=idx, det_conf=1.0, tags=tags)


def test_inverted_postings_and_idf() -> None:
    docs = [
        _doc(0, ["red_jacket", "walking"]),
        _doc(1, ["red_jacket", "running"]),
        _doc(2, ["blue_jacket", "walking"]),
        _doc(3, []),
    ]
    inv = build_inverted(docs)
    assert inv.n_docs == 4
    assert inv.postings["red_jacket"] == {0, 1}
    assert inv.postings["walking"] == {0, 2}

    # IDF(red_jacket) > IDF(walking) only if red_jacket is rarer; both appear in 2/4 docs here,
    # so they should be equal. Sanity-check the formula with a rarer term.
    assert math.isclose(inv.idf["red_jacket"], inv.idf["walking"])
    assert inv.idf["running"] > inv.idf["red_jacket"]


def test_inverted_boolean_modes() -> None:
    docs = [
        _doc(0, ["a", "b"]),
        _doc(1, ["a"]),
        _doc(2, ["b"]),
    ]
    inv = build_inverted(docs)
    assert boolean_search(inv, ["a", "b"], mode="and") == {0}
    assert boolean_search(inv, ["a", "b"], mode="or") == {0, 1, 2}
    assert boolean_search(inv, [], mode="or") == set()


def test_score_tags_picks_matched_idf() -> None:
    docs = [
        _doc(0, ["red_jacket", "walking"]),
        _doc(1, ["walking"]),
        _doc(2, ["walking"]),
    ]
    inv = build_inverted(docs)
    s = score_tags(inv, ["red_jacket", "running"], ["red_jacket", "walking"])
    # Only `red_jacket` matches; `running` is unknown → 0 contribution.
    assert math.isclose(s, inv.idf["red_jacket"])
