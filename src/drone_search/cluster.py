"""DBSCAN over track centroids for re-observation deduplication.

Maps to proposal §3 Layer 3 "identity index" dedup, document clustering lecture.
Same person seen by the drone in two passes → one cluster.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drone_search.document import Document


def dedupe(centroids, *, eps: float = 0.2, min_samples: int = 2):
    """Cluster (N, D) centroids with cosine-distance DBSCAN; return cluster labels.

    Labels are int per sklearn convention: -1 = noise (singleton, no neighbors
    within eps), 0..K-1 = clusters. Caller decides whether noise stays as its
    own identity (typical) or gets merged into nearest cluster.
    """
    import numpy as np
    from sklearn.cluster import DBSCAN

    if len(centroids) == 0:
        return np.empty(0, dtype=int)
    X = np.asarray(centroids, dtype=np.float32)
    return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(X)


def merge_tracks(documents: list["Document"], *, eps: float = 0.2) -> tuple[int, int]:
    """Re-label fragmented tracks so the same person across re-entries shares one id.

    For each real track (track_id != -1), compute the L2-normalized mean of its
    crop embeddings. Cluster those centroids with cosine DBSCAN; tracks in the
    same cluster get remapped to the smallest original track_id in the cluster.
    DBSCAN noise points keep their original id. Untracked detections (-1) are
    untouched — they stay -1 and flow through the existing untracked-cap path.

    Mutates `documents` in place. Returns (n_merged_groups, n_tracks_collapsed)
    for logging.
    """
    import numpy as np

    track_ids = sorted({d.track_id for d in documents if d.track_id != -1})
    if len(track_ids) < 2:
        return 0, 0

    embs_by_tid: dict[int, list] = {tid: [] for tid in track_ids}
    for d in documents:
        if d.track_id != -1 and d.embedding is not None:
            embs_by_tid[d.track_id].append(d.embedding)

    centroids = []
    valid_tids: list[int] = []
    for tid in track_ids:
        embs = embs_by_tid[tid]
        if not embs:
            continue
        c = np.mean(np.stack(embs, axis=0), axis=0)
        n = float(np.linalg.norm(c)) or 1.0
        centroids.append((c / n).astype(np.float32))
        valid_tids.append(tid)

    if len(valid_tids) < 2:
        return 0, 0

    labels = dedupe(centroids, eps=eps, min_samples=2)

    cluster_rep: dict[int, int] = {}
    for tid, lbl in zip(valid_tids, labels, strict=True):
        if lbl == -1:
            continue
        if lbl not in cluster_rep or tid < cluster_rep[lbl]:
            cluster_rep[lbl] = tid

    remap: dict[int, int] = {}
    for tid, lbl in zip(valid_tids, labels, strict=True):
        if lbl != -1:
            remap[tid] = cluster_rep[lbl]

    n_collapsed = sum(1 for tid, target in remap.items() if tid != target)
    n_groups = len({lbl for lbl in labels if lbl != -1})

    for d in documents:
        if d.track_id in remap:
            d.track_id = remap[d.track_id]

    return n_groups, n_collapsed
