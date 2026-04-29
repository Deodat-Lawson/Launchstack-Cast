"""DBSCAN over track centroids for re-observation deduplication.

Maps to proposal §3 Layer 3 "identity index" dedup, document clustering lecture.
Same person seen by the drone in two passes → one cluster.
"""
from __future__ import annotations


def dedupe(centroids, *, eps: float = 0.2, min_samples: int = 2):
    """Cluster (N, D) centroids with cosine-distance DBSCAN; return cluster labels."""
    raise NotImplementedError("week 9 deliverable")
