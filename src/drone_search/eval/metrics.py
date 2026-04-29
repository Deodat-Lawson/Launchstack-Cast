"""Standard IR evaluation metrics.

Maps to proposal §5 evaluation plan. Bootstrap CIs over a frozen query set.
"""
from __future__ import annotations


def precision_at_k(retrieved, relevant, k: int):
    raise NotImplementedError("week 4 deliverable")


def mean_average_precision(retrieved_per_query, relevant_per_query):
    raise NotImplementedError("week 4 deliverable")


def ndcg_at_k(retrieved, relevance_grades, k: int):
    raise NotImplementedError("week 7 deliverable")


def bootstrap_ci(metric_values, *, alpha: float = 0.05, n_resamples: int = 1000):
    raise NotImplementedError("week 12 deliverable")
