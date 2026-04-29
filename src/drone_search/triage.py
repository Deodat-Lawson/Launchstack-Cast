"""RescueRank — Bayesian distress-likelihood classifier.

Maps to HW3 Bayesian classifier requirement and proposal §3 Layer 4 triage.
Hand-engineered features: prone_pose, unusual_terrain, isolated, motionless_duration.
"""
from __future__ import annotations


def featurize(documents):
    """Compute hand-engineered triage features per Document."""
    raise NotImplementedError("week 10 deliverable")


def fit(features, labels):
    """Train sklearn GaussianNB or BernoulliNB on (features, labels)."""
    raise NotImplementedError("week 10 deliverable")


def predict_proba(model, features):
    """Return P(distress | features)."""
    raise NotImplementedError("week 10 deliverable")
