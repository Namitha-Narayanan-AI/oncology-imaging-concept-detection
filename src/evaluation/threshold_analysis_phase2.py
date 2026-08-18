"""Validation-only threshold selection for Phase 2 binary predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def select_balanced_accuracy_threshold(labels, probabilities) -> tuple[float, float]:
    """Return the validation threshold with best balanced accuracy."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        return 0.5, float("nan")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    scored = [
        (balanced_accuracy_score(labels, probabilities >= threshold), float(threshold))
        for threshold in candidates
    ]
    best_score = max(score for score, _ in scored)
    best_threshold = min(
        (threshold for score, threshold in scored if np.isclose(score, best_score)),
        key=lambda value: (abs(value - 0.5), value),
    )
    return best_threshold, float(best_score)
