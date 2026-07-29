from __future__ import annotations
import numpy as np

def weighted_quantile_with_test_mass(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    test_weight: float = 1.0,
) -> float:
    """Weighted conformal quantile including a test-point atom at +infinity."""
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scores.ndim != 1 or weights.ndim != 1 or len(scores) != len(weights):
        raise ValueError("scores and weights must be one-dimensional and aligned")
    if len(scores) == 0:
        return float("inf")
    if np.any(weights < 0) or test_weight < 0:
        raise ValueError("weights must be nonnegative")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    total = float(weights.sum() + test_weight)
    if total <= 0:
        return float("inf")
    if test_weight / total > alpha:
        return float("inf")

    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    w = weights[order] / total
    cumulative = np.cumsum(w)
    target = 1.0 - alpha
    idx = int(np.searchsorted(cumulative, target, side="left"))
    return float(s[idx]) if idx < len(s) else float("inf")

def ordinary_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return float("inf")
    level = np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores)
    return float(np.quantile(scores, min(level, 1.0), method="higher"))
