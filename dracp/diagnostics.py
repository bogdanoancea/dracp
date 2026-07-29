from __future__ import annotations
import numpy as np

def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    return float(1.0 / np.sum(p * p))

def weight_entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, float)
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))

def max_normalized_weight(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.max(weights) / total) if total > 0 else 1.0

def perturbation_proxies(
    raw_ratio: np.ndarray,
    clipped_ratio: np.ndarray,
    local_weights: np.ndarray,
    regime_weights: np.ndarray,
    ess: float,
    bandwidth: float,
) -> dict[str, float]:
    """Observable proxies, not estimators of unidentifiable exact theorem terms."""
    denom = max(float(np.sum(np.abs(raw_ratio))), 1e-12)
    b_clip = float(np.sum(np.abs(raw_ratio - clipped_ratio)) / denom)
    b_local = float(bandwidth + 1.0 / np.sqrt(max(ess, 1.0)))
    b_kernel = float(1.0 - np.mean(np.clip(regime_weights, 0.0, 1.0)))
    return {
        "b_ratio_proxy": float(np.std(np.log(np.clip(raw_ratio, 1e-12, None)))),
        "b_clip_proxy": b_clip,
        "b_conditional_proxy": float("nan"),
        "b_kernel_proxy": b_kernel,
        "b_regime_proxy": float(np.mean(-np.log(np.clip(regime_weights, 1e-12, 1.0)))),
        "b_local_proxy": b_local,
        "local_weight_mean": float(np.mean(local_weights)),
    }
