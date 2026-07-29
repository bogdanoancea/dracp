from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class DensityRatioEstimator:
    """Domain-classification density-ratio estimator."""

    def __init__(self, clip_min: float = 0.05, clip_max: float = 20.0,
                 random_state: int = 42):
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_state = random_state
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=random_state),
        )
        self.source_prior_: float | None = None
        self.target_prior_: float | None = None

    def fit(self, source_x: np.ndarray, target_x: np.ndarray):
        source_x = np.asarray(source_x, dtype=float)
        target_x = np.asarray(target_x, dtype=float)
        x = np.vstack([source_x, target_x])
        y = np.concatenate([np.zeros(len(source_x)), np.ones(len(target_x))])
        self.model.fit(x, y)
        self.source_prior_ = len(source_x) / len(x)
        self.target_prior_ = len(target_x) / len(x)
        return self

    def predict_ratio(self, x: np.ndarray, clipped: bool = True) -> np.ndarray:
        if self.source_prior_ is None or self.target_prior_ is None:
            raise RuntimeError("fit must be called before predict_ratio")
        prob = self.model.predict_proba(np.asarray(x, dtype=float))[:, 1]
        eps = np.finfo(float).eps
        odds = np.clip(prob, eps, 1 - eps) / np.clip(1 - prob, eps, None)
        ratio = odds * self.source_prior_ / self.target_prior_
        if clipped:
            ratio = np.clip(ratio, self.clip_min, self.clip_max)
        return ratio

def temporal_weights(n: int, decay: float) -> np.ndarray:
    if n <= 0:
        return np.empty(0)
    ages = np.arange(n - 1, -1, -1, dtype=float)
    return np.power(decay, ages)

def gaussian_local_weights(x_cal: np.ndarray, x_test: np.ndarray,
                           bandwidth: float) -> np.ndarray:
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    diffs = np.asarray(x_cal, float) - np.asarray(x_test, float)
    sq = np.sum(diffs * diffs, axis=1)
    return np.exp(-0.5 * sq / (bandwidth * bandwidth))

def regime_similarity_weights(pi_cal: np.ndarray, pi_test: np.ndarray,
                              beta: float = 1.0,
                              epsilon: float = 1e-12) -> np.ndarray:
    sim = np.asarray(pi_cal, float) @ np.asarray(pi_test, float)
    return np.power(np.clip(sim, epsilon, None), beta)

def stabilize_product(*parts: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(p, float) for p in parts]
    logw = np.zeros_like(arrays[0])
    for arr in arrays:
        logw += np.log(np.clip(arr, 1e-300, None))
    logw -= np.max(logw)
    return np.exp(logw)

def normalized_finite_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, float)
    total = w.sum()
    if total <= 0:
        return np.full(len(w), 1.0 / max(len(w), 1))
    return w / total
