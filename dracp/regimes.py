from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class CausalRegimeEstimator:
    """Probabilistic regime model based on expanding-window GMMs.

    Causal approximation: the model is fitted only to observations available
    before the prediction block.

    In addition to the posterior ``predict_proba``, the estimator records a
    first-order transition matrix over regimes, estimated from the *sequence*
    of posteriors on the fitting sample by soft counts,

        P[j, k]  proportional to  sum_t  pi_t[j] * pi_{t+1}[k].

    The transition matrix is what makes multi-step forecasting possible: at a
    horizon of ``h`` periods the regime that matters is the one prevailing at
    the *target* date, not at the forecast origin, and the best available
    forecast of it from the origin is ``pi_t @ P^h``. At ``h = 1`` nothing here
    is used by the default configuration, so one-step behaviour is unchanged.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42,
                 covariance_type: str = "full"):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=n_regimes, covariance_type=covariance_type,
            reg_covar=1e-6, random_state=random_state, n_init=5,
        )
        self.transition_: np.ndarray | None = None

    def fit(self, x: np.ndarray):
        z = self.scaler.fit_transform(np.asarray(x, float))
        self.model.fit(z)
        pi = self.model.predict_proba(z)
        self.transition_ = self._estimate_transition(pi)
        return self

    @staticmethod
    def _estimate_transition(pi: np.ndarray, prior: float = 1.0) -> np.ndarray:
        """Soft-count first-order transition matrix from a posterior sequence.

        ``prior`` adds a Dirichlet smoothing count to every cell, which keeps
        the matrix strictly positive (and therefore its powers well behaved)
        when a regime is rarely visited.
        """
        pi = np.asarray(pi, float)
        k = pi.shape[1]
        if len(pi) < 2:
            return np.full((k, k), 1.0 / k)
        counts = pi[:-1].T @ pi[1:]
        counts = counts + prior
        return counts / counts.sum(axis=1, keepdims=True)

    def propagate(self, pi: np.ndarray, steps: int) -> np.ndarray:
        """Forecast regime posteriors ``steps`` periods ahead.

        Returns ``pi @ P^steps``. With ``steps <= 0`` or no transition matrix
        the input is returned unchanged.
        """
        pi = np.atleast_2d(np.asarray(pi, float))
        if steps <= 0 or self.transition_ is None:
            return pi
        return pi @ np.linalg.matrix_power(self.transition_, int(steps))

    def stationary(self) -> np.ndarray:
        """Stationary distribution of the transition matrix (uniform fallback)."""
        k = self.n_regimes
        if self.transition_ is None:
            return np.full(k, 1.0 / k)
        vals, vecs = np.linalg.eig(self.transition_.T)
        i = int(np.argmin(np.abs(vals - 1.0)))
        v = np.real(vecs[:, i])
        v = np.abs(v)
        s = v.sum()
        return v / s if s > 0 else np.full(k, 1.0 / k)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(np.asarray(x, float))
        return self.model.predict_proba(z)
