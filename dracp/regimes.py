from __future__ import annotations
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class CausalRegimeEstimator:
    """Practical probabilistic regime model based on expanding-window GMMs.

    It is a causal approximation: the model is fitted only to observations
    available before the prediction block. For rigorous filtering with an HMM,
    this class can be replaced without changing the DRACP interface.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42,
                 covariance_type: str = "full"):
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type=covariance_type,
            reg_covar=1e-6,
            random_state=random_state,
            n_init=5,
        )

    def fit(self, x: np.ndarray):
        z = self.scaler.fit_transform(np.asarray(x, float))
        self.model.fit(z)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(np.asarray(x, float))
        return self.model.predict_proba(z)
