"""User-facing API: wrap any regressor in DRACP conformal prediction intervals.

Example
-------
>>> from sklearn.ensemble import RandomForestRegressor
>>> from dracp import ConformalForecaster
>>> cf = ConformalForecaster(model=RandomForestRegressor(), alpha=0.1)
>>> cf.fit(X_train, y_train)
>>> lower, upper = cf.predict_interval(X_test)              # static
>>> lower, upper = cf.predict_interval(X_test, y_test)      # sequential (recommended)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .core import DRACPConfig, DRACPRegressor
from .metrics import interval_metrics


class ConformalForecaster:
    """Distribution-free prediction intervals for any point-prediction model.

    Parameters
    ----------
    model : estimator, optional
        Any object with scikit-learn's ``fit``/``predict`` interface. Defaults to
        gradient boosting. This is your forecasting model; DRACP only calibrates it.
    alpha : float
        Target miscoverage, so intervals have nominal coverage ``1 - alpha``.
    calibration_fraction : float
        Share of the fitting data (most recent block) held out for calibration.
    **config
        Any :class:`DRACPConfig` field, e.g. ``calibration_window``, ``n_regimes``,
        ``bandwidth``, ``use_faci_control``, ``asymmetric``.
    """

    def __init__(self, model=None, alpha: float = 0.1,
                 calibration_fraction: float = 0.3, random_state: int = 42, **config):
        self.model = model
        self.alpha = float(alpha)
        self.calibration_fraction = float(calibration_fraction)
        self.random_state = int(random_state)
        self.config = config

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _as_array(X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    # -------------------------------------------------------------------- fit
    def fit(self, X, y):
        """Fit the model and calibrate. Data must be in chronological order."""
        X = self._as_array(X); y = self._as_array(y).ravel()
        n = len(y)
        n_cal = max(int(round(self.calibration_fraction * n)), 10)
        if n_cal >= n:
            raise ValueError("calibration_fraction too large for the sample size")
        split = n - n_cal
        cfg = DRACPConfig(alpha=self.alpha, random_state=self.random_state, **self.config)
        self.regressor_ = DRACPRegressor(cfg, base_model=self.model)
        self.regressor_.fit(X[:split], y[:split], X[split:], y[split:])
        return self

    # ---------------------------------------------------------------- predict
    def predict(self, X):
        """Point predictions for ``X`` (the underlying model's forecasts)."""
        X = self._as_array(X)
        return np.array([self.regressor_.predict_one(x)["prediction"] for x in X])

    def predict_interval(self, X, y=None):
        """Prediction intervals for ``X``.

        If the realised outcomes ``y`` are supplied, they are revealed one step at a
        time — after each prediction — so the calibration adapts online, which is the
        intended use for sequential forecasting. Without ``y`` the calibration set is
        held fixed. Returns ``(lower, upper)``.
        """
        X = self._as_array(X)
        y = None if y is None else self._as_array(y).ravel()
        lo, up, rows = [], [], []
        for i, x in enumerate(X):
            r = self.regressor_.predict_one(x)
            lo.append(r["lower"]); up.append(r["upper"])
            if y is not None:
                self.regressor_.update_one(x, float(y[i]), r)
                r["y"] = float(y[i]); rows.append(r)
        self.last_predictions_ = pd.DataFrame(rows) if rows else None
        return np.array(lo), np.array(up)

    def predict_frame(self, X, y=None):
        """Same as :meth:`predict_interval` but returns a tidy DataFrame."""
        lo, up = self.predict_interval(X, y)
        out = pd.DataFrame({"prediction": self.predict(X) if y is None else
                            self.last_predictions_["prediction"].to_numpy(),
                            "lower": lo, "upper": up})
        if y is not None:
            yy = self._as_array(y).ravel()
            out["y"] = yy
            out["covered"] = (yy >= lo) & (yy <= up)
        return out

    # ------------------------------------------------------------------ score
    def score(self, X, y):
        """Evaluate on ``(X, y)``: coverage, mean width and mean interval score."""
        self.predict_interval(X, y)
        m = interval_metrics(self.last_predictions_, self.alpha)
        return {"coverage": m["coverage"], "mean_width": m["mean_width"],
                "mean_interval_score": m["mean_interval_score"]}
