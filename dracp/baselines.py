from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from .controller import AdaptiveAlphaController
from .quantiles import ordinary_conformal_quantile

class SplitConformalRegressor:
    def __init__(self, alpha=0.1, model=None):
        self.alpha = alpha
        self.model = model if model is not None else HistGradientBoostingRegressor(random_state=42)

    def fit(self, x_train, y_train, x_cal, y_cal):
        self.model_ = clone(self.model).fit(x_train, y_train)
        self.scores_ = np.abs(y_cal - self.model_.predict(x_cal))
        return self

    def predict_one(self, x):
        pred = float(self.model_.predict(np.asarray(x).reshape(1, -1))[0])
        q = ordinary_conformal_quantile(self.scores_, self.alpha)
        return {"prediction": pred, "lower": pred-q, "upper": pred+q,
                "alpha_t": self.alpha, "fallback_used": False,
                "abstained": False}

    def update_one(self, x, y, result):
        return int(not (result["lower"] <= y <= result["upper"]))

class RollingConformalRegressor(SplitConformalRegressor):
    def __init__(self, alpha=0.1, window=300, model=None):
        super().__init__(alpha, model)
        self.window = window

    def update_one(self, x, y, result):
        error = super().update_one(x, y, result)
        pred = float(self.model_.predict(np.asarray(x).reshape(1, -1))[0])
        self.scores_ = np.append(self.scores_, abs(y-pred))[-self.window:]
        return error

class AdaptiveConformalRegressor(RollingConformalRegressor):
    def __init__(self, alpha=0.1, window=300, step=0.01, model=None):
        super().__init__(alpha, window, model)
        self.controller = AdaptiveAlphaController(alpha, step)

    def predict_one(self, x):
        pred = float(self.model_.predict(np.asarray(x).reshape(1, -1))[0])
        q = ordinary_conformal_quantile(self.scores_, self.controller.used_alpha)
        return {"prediction": pred, "lower": pred-q, "upper": pred+q,
                "alpha_t": self.controller.used_alpha,
                "fallback_used": False, "abstained": False}

    def update_one(self, x, y, result):
        error = super().update_one(x, y, result)
        self.controller.update(error)
        return error
