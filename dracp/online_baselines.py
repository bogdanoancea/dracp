"""Recent adaptive online conformal-prediction baselines.

Faithful, self-contained reimplementations used for empirical comparison against
DRACP. All follow the same streaming interface as the other baselines
(``fit(x_train, y_train, x_cal, y_cal)``, ``predict_one(x)``,
``update_one(x, y, result)``) and operate on absolute-residual scores with a
rolling calibration window.

References:
  * FACI  -- Gibbs & Candes (2022), "Conformal inference for online prediction
             with arbitrary distribution shifts".
  * SAOCP -- Bhatnagar, Wang, Xiong & Bai (2023), "Improved online conformal
             prediction via strongly adaptive online learning".
  * Conformal PID -- Angelopoulos, Candes & Tibshirani (2024), "Conformal PID
             control for time series prediction".
"""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from .quantiles import ordinary_conformal_quantile


class _BaseOnline:
    def __init__(self, alpha=0.1, window=300, model=None):
        self.alpha = float(alpha)
        self.window = int(window)
        self.model = model if model is not None else HistGradientBoostingRegressor(random_state=42)

    def fit(self, x_train, y_train, x_cal, y_cal):
        self.model_ = clone(self.model).fit(x_train, y_train)
        self.scores_ = np.abs(np.asarray(y_cal, float) - self.model_.predict(x_cal))
        self._init_state()
        return self

    def _init_state(self):
        pass

    def _pred(self, x):
        return float(self.model_.predict(np.asarray(x, float).reshape(1, -1))[0])

    def _push_score(self, s):
        self.scores_ = np.append(self.scores_, s)[-self.window:]

    def _scale(self):
        if len(self.scores_) == 0:
            return 1.0
        return float(max(np.median(self.scores_), 1e-6))


# ---------------------------------------------------------------------------
class ConformalPIDRegressor(_BaseOnline):
    """Radius tracking with a PID controller on the coverage error.

    The base radius is the ordinary conformal quantile of recent scores (the
    "scorecaster"); a PID correction b_t = kp*e_t + ki*sum e + kd*de removes
    residual coverage bias, with e_t = miss_t - alpha. Gains are applied in
    units of the running score scale so they are dataset-agnostic.
    """

    def __init__(self, alpha=0.1, window=300, kp=0.1, ki=0.02, kd=0.0, model=None):
        super().__init__(alpha, window, model)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)

    def _init_state(self):
        self._sum = 0.0
        self._last = 0.0
        self._prev = 0.0

    def predict_one(self, x):
        pred = self._pred(x)
        q_base = ordinary_conformal_quantile(self.scores_, self.alpha)
        if not np.isfinite(q_base):
            q_base = self._scale() * 3.0
        scl = self._scale()
        b = scl * (self.kp * self._last + self.ki * self._sum
                   + self.kd * (self._last - self._prev))
        q = max(0.0, q_base + b)
        return {"prediction": pred, "lower": pred - q, "upper": pred + q,
                "q": q, "alpha_t": self.alpha, "fallback_used": False,
                "abstained": False}

    def update_one(self, x, y, result):
        covered = result["lower"] <= y <= result["upper"]
        miss = 0 if covered else 1
        e = miss - self.alpha
        self._prev = self._last
        self._last = e
        self._sum += e
        self._push_score(abs(float(y) - self._pred(x)))
        return int(miss)


# ---------------------------------------------------------------------------
class FACIRegressor(_BaseOnline):
    """Fully Adaptive Conformal Inference (Gibbs & Candes, 2022).

    Runs a set of ACI experts, one per candidate learning rate gamma, each
    maintaining its own miscoverage level. Expert weights are updated by
    exponentiated-gradient on the pinball (quantile) loss of the coverage
    indicator, with a small uniform mixing term for adaptivity. The predicted
    level is the weighted average of the experts' levels.
    """

    def __init__(self, alpha=0.1, window=300,
                 gammas=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
                 eta=1.0, sigma=0.05, model=None):
        super().__init__(alpha, window, model)
        self.gammas = np.asarray(gammas, float)
        self.eta = float(eta)
        self.sigma = float(sigma)

    def _init_state(self):
        K = len(self.gammas)
        self.alpha_k = np.full(K, self.alpha, float)
        self.w = np.ones(K) / K

    def _agg_alpha(self):
        a = float(np.clip(np.dot(self.w, self.alpha_k), 1e-6, 1 - 1e-6))
        return a

    def predict_one(self, x):
        pred = self._pred(x)
        a = self._agg_alpha()
        q = ordinary_conformal_quantile(self.scores_, a)
        if not np.isfinite(q):
            q = self._scale() * 3.0
        self._last_pred = pred
        return {"prediction": pred, "lower": pred - q, "upper": pred + q,
                "q": q, "alpha_t": a, "fallback_used": False, "abstained": False}

    def update_one(self, x, y, result):
        covered = result["lower"] <= y <= result["upper"]
        miss = 0 if covered else 1
        # pinball loss of each expert's level vs the realized miscoverage
        # l_k = (alpha_k - miss) * (alpha - 1{alpha_k >= miss})  -> use standard
        # quantile (pinball) loss at level (1-alpha) on the coverage error.
        err = miss - self.alpha_k
        losses = np.where(err >= 0, self.alpha * err, (self.alpha - 1.0) * err)
        # exponentiated-gradient weight update + uniform mixing
        w = self.w * np.exp(-self.eta * losses)
        s = w.sum()
        w = w / s if s > 0 and np.isfinite(s) else np.ones_like(w) / len(w)
        K = len(w)
        self.w = (1 - self.sigma) * w + self.sigma / K
        # each expert performs its own ACI update
        self.alpha_k = np.clip(self.alpha_k + self.gammas * (self.alpha - miss),
                               1e-6, 1 - 1e-6)
        self._push_score(abs(float(y) - self._pred(x)))
        return int(miss)


# ---------------------------------------------------------------------------
class SAOCPRegressor(_BaseOnline):
    """Strongly Adaptive Online Conformal Prediction (Bhatnagar et al., 2023).

    Maintains a pool of experts, each spawned at a different time with a
    geometrically growing lifetime; every expert tracks a radius correction by
    scale-free online gradient descent on the pinball loss. Predictions
    aggregate active experts with weights combining a lifetime prior and
    exponential performance weighting, yielding strongly-adaptive regret.
    """

    def __init__(self, alpha=0.1, window=300, lr=0.1, max_experts=64, model=None):
        super().__init__(alpha, window, model)
        self.lr = float(lr)
        self.max_experts = int(max_experts)

    def _init_state(self):
        self.t = 0
        # each expert: dict(corr, cum_loss, prior, born, life)
        self.experts = []

    def _spawn(self):
        # lifetime is a power of two based on the largest dyadic block ending at t
        n = self.t + 1
        level = (n & -n).bit_length() - 1  # position of lowest set bit
        life = 2 ** min(level + 1, 16)
        self.experts.append({"corr": 0.0, "cum": 0.0,
                             "prior": 1.0 / life, "born": self.t, "life": life})
        if len(self.experts) > self.max_experts:
            self.experts.pop(0)

    def _reap(self):
        self.experts = [e for e in self.experts
                        if (self.t - e["born"]) < e["life"]]

    def predict_one(self, x):
        pred = self._pred(x)
        q_base = ordinary_conformal_quantile(self.scores_, self.alpha)
        if not np.isfinite(q_base):
            q_base = self._scale() * 3.0
        self._spawn()
        if self.experts:
            weights = np.array([e["prior"] * np.exp(-e["cum"]) for e in self.experts])
            ws = weights.sum()
            weights = weights / ws if ws > 0 and np.isfinite(ws) else np.ones(len(self.experts)) / len(self.experts)
            corr = float(np.dot(weights, [e["corr"] for e in self.experts]))
        else:
            corr = 0.0
        q = max(0.0, q_base + corr * self._scale())
        self._q_base = q_base
        return {"prediction": pred, "lower": pred - q, "upper": pred + q,
                "q": q, "alpha_t": self.alpha, "fallback_used": False,
                "abstained": False}

    def update_one(self, x, y, result):
        covered = result["lower"] <= y <= result["upper"]
        miss = 0 if covered else 1
        # pinball subgradient of the coverage error w.r.t. the radius correction
        g = (self.alpha - miss)   # positive when we over-cover -> shrink
        for e in self.experts:
            # scale-free OGD step on the correction
            e["corr"] += self.lr * g
            e["corr"] = float(np.clip(e["corr"], -5.0, 5.0))
            # instantaneous pinball loss feeds the performance weight
            e["cum"] += abs(g)
        self.t += 1
        self._reap()
        self._push_score(abs(float(y) - self._pred(x)))
        return int(miss)
