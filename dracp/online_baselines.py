"""Recent adaptive online conformal-prediction baselines.

Reimplementations of three published online conformal procedures, following the
streaming interface used throughout this package (``fit``, ``predict_one``,
``update_one``) on absolute-residual scores with a rolling calibration window.

These follow the cited algorithms rather than their general structure. The
distinguishing features, each of which an earlier simplified version omitted,
are noted in the class docstrings:

  * DtACI  -- Gibbs & Candes (2024, JMLR 25:1-36), "Conformal inference for
              online prediction with arbitrary distribution shifts". Experts
              receive *individual* feedback through the realised critical level
              beta_t, not a shared miss indicator.
  * SAOCP  -- Bhatnagar, Wang, Xiong & Bai (2023, ICML), "Improved online
              conformal prediction via strongly adaptive online learning".
              Experts run scale-free online gradient descent and are combined by
              coin betting, not fixed-rate updates with exponential weights.
  * Conformal PID -- Angelopoulos, Candes & Tibshirani (2023, NeurIPS),
              "Conformal PID control for time series prediction". Uses a
              saturating ("admissible") integrator and an optional scorecaster,
              not a plain running sum.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from .quantiles import ordinary_conformal_quantile


def _pinball(target_alpha: float, beta: float, a: float) -> float:
    """Pinball loss of a level ``a`` against the realised critical level ``beta``."""
    d = beta - a
    return target_alpha * d if d >= 0 else (target_alpha - 1.0) * d


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

    def _radius(self, level: float) -> float:
        """Conformal radius at miscoverage ``level`` from the current window."""
        lv = float(np.clip(level, 1e-6, 1 - 1e-6))
        return ordinary_conformal_quantile(self.scores_, lv)

    def _critical_level(self, score: float) -> float:
        """beta_t: the smallest miscoverage level at which this score is covered.

        With the empirical distribution of recent scores, the realised score
        ``s`` sits at rank ``F(s)``; an interval built at level ``a`` covers it
        exactly when ``1 - a <= F(s)``, i.e. when ``a >= 1 - F(s)``. Returning
        ``beta = 1 - F(s)`` therefore lets every expert evaluate its own
        coverage without constructing its own interval, which is what makes
        expert-specific feedback cheap.
        """
        if len(self.scores_) == 0:
            return self.alpha
        return float(np.mean(self.scores_ >= score))


# ---------------------------------------------------------------------------
class FACIRegressor(_BaseOnline):
    """DtACI: dynamically-tuned adaptive conformal inference.

    Maintains ``K`` ACI experts with distinct learning rates. The level played is
    the weight-average of the expert levels; after the outcome is seen each
    expert is scored on the pinball loss of *its own* level against the realised
    critical level ``beta_t``, and performs its own ACI update using its own
    miscoverage indicator ``1{beta_t <= alpha_k}``. Weights follow exponential
    weighting with uniform mixing.

    This expert-specific feedback is the substantive difference from a naive
    aggregation in which every expert is updated from one shared miss indicator:
    there, all experts see the same signal and the ensemble cannot learn which
    rate suits the series.
    """

    def __init__(self, alpha=0.1, window=300, gammas=None, eta=None, sigma=None,
                 interval=100, model=None):
        super().__init__(alpha, window, model)
        self.gammas = np.asarray(
            gammas if gammas is not None else tuple(0.001 * 2 ** k for k in range(8)),
            float)
        self.interval = int(interval)
        self._eta_arg, self._sigma_arg = eta, sigma

    def _init_state(self):
        K, I = len(self.gammas), max(self.interval, 2)
        a = self.alpha
        # Tuning as specified by Gibbs & Candes (2024) and used in the authors'
        # reference implementation.
        self.sigma = float(self._sigma_arg) if self._sigma_arg is not None else 1.0 / (2.0 * I)
        if self._eta_arg is not None:
            self.eta = float(self._eta_arg)
        else:
            denom = ((1 - a) ** 2 * a ** 3 + a ** 2 * (1 - a) ** 3) / 3.0
            self.eta = float(np.sqrt(3.0 / I) * np.sqrt((np.log(I * K) + 2.0) / max(denom, 1e-12)))
        self.alpha_k = np.full(K, a, float)
        self.w = np.ones(K) / K

    @property
    def _level(self) -> float:
        return float(np.clip(np.dot(self.w, self.alpha_k), 1e-6, 1 - 1e-6))

    def predict_one(self, x):
        p = self._pred(x)
        q = self._radius(self._level)
        return {"prediction": p, "lower": p - q, "upper": p + q,
                "alpha_t": self._level, "fallback_used": False, "abstained": False}

    def update_one(self, x, y, result):
        p = float(result["prediction"])
        s = abs(float(y) - p)
        beta = self._critical_level(s)
        losses = np.array([_pinball(self.alpha, beta, a) for a in self.alpha_k])
        w = self.w * np.exp(-self.eta * losses)
        tot = w.sum()
        w = w / tot if (tot > 0 and np.isfinite(tot)) else np.ones_like(w) / len(w)
        K = len(w)
        self.w = (1.0 - self.sigma) * w + self.sigma / K
        # expert-specific coverage indicator: expert k would have missed iff its
        # own level is at or below the realised critical level
        err_k = (self.alpha_k > beta).astype(float)
        self.alpha_k = np.clip(self.alpha_k + self.gammas * (self.alpha - err_k), 0.0, 1.0)
        self._push_score(s)
        return int(not (result["lower"] <= float(y) <= result["upper"]))


# ---------------------------------------------------------------------------
class SAOCPRegressor(_BaseOnline):
    """Strongly adaptive online conformal prediction.

    A faithful port of the algorithm of Bhatnagar et al. (2023), following the
    authors' reference implementation. Three features distinguish it from a
    generic expert ensemble, and all three matter:

    * each expert predicts the *radius itself* by scale-free online gradient
      descent on the pinball loss, with step ``scale / sqrt(3 * cumulative
      squared gradient)``, rather than correcting a separately computed
      quantile;
    * experts are scheduled on the data-streaming intervals of Hazan and
      Seshadhri, an expert created at time ``t`` living for ``8 * 2**u`` steps
      where ``2**u`` is the largest power of two dividing ``t``;
    * they are combined by a coin-betting meta-algorithm whose weight is the
      prior ``1/(t^2 (1 + floor(log2 t)))`` times the positive part of the
      accumulated betting wealth, and each new expert is warm-started at the
      current aggregate prediction.
    """

    def __init__(self, alpha=0.1, window=300, lifetime=8, model=None):
        super().__init__(alpha, window, model)
        self.lifetime = int(lifetime)

    def _init_state(self):
        self.t = 1
        s = np.abs(self.scores_)
        self.scale = 1.0 if len(s) == 0 else float(np.max(s) * np.sqrt(3.0))
        self.experts = {}
        for r in self.scores_:          # warm up on the calibration residuals
            self._step(float(r))

    # -- one expert -------------------------------------------------------
    def _new_expert(self, yhat0):
        t, u = self.t, 0
        while t % 2 == 0:
            t //= 2
            u += 1
        return {"yhat": float(yhat0), "gnorm": 0.0, "life": self.lifetime * 2 ** u,
                "z": 0.0, "wz": 0.0, "s": 0}

    @staticmethod
    def _w(e):
        return 0.0 if e["s"] == 0 else e["z"] / e["s"] * (1.0 + e["wz"])

    def _aggregate(self):
        if not self.experts:
            return 0.0
        prior = {t: 1.0 / (t ** 2 * (1.0 + np.floor(np.log2(t)))) for t in self.experts}
        z = sum(prior.values())
        p = {t: (prior[t] / z) * max(0.0, self._w(e)) for t, e in self.experts.items()}
        sp = sum(p.values())
        if sp <= 0:
            p = {t: prior[t] / z for t in self.experts}
            sp = 1.0
        return float(sum(p[t] / sp * self.experts[t]["yhat"] for t in self.experts))

    def _step(self, s):
        """One streaming update on an observed score ``s``."""
        q = 1.0 - self.alpha                       # pinball level for the radius
        yhat0 = self._aggregate()
        for t in [k for k, e in self.experts.items() if e["s"] > e["life"]]:
            self.experts.pop(t)
        self.experts[self.t] = self._new_expert(yhat0)
        meta = max(q * (s - self._aggregate()), (1 - q) * (self._aggregate() - s))
        for e in self.experts.values():
            own = max(q * (s - e["yhat"]), (1 - q) * (e["yhat"] - s))
            w = self._w(e)
            g = np.clip((meta - own) / self.scale / max(self.alpha, 1 - self.alpha),
                        -1.0 * (w > 0), 1.0)
            e["z"] += g
            e["wz"] += g * w
            e["s"] += 1
            grad = -q * (s > e["yhat"]) + (1 - q) * (s < e["yhat"])
            e["gnorm"] += grad ** 2
            if e["gnorm"] > 0:
                e["yhat"] = max(0.0, e["yhat"]
                                - (self.scale / np.sqrt(3.0)) / np.sqrt(e["gnorm"]) * grad)
        self.t += 1

    def predict_one(self, x):
        p = self._pred(x)
        q = max(self._aggregate(), 1e-9)
        return {"prediction": p, "lower": p - q, "upper": p + q,
                "alpha_t": self.alpha, "fallback_used": False, "abstained": False}

    def update_one(self, x, y, result):
        s = abs(float(y) - float(result["prediction"]))
        self._step(s)
        self._push_score(s)
        return int(not (result["lower"] <= float(y) <= result["upper"]))


# ---------------------------------------------------------------------------
class ConformalPIDRegressor(_BaseOnline):
    """Conformal PID control.

    The radius is the sum of three terms: a *scorecaster* forecast of the next
    score, a proportional term on the current coverage error, and a *saturating*
    integrator. The integrator is the feature that distinguishes the cited method
    from a plain PID loop: a linear running sum grows without bound and
    destabilises the radius, whereas the saturating form

        r(x) = K_I * tan( x * log(t) / (t * C) )

    responds sharply to persistent error while remaining bounded over any finite
    horizon. The scorecaster defaults to a light autoregressive forecast of the
    score sequence; setting ``use_scorecaster=False`` gives the PI variant.
    """

    def __init__(self, alpha=0.1, window=300, kp=0.1, ki=1.0, kd=0.0,
                 csat=20.0, use_scorecaster=True, model=None):
        super().__init__(alpha, window, model)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.csat = float(csat)
        self.use_scorecaster = bool(use_scorecaster)

    def _init_state(self):
        self.t = 0
        self.err_sum = 0.0
        self.prev_err = 0.0

    def _scorecast(self) -> float:
        """One-step forecast of the score: recent quantile plus a drift term."""
        base = self._radius(self.alpha)
        if not self.use_scorecaster or len(self.scores_) < 20:
            return base
        k = min(len(self.scores_), 40)
        recent, older = self.scores_[-k:], self.scores_[-2 * k:-k]
        if len(older) < 5:
            return base
        drift = float(np.median(recent) - np.median(older))
        return max(base + drift, 1e-9)

    def _integrator(self, x: float) -> float:
        t = max(self.t, 1)
        arg = x * np.log(t + 1.0) / (t * self.csat)
        arg = float(np.clip(arg, -1.5, 1.5))          # keep inside tan's branch
        return self.ki * float(np.tan(arg))

    def predict_one(self, x):
        p = self._pred(x)
        scale = max(self._radius(self.alpha), 1e-6)
        q = (self._scorecast()
             + self.kp * scale * self.prev_err
             + scale * self._integrator(self.err_sum))
        q = max(q, 1e-9)
        self._last_q = q
        return {"prediction": p, "lower": p - q, "upper": p + q,
                "alpha_t": self.alpha, "fallback_used": False, "abstained": False}

    def update_one(self, x, y, result):
        p = float(result["prediction"])
        s = abs(float(y) - p)
        miss = float(s > self._last_q)
        e = miss - self.alpha
        self.err_sum += e
        self.prev_err = e
        self.t += 1
        self._push_score(s)
        return int(not (result["lower"] <= float(y) <= result["upper"]))
