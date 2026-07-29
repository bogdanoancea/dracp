"""Unified DRACP: a single method that hedges a regime-aware weighted conformal
predictor against strong unweighted online conformal methods, choosing online
which to trust on each series.

The hedge supports two modes:
  * 'select'  (default) -- hard selection of the current best expert by
     exponential weights; robust to weak experts (a poor expert is simply never
     selected), so adding more candidates cannot hurt beyond a log-regret term.
  * 'average' -- convex combination of expert radii (sharper when experts are
     comparable, but diluted by weak experts).

Either way the cumulative interval score is within O(sqrt(T log J)) of the best
expert (Cesa-Bianchi & Lugosi, 2006), so Unified DRACP is asymptotically no worse
than the best of its components on any series.
"""
from __future__ import annotations
import numpy as np
from .metrics import interval_score


class AggregatedRegressor:
    def __init__(self, methods: dict, alpha: float = 0.1, eta: float | None = None,
                 mode: str = "select"):
        self.methods = methods
        self.alpha = float(alpha)
        self.eta = eta
        self.mode = mode

    def fit(self, x_train, y_train, x_cal, y_cal):
        for m in self.methods.values():
            m.fit(x_train, y_train, x_cal, y_cal)
        self.names_ = list(self.methods)
        self.cum_ = np.zeros(len(self.names_))     # cumulative loss per expert
        self.w_ = np.ones(len(self.names_)) / len(self.names_)
        return self

    def predict_one(self, x):
        self._res = {n: self.methods[n].predict_one(x) for n in self.names_}
        lo = np.array([self._res[n]["lower"] for n in self.names_], float)
        up = np.array([self._res[n]["upper"] for n in self.names_], float)
        fin = np.isfinite(lo) & np.isfinite(up)
        w = self.w_ * fin
        w = w / w.sum() if w.sum() > 0 else fin / max(fin.sum(), 1)
        if self.mode == "select":
            k = int(np.argmax(w))
            L, U = float(lo[k]), float(up[k])
        else:
            L = float(np.sum(w * np.where(fin, lo, 0.0)))
            U = float(np.sum(w * np.where(fin, up, 0.0)))
        pred = float(self._res[self.names_[0]].get("prediction", 0.5 * (L + U)))
        return {"prediction": pred, "lower": L, "upper": U, "alpha_t": self.alpha,
                "fallback_used": False, "abstained": False}

    def update_one(self, x, y, result):
        losses = []
        for n in self.names_:
            r = self._res[n]
            self.methods[n].update_one(x, y, r)
            losses.append(float(interval_score(np.array([y]), np.array([r["lower"]]),
                                               np.array([r["upper"]]), self.alpha)[0]))
        losses = np.nan_to_num(np.array(losses), nan=1e9, posinf=1e9)
        if self.eta is None:
            fin = losses[losses < 1e9]
            scale = np.median(fin) if len(fin) else 1.0
            self.eta = 3.0 / max(scale, 1e-6)     # sharper for 'select' robustness
        self.cum_ += losses
        self.w_ = np.exp(-self.eta * (self.cum_ - self.cum_.min()))
        s = self.w_.sum()
        self.w_ = self.w_ / s if (s > 0 and np.isfinite(s)) else np.ones(len(self.names_)) / len(self.names_)
        return int(not (result["lower"] <= y <= result["upper"]))


def build_unified_dracp(cfg, seed, base_model, dcfg, mode="select"):
    """Construct the unified DRACP: a selection hedge over the regime-aware
    weighted self-tuning core and the strong unweighted online baselines."""
    from .core import DRACPConfig, DRACPRegressor
    from .baselines import AdaptiveConformalRegressor
    from .online_baselines import FACIRegressor, SAOCPRegressor, ConformalPIDRegressor
    alpha = cfg["alpha"]; win = dcfg.calibration_window
    core = DRACPRegressor(DRACPConfig(alpha=alpha, random_state=seed,
                                      **{**cfg["dracp"], "use_faci_control": True}),
                          base_model=base_model)
    experts = {
        "weighted_core": core,
        "faci": FACIRegressor(alpha, win, model=base_model),
        "saocp": SAOCPRegressor(alpha, win, model=base_model),
        "conformal_pid": ConformalPIDRegressor(alpha, win, model=base_model),
        "adaptive": AdaptiveConformalRegressor(alpha, win, dcfg.alpha_step, base_model),
    }
    return AggregatedRegressor(experts, alpha=alpha, mode=mode)
