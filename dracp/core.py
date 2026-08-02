from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from .controller import AdaptiveAlphaController, PIDAlphaController, FACIController
from .diagnostics import (
    effective_sample_size, max_normalized_weight, weight_entropy,
    perturbation_proxies,
)
from .quantiles import weighted_quantile_with_test_mass
from .regimes import CausalRegimeEstimator
from .weights import (
    DensityRatioEstimator, gaussian_local_weights, normalized_finite_weights,
    regime_similarity_weights, stabilize_product, temporal_weights,
)

@dataclass
class DRACPConfig:
    alpha: float = 0.10
    calibration_window: int = 300
    target_window: int = 60
    temporal_decay: float = 0.995
    bandwidth: float = 2.0
    adaptive_bandwidth: bool = True
    bandwidth_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    min_ess: float = 30.0
    max_weight: float = 0.25
    ratio_clip_min: float = 0.05
    ratio_clip_max: float = 20.0
    n_regimes: int = 3
    regime_beta: float = 1.0
    alpha_step: float = 0.01
    use_temporal: bool = True
    use_ratio: bool = True
    use_local: bool = True
    use_regime: bool = True
    use_adaptive_alpha: bool = True
    fallback: str = "global"
    test_weight: float = 1.0
    random_state: int = 42
    # --- enhancements (opt-in; defaults reproduce the original method) ---
    use_pid: bool = False        # PID significance controller (adds I,D terms)
    pid_ki: float = 0.001        # integral gain
    pid_kd: float = 0.0          # derivative gain
    asymmetric: bool = False     # signed two-sided scores (adapts to skew)
    use_faci_control: bool = False   # self-tuning FACI significance controller
    faci_gammas: tuple = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
    faci_eta: float = 1.0
    faci_sigma: float = 0.05
    # --- multi-step horizon support -------------------------------------
    # `horizon` is the forecast horizon h the model is being used at: the
    # calibration targets are h periods after their feature rows. It changes
    # nothing at h = 1, so all one-step results reproduce exactly. At h > 1 and
    # with `horizon_adapt` enabled, four adjustments are made, each of which
    # addresses a specific way the one-step design degrades with horizon:
    #
    #   * regime weights are aligned to the *target* period rather than the
    #     forecast origin (calibration rows use the realised posterior h steps
    #     later; the test point uses pi_t @ P^h from the estimated transition
    #     matrix). This is the main fix: at the origin the regime is known but
    #     largely irrelevant to a target h periods away.
    #   * the temporal decay is softened to decay**(1/h), because overlapping
    #     h-step targets carry roughly T/h independent observations and a memory
    #     tuned for one-step data discards too much of them.
    #   * the localization bandwidth grid is inflated by sqrt(h) and the ESS
    #     floor raised, because Y_{t+h} | X_t is more diffuse than Y_{t+1} | X_t,
    #     so tight localization buys bias without buying relevance.
    #   * the controller step is damped by 1/sqrt(h), because overlapping errors
    #     make the coverage indicator autocorrelated over h periods and an
    #     undamped controller over-reacts to runs of correlated misses.
    horizon: int = 1
    horizon_adapt: bool = True
    horizon_target_regime: bool = True   # align regime weights to target period
    horizon_max_ess_frac: float = 0.5    # cap on the inflated ESS floor

class DRACPRegressor:
    def _step_damping(self) -> float:
        """Controller step multiplier: 1 at h = 1, 1/sqrt(h) beyond."""
        cfg = self.config
        if not cfg.horizon_adapt or int(cfg.horizon) <= 1:
            return 1.0
        return 1.0 / float(np.sqrt(int(cfg.horizon)))

    def __init__(self, config: DRACPConfig | None = None, base_model=None,
                 scale_model=None):
        self.config = config or DRACPConfig()
        self.base_model = base_model if base_model is not None else HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05,
            random_state=self.config.random_state
        )
        self.scale_model = scale_model if scale_model is not None else RandomForestRegressor(
            n_estimators=150, min_samples_leaf=8, n_jobs=None,
            random_state=self.config.random_state
        )
        # damping applied to every controller's step size (1.0 at h = 1)

        if self.config.use_faci_control:
            d = self._step_damping()
            self.controller = FACIController(
                self.config.alpha, tuple(g * d for g in self.config.faci_gammas),
                self.config.faci_eta, self.config.faci_sigma)
        elif self.config.use_pid:
            self.controller = PIDAlphaController(
                self.config.alpha, self.config.alpha_step * self._step_damping(),
                ki=self.config.pid_ki, kd=self.config.pid_kd)
        else:
            self.controller = AdaptiveAlphaController(
                self.config.alpha, self.config.alpha_step * self._step_damping())
        self.feature_scaler = StandardScaler()
        self._fitted = False

    # -- horizon-dependent effective settings ---------------------------------
    @property
    def _h(self) -> int:
        """Effective horizon; 1 disables every horizon adjustment."""
        cfg = self.config
        if not cfg.horizon_adapt:
            return 1
        return max(1, int(cfg.horizon))

    def _eff_decay(self) -> float:
        h = self._h
        return self.config.temporal_decay if h == 1 else float(
            self.config.temporal_decay ** (1.0 / h))

    def _eff_bandwidth_grid(self) -> tuple:
        h = self._h
        g = self.config.bandwidth_grid
        return g if h == 1 else tuple(float(b) * np.sqrt(h) for b in g)

    def _eff_min_ess(self, n_cal: int) -> float:
        h = self._h
        if h == 1:
            return float(self.config.min_ess)
        cap = self.config.horizon_max_ess_frac * max(n_cal, 1)
        return float(min(self.config.min_ess * h, cap))

    def _eff_target_window(self) -> int:
        h = self._h
        return self.config.target_window if h == 1 else int(
            self.config.target_window * h)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_cal: np.ndarray, y_cal: np.ndarray):
        x_train = np.asarray(x_train, float)
        y_train = np.asarray(y_train, float)
        x_cal = np.asarray(x_cal, float)
        y_cal = np.asarray(y_cal, float)

        self.base_model_ = clone(self.base_model).fit(x_train, y_train)
        train_abs = np.abs(y_train - self.base_model_.predict(x_train))
        self.scale_model_ = clone(self.scale_model).fit(x_train, train_abs + 1e-6)

        self.x_cal_ = x_cal.copy()
        self.y_cal_ = y_cal.copy()
        self.pred_cal_ = self.base_model_.predict(x_cal)
        self.scale_cal_ = np.clip(self.scale_model_.predict(x_cal), 1e-6, None)
        self.scores_ = np.abs(y_cal - self.pred_cal_) / self.scale_cal_
        self.signed_ = (y_cal - self.pred_cal_) / self.scale_cal_

        all_x = np.vstack([x_train, x_cal])
        self.feature_scaler.fit(all_x)
        self.regime_model_ = CausalRegimeEstimator(
            self.config.n_regimes, self.config.random_state
        ).fit(all_x)
        self.pi_cal_ = self.regime_model_.predict_proba(x_cal)
        self._fitted = True
        return self

    def _target_aligned_pi(self, pi_window: np.ndarray) -> np.ndarray:
        """Regime posteriors aligned to the *target* period of each row.

        Calibration row ``i`` carries features from origin ``i`` and a target
        realised ``h`` periods later, so the regime that governed its outcome is
        the one at ``i + h``. For rows where ``i + h`` is inside the window the
        realised posterior is used directly (it is in the past, so this is
        causal). For the final ``h`` rows the realised posterior is not yet
        available and the transition matrix is used to propagate forward.
        """
        h = self._h
        if h <= 1:
            return pi_window
        n = len(pi_window)
        out = np.empty_like(pi_window)
        keep = max(n - h, 0)
        if keep:
            out[:keep] = pi_window[h:h + keep]
        if keep < n:
            tail = pi_window[keep:]
            steps = np.arange(1, n - keep + 1)[::-1]      # h, h-1, ..., 1
            for j, st in enumerate(steps):
                out[keep + j] = self.regime_model_.propagate(
                    tail[j:j + 1], int(st))[0]
        return out

    def _select_bandwidth(self, z_cal, z_test, base_parts):
        if self.config.adaptive_bandwidth:
            candidates = self._eff_bandwidth_grid()
        else:
            candidates = (self.config.bandwidth * (np.sqrt(self._h) if self._h > 1 else 1.0),)
        floor = self._eff_min_ess(len(z_cal))
        chosen = float(candidates[-1])
        chosen_local = np.ones(len(z_cal))
        for bw in candidates:
            local = gaussian_local_weights(z_cal, z_test, float(bw))
            w = stabilize_product(*base_parts, local)
            chosen, chosen_local = float(bw), local
            if effective_sample_size(w) >= floor:
                break
        return chosen, chosen_local

    def predict_one(self, x_t: np.ndarray) -> dict:
        if not self._fitted:
            raise RuntimeError("fit must be called before prediction")
        cfg = self.config
        x_t = np.asarray(x_t, float).reshape(1, -1)
        start = max(0, len(self.x_cal_) - cfg.calibration_window)
        x_cal = self.x_cal_[start:]
        scores = self.scores_[start:]
        signed = self.signed_[start:]
        pi_cal = self.pi_cal_[start:]

        pred = float(self.base_model_.predict(x_t)[0])
        scale = float(max(self.scale_model_.predict(x_t)[0], 1e-6))
        z_cal = self.feature_scaler.transform(x_cal)
        z_t = self.feature_scaler.transform(x_t)[0]

        temporal = temporal_weights(len(x_cal), self._eff_decay()) if cfg.use_temporal else np.ones(len(x_cal))

        raw_ratio = np.ones(len(x_cal))
        clipped_ratio = np.ones(len(x_cal))
        if cfg.use_ratio and len(x_cal) >= 20:
            target_start = max(0, len(x_cal) - self._eff_target_window())
            source_x = x_cal[:max(target_start, 1)]
            target_x = np.vstack([x_cal[target_start:], x_t])
            if len(source_x) >= 10 and len(target_x) >= 5:
                ratio_model = DensityRatioEstimator(
                    cfg.ratio_clip_min, cfg.ratio_clip_max, cfg.random_state
                ).fit(source_x, target_x)
                raw_ratio = ratio_model.predict_ratio(x_cal, clipped=False)
                clipped_ratio = np.clip(raw_ratio, cfg.ratio_clip_min, cfg.ratio_clip_max)

        pi_t_now = self.regime_model_.predict_proba(x_t)[0]
        if self._h > 1 and cfg.horizon_target_regime:
            # forecast the regime at the target date, and compare it with the
            # regime that actually prevailed at each calibration row's target
            pi_t = self.regime_model_.propagate(pi_t_now.reshape(1, -1), self._h)[0]
            pi_ref = self._target_aligned_pi(pi_cal)
        else:
            pi_t = pi_t_now
            pi_ref = pi_cal
        regime = (
            regime_similarity_weights(pi_ref, pi_t, cfg.regime_beta)
            if cfg.use_regime else np.ones(len(x_cal))
        )

        base_parts = [temporal, clipped_ratio, regime]
        if cfg.use_local:
            bandwidth, local = self._select_bandwidth(z_cal, z_t, base_parts)
        else:
            bandwidth, local = cfg.bandwidth, np.ones(len(x_cal))

        weights = stabilize_product(temporal, clipped_ratio, regime, local)
        ess = effective_sample_size(weights)
        pmax = max_normalized_weight(weights)
        supported = ess >= self._eff_min_ess(len(x_cal)) and pmax <= cfg.max_weight

        alpha_t = self.controller.used_alpha if cfg.use_adaptive_alpha else cfg.alpha

        def _offsets(w):
            if cfg.asymmetric:
                a2 = alpha_t / 2.0
                hi = weighted_quantile_with_test_mass(signed, w, a2, cfg.test_weight)
                lo = weighted_quantile_with_test_mass(-signed, w, a2, cfg.test_weight)
                return lo, hi
            qq = weighted_quantile_with_test_mass(scores, w, alpha_t, cfg.test_weight)
            return qq, qq

        q_lo, q_hi = _offsets(weights)
        fallback_used = False
        abstained = False

        if not supported or not (np.isfinite(q_lo) and np.isfinite(q_hi)):
            if cfg.fallback == "global":
                fallback_used = True
                global_w = temporal if cfg.use_temporal else np.ones(len(scores))
                q_lo, q_hi = _offsets(global_w)
            else:
                abstained = True
                q_lo = q_hi = float("inf")
        q = 0.5 * (q_lo + q_hi)

        proxies = perturbation_proxies(
            raw_ratio, clipped_ratio, local, regime, ess, bandwidth
        )
        return {
            "prediction": pred,
            "scale": scale,
            "lower": pred - q_lo * scale,
            "upper": pred + q_hi * scale,
            "q": q,
            "alpha_t": alpha_t,
            "ess": ess,
            "max_weight": pmax,
            "weight_entropy": weight_entropy(weights),
            "bandwidth": bandwidth,
            "regime_entropy": float(-np.sum(pi_t * np.log(np.clip(pi_t, 1e-12, 1)))),
            "regime": int(np.argmax(pi_t)),
            "regime_now": int(np.argmax(pi_t_now)),
            "horizon": int(self._h),
            "supported": bool(supported),
            "fallback_used": bool(fallback_used),
            "abstained": bool(abstained),
            **proxies,
        }

    def update_one(self, x_t: np.ndarray, y_t: float, result: dict):
        error = int(not (result["lower"] <= y_t <= result["upper"]))
        if self.config.use_adaptive_alpha:
            self.controller.update(error)
        x = np.asarray(x_t, float).reshape(1, -1)
        pred = float(self.base_model_.predict(x)[0])
        scale = float(max(self.scale_model_.predict(x)[0], 1e-6))
        score = abs(float(y_t) - pred) / scale
        pi = self.regime_model_.predict_proba(x)
        self.x_cal_ = np.vstack([self.x_cal_, x])
        self.y_cal_ = np.append(self.y_cal_, y_t)
        self.pred_cal_ = np.append(self.pred_cal_, pred)
        self.scale_cal_ = np.append(self.scale_cal_, scale)
        self.scores_ = np.append(self.scores_, score)
        self.signed_ = np.append(self.signed_, (float(y_t) - pred) / scale)
        self.pi_cal_ = np.vstack([self.pi_cal_, pi])
        max_keep = max(self.config.calibration_window * 2, self.config.calibration_window + 10)
        if len(self.x_cal_) > max_keep:
            self.x_cal_ = self.x_cal_[-max_keep:]
            self.y_cal_ = self.y_cal_[-max_keep:]
            self.pred_cal_ = self.pred_cal_[-max_keep:]
            self.scale_cal_ = self.scale_cal_[-max_keep:]
            self.scores_ = self.scores_[-max_keep:]
            self.signed_ = self.signed_[-max_keep:]
            self.pi_cal_ = self.pi_cal_[-max_keep:]
        return error
