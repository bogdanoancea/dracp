from __future__ import annotations

class AdaptiveAlphaController:
    def __init__(self, target_alpha: float = 0.1, step_size: float = 0.01,
                 initial_alpha: float | None = None):
        self.target_alpha = float(target_alpha)
        self.step_size = float(step_size)
        self.state = float(target_alpha if initial_alpha is None else initial_alpha)

    @property
    def used_alpha(self) -> float:
        return min(max(self.state, 1e-6), 1 - 1e-6)

    def update(self, error: int) -> float:
        self.state += self.step_size * (self.target_alpha - int(error))
        return self.state


class PIDAlphaController:
    """PID controller on the miscoverage level.

    Extends the ACI proportional update with an integral term (removes
    steady-state coverage bias) and an optional derivative term (damps
    oscillations). The error signal is e_t = target_alpha - miss_t, where
    miss_t = 1 if the last interval failed to cover. Reduces to ACI when
    ki = kd = 0. Cf. Angelopoulos, Candes & Tibshirani (2024), "Conformal PID
    control for time series prediction".
    """

    def __init__(self, target_alpha: float = 0.1, step_size: float = 0.01,
                 ki: float = 0.001, kd: float = 0.0,
                 integral_clip: float = 12.0, initial_alpha: float | None = None):
        self.target_alpha = float(target_alpha)
        self.kp = float(step_size)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_clip = float(integral_clip)
        self.state = float(target_alpha if initial_alpha is None else initial_alpha)
        self._integral = 0.0
        self._prev_error = 0.0

    @property
    def used_alpha(self) -> float:
        return min(max(self.state, 1e-6), 1 - 1e-6)

    def update(self, error: int) -> float:
        e = self.target_alpha - int(error)
        self._integral = float(
            max(-self.integral_clip, min(self.integral_clip, self._integral + e))
        )
        deriv = e - self._prev_error
        self.state += self.kp * e + self.ki * self._integral + self.kd * deriv
        self._prev_error = e
        return self.state


class FACIController:
    """Self-tuning significance controller: a fully-adaptive aggregation of ACI
    experts over a grid of learning rates (Gibbs & Candes, 2022).

    Each expert keeps its own miscoverage level with its own rate; the used level
    is the weight-averaged expert level, and weights follow exponentiated-gradient
    on the pinball loss of the coverage error with a small uniform-mixing term.
    Interface matches AdaptiveAlphaController (``used_alpha`` + ``update``), so it
    is a drop-in significance controller. Reduces to ACI with a single rate.
    """

    def __init__(self, target_alpha: float = 0.1,
                 gammas=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
                 eta: float = 1.0, sigma: float = 0.05):
        import numpy as _np
        self._np = _np
        self.target_alpha = float(target_alpha)
        self.gammas = _np.asarray(gammas, float)
        self.eta = float(eta)
        self.sigma = float(sigma)
        K = len(self.gammas)
        self.alpha_k = _np.full(K, self.target_alpha, float)
        self.w = _np.ones(K) / K

    @property
    def used_alpha(self) -> float:
        a = float(self._np.dot(self.w, self.alpha_k))
        return min(max(a, 1e-6), 1 - 1e-6)

    def update(self, error: int) -> float:
        np = self._np
        miss = int(error)
        # pinball (quantile) loss of each expert's level against realised coverage
        err = miss - self.alpha_k
        losses = np.where(err >= 0, self.target_alpha * err,
                          (self.target_alpha - 1.0) * err)
        w = self.w * np.exp(-self.eta * losses)
        s = w.sum()
        w = w / s if (s > 0 and np.isfinite(s)) else np.ones_like(w) / len(w)
        K = len(w)
        self.w = (1 - self.sigma) * w + self.sigma / K
        # each expert performs its own ACI update on the shared coverage feedback
        self.alpha_k = np.clip(self.alpha_k + self.gammas * (self.target_alpha - miss),
                               1e-6, 1 - 1e-6)
        return self.used_alpha
