"""Moving-block bootstrap for interval-forecast losses.

Replaces the ad-hoc resampler previously embedded in ``experiments.montecarlo``.
Four properties of that version were statistically unsatisfactory and are
corrected here.

1. **Data-driven block length.** Chosen per series by the Politis--White (2004)
   automatic rule rather than fixed at 50. The fixed choice was badly
   miscalibrated for short monthly series: a block of 50 on 84 observations
   leaves only 34 distinct starts, so replicates overlap heavily and the
   resulting intervals are far too narrow.

2. **Circular blocks.** Blocks wrap around the end of the series so every
   observation is sampled with equal probability; the non-circular version
   under-weighted both ends.

3. **Paired resampling.** All methods share one set of block indices per
   replicate, so differences are computed at the same resampled time points.
   This makes the bootstrap valid for *differences*. Comparing two marginal
   intervals for overlap -- what the old code invited -- is conservative and is
   not a test of a difference.

4. **Repeated fits pooled.** When several fits (seeds) are supplied, estimator
   variability and time-series variability are combined in one interval by
   drawing a fit index at each replicate, instead of discarding all but the
   last fit.

Entry point: :func:`block_bootstrap`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------- block length
def _flat_top_lag_window(n: int) -> int:
    return int(max(5, np.ceil(np.sqrt(np.log10(max(n, 10))))))


def politis_white_block_length(x, max_frac: float = 0.25) -> int:
    """Automatic block length for the moving-block bootstrap.

    Implements Politis and White (2004) with the Patton, Politis and White
    (2009) correction: the block length balances bias and variance of the
    long-run variance estimator. The result is capped at ``max_frac`` of the
    sample so short series cannot get a block that makes the resample
    degenerate (the failure mode of the previous fixed choice).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return max(1, n)
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return 1

    def rho(k: int) -> float:
        return float(np.dot(x[k:], x[: n - k]) / (n * var))

    k_n = _flat_top_lag_window(n)
    m_max = min(int(np.ceil(np.sqrt(n))) + k_n, n - 1)
    crit = 2.0 * np.sqrt(np.log10(n) / n)
    m_hat = None
    for m in range(1, max(m_max - k_n, 1) + 1):
        if all(abs(rho(m + k)) < crit for k in range(1, k_n + 1) if m + k < n):
            m_hat = m
            break
    if m_hat is None:
        m_hat = max(1, min(m_max, int(np.ceil(np.sqrt(n)))))
    big_m = min(2 * m_hat, n - 1)

    def lam(t: float) -> float:
        a = abs(t)
        if a <= 0.5:
            return 1.0
        if a <= 1.0:
            return 2.0 * (1.0 - a)
        return 0.0

    g_hat = 0.0
    acc = var
    for k in range(1, big_m + 1):
        r = rho(k) * var
        w = lam(k / big_m)
        g_hat += 2.0 * w * k * r
        acc += 2.0 * w * r
    d_hat = 2.0 * acc ** 2
    if d_hat <= 0 or not np.isfinite(g_hat):
        return max(1, int(np.ceil(n ** (1 / 3))))
    b = ((2.0 * g_hat ** 2) / d_hat) ** (1 / 3) * n ** (1 / 3)
    if not np.isfinite(b):
        return max(1, int(np.ceil(n ** (1 / 3))))
    return int(np.clip(np.round(b), 1, max(1, int(max_frac * n))))


# ------------------------------------------------------------------ resampling
def circular_block_indices(n: int, block: int, rng) -> np.ndarray:
    """Indices for one circular moving-block replicate of length ``n``."""
    if block >= n:
        return np.arange(n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
    return idx[:n]


@dataclass
class BootstrapResult:
    block_length: int
    n_replicates: int
    n_fits: int
    marginal: dict = field(default_factory=dict)
    paired: dict = field(default_factory=dict)
    reference: str | None = None

    def to_frame(self):
        import pandas as pd
        rows = []
        for m, met in self.marginal.items():
            row = {"method": m, "block_length": self.block_length,
                   "n_replicates": self.n_replicates, "n_fits": self.n_fits}
            for k, (mean, lo, hi) in met.items():
                row[k], row[f"{k}_lo"], row[f"{k}_hi"] = mean, lo, hi
            for k, (d, lo, hi, p) in self.paired.get(m, {}).items():
                row[f"d_{k}"], row[f"d_{k}_lo"] = d, lo
                row[f"d_{k}_hi"], row[f"d_{k}_p"] = hi, p
            row["reference"] = self.reference
            rows.append(row)
        return pd.DataFrame(rows)


def block_bootstrap(losses: Mapping[str, "Sequence[np.ndarray] | np.ndarray"],
                    *, reference: str | None = None, n_replicates: int = 2000,
                    block: int | None = None, seed: int = 0,
                    alpha: float = 0.05) -> BootstrapResult:
    """Paired circular moving-block bootstrap over per-period losses.

    ``losses`` maps method name to either a 1-D array of per-period losses (one
    fit) or a 2-D array with one row per repeated fit. All methods must share
    the number of periods, and (when >1) the number of fits, so that block
    indices and the fit index can be drawn jointly -- this is what makes the
    differences paired.

    ``reference`` selects the method against which paired differences are
    formed as ``reference - method``; with a negatively oriented loss a positive
    difference favours ``method``.
    """
    names = list(losses)
    if not names:
        raise ValueError("no methods supplied")
    mats = {k: np.atleast_2d(np.asarray(v, float)) for k, v in losses.items()}
    n_periods = mats[names[0]].shape[1]
    for k, a in mats.items():
        if a.shape[1] != n_periods:
            raise ValueError(f"method {k!r} has {a.shape[1]} periods, expected {n_periods}")
    n_fits = max(a.shape[0] for a in mats.values())
    for k, a in mats.items():
        if a.shape[0] not in (1, n_fits):
            raise ValueError(f"method {k!r} has {a.shape[0]} fits, expected 1 or {n_fits}")
    if reference is not None and reference not in mats:
        raise KeyError(f"reference {reference!r} not among {names}")

    if block is None:
        if reference is not None and len(names) > 1:
            other = next(k for k in names if k != reference)
            probe = mats[reference][0] - mats[other][0]
        else:
            probe = np.mean([a[0] for a in mats.values()], axis=0)
        block = politis_white_block_length(probe)
    block = int(max(1, min(block, n_periods)))

    rng = np.random.default_rng(seed)
    means = {k: np.empty(n_replicates) for k in names}
    for b in range(n_replicates):
        idx = circular_block_indices(n_periods, block, rng)
        fit = int(rng.integers(0, n_fits)) if n_fits > 1 else 0
        for k, a in mats.items():
            row = a[fit if a.shape[0] > 1 else 0]
            means[k][b] = np.nanmean(row[idx])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    res = BootstrapResult(block, n_replicates, n_fits, reference=reference)
    for k in names:
        point = float(np.nanmean([np.nanmean(r) for r in mats[k]]))
        lo, hi = np.percentile(means[k], [lo_q, hi_q])
        res.marginal[k] = {"loss": (point, float(lo), float(hi))}
    if reference is not None:
        for k in names:
            if k == reference:
                continue
            d = means[reference] - means[k]
            point = float(np.nanmean([np.nanmean(r) for r in mats[reference]])
                          - np.nanmean([np.nanmean(r) for r in mats[k]]))
            lo, hi = np.percentile(d, [lo_q, hi_q])
            p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
            res.paired[k] = {"loss": (point, float(lo), float(hi), float(min(1.0, p)))}
    return res
