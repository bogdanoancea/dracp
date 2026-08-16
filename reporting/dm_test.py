"""Diebold--Mariano tests of interval-score equality between DRACP and each
baseline, per real dataset, from the saved single-run predictions.

For each dataset it reads results/<dataset>/<scenario>__<method>__predictions.csv,
computes the per-step interval (Winkler) score, forms the loss differential
d_t = score^{base}_t - score^{DRACP}_t, and reports the Diebold--Mariano (1995)
statistic with a Newey--West (HAC) long-run variance and the Harvey--Leybourne--
Newbold (1997) small-sample correction. A positive statistic favours DRACP.

Two distinct quantities were previously conflated in a single argument, which
made the test inconsistent with how it is described:

  * ``forecast_horizon`` -- the h of an h-step-ahead forecast. It enters the
    HLN small-sample correction and, for overlapping forecasts, sets the
    minimum admissible HAC lag (h - 1 periods of overlap).
  * ``hac_lag``          -- the Newey--West truncation lag. Defaults to the
    data-driven rule floor(T^{1/3}), raised to h - 1 when that is larger.

    python -m reporting.dm_test --results-root results --alpha 0.1 --horizon 1
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def interval_score(y, lo, up, alpha):
    y, lo, up = map(lambda z: np.asarray(z, float), (y, lo, up))
    return (up - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - up) * (y > up)


def dm_stat(d, forecast_horizon: int = 1, hac_lag: int | None = None):
    """Diebold--Mariano statistic with the HLN correction.

    ``forecast_horizon`` is the forecast horizon h; ``hac_lag`` is the
    Newey--West truncation lag. They are different quantities and are supplied
    separately. If ``hac_lag`` is omitted it is set to the data-driven
    floor(T^{1/3}), and never below h - 1, since h-step-ahead forecast errors
    overlap over h - 1 periods.
    """
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 8:
        return np.nan, np.nan
    h = max(1, int(forecast_horizon))
    if hac_lag is None:
        hac_lag = max(int(np.floor(T ** (1 / 3))), h - 1)
    hac_lag = int(max(0, min(hac_lag, T - 1)))

    dbar = d.mean()

    def gamma(k):
        return float(np.sum((d[k:] - dbar) * (d[: T - k])) / T) if k else float(np.sum((d - dbar) ** 2) / T)

    # Bartlett-weighted long-run variance over hac_lag lags
    lrv = gamma(0)
    for k in range(1, hac_lag + 1):
        lrv += 2.0 * (1.0 - k / (hac_lag + 1.0)) * float(np.sum((d[k:] - dbar) * (d[: T - k] - dbar)) / T)
    if lrv <= 0:
        return np.nan, np.nan

    dm = dbar / np.sqrt(lrv / T)
    # Harvey, Leybourne & Newbold (1997) small-sample correction, in h
    corr = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= corr
    p = 2 * (1 - stats.t.cdf(abs(dm), df=T - 1))
    return float(dm), float(p)


def main(root, alpha, target="dracp", horizon=1, hac_lag=None, exclude=()):
    root = Path(root)
    baselines = ["split_cp", "rolling_cp", "adaptive_cp", "faci", "saocp", "conformal_pid"]
    rows = []
    for d in sorted(glob.glob(str(root / "*"))):
        d = Path(d)
        if not d.is_dir() or d.name in exclude:
            continue
        preds = glob.glob(str(d / "*__{}__predictions.csv".format(target)))
        if not preds:
            continue
        scen = os.path.basename(preds[0]).split("__")[0]

        def load(m):
            p = d / f"{scen}__{m}__predictions.csv"
            if not p.exists():
                return None
            df = pd.read_csv(p)
            return interval_score(df["y"], df["lower"], df["upper"], alpha)

        st = load(target)
        if st is None:
            continue
        for b in baselines:
            sb = load(b)
            if sb is None:
                continue
            n = min(len(st), len(sb))
            dm, p = dm_stat(sb[:n] - st[:n], forecast_horizon=horizon, hac_lag=hac_lag)
            rows.append({"dataset": d.name, "comparison": f"{b} vs {target}",
                         "dm_stat": dm, "p_value": p, "n": n,
                         "forecast_horizon": horizon,
                         "hac_lag": hac_lag if hac_lag is not None else int(np.floor(n ** (1 / 3)))})
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no prediction files found; run the real stage first")

    # Guard the panel: the manuscript reports 48 series and 288 comparisons.
    n_series, n_tests = out.dataset.nunique(), len(out)
    print(f"series: {n_series}   tests: {n_tests}")
    if n_series * len(baselines) != n_tests:
        print("WARNING: not every series has all six baselines")
    dup = {"inflation", "european_inflation"} & set(out.dataset)
    if dup and "euro_area_inflation" in set(out.dataset):
        print(f"WARNING: {sorted(dup)} duplicates euro_area_inflation; "
              f"pass --exclude {' '.join(sorted(dup))}")

    out.to_csv(root / "dm_results.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {root/'dm_results.csv'}  (positive statistic favours {target})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--target", default="dracp")
    ap.add_argument("--horizon", type=int, default=1,
                    help="forecast horizon h for the HLN correction")
    ap.add_argument("--hac-lag", type=int, default=None,
                    help="Newey-West truncation lag (default floor(T^(1/3)))")
    ap.add_argument("--exclude", nargs="*", default=["inflation"],
                    help="dataset folders to skip (duplicates)")
    a = ap.parse_args()
    main(a.results_root, a.alpha, a.target, a.horizon, a.hac_lag, set(a.exclude))
