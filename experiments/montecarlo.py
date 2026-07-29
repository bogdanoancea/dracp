"""Monte-Carlo repetition of DRACP experiments with error bars.

Runs an experiment over many random seeds and aggregates each metric into
mean, std, standard error and a 95% confidence interval, so the main results
can be reported with uncertainty rather than from a single run.

  Synthetic:  each seed re-draws the data-generating process AND re-seeds the
              models  ->  genuine sampling error bars.
  Real data:  the data are fixed, so each seed only re-seeds the stochastic
              estimators (gradient-boosting subsampling, the density-ratio
              classifier, the Gaussian-mixture regime model)  ->  error bars on
              estimation variability. A block-bootstrap CI over the test stream
              is additionally reported for the real-data coverage/width/score.

Usage:
  python -m experiments.montecarlo --config configs/synthetic.yaml \
      --kind synthetic --seeds 20
  python -m experiments.montecarlo --config configs/us_unemployment.yaml \
      --kind real --seeds 20
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from dracp import (DRACPConfig, DRACPRegressor, SplitConformalRegressor,
                   RollingConformalRegressor, AdaptiveConformalRegressor)
from dracp.utils import chronological_split, ensure_dir, load_yaml, set_seed
from dracp.metrics import interval_metrics, interval_score
from synthetic import generate_scenario
from .common import sequential_evaluate, build_methods


# ----------------------------------------------------------------------------
def _make_methods(cfg, seed):
    """Delegates to experiments.common.build_methods (online baselines +
    the enhanced asymmetric DRACP variant), honouring the module-level
    INCLUDE_ONLINE flag and any ablations declared in the config."""
    return build_methods(cfg, seed, include_online=INCLUDE_ONLINE,
                         include_ablations=bool(cfg.get("ablations")))

def _one_seed_synthetic(cfg, seed):
    rows = []
    for s_idx, scenario in enumerate(cfg["scenarios"]):
        df = generate_scenario(scenario, cfg["n_samples"], cfg["n_features"], seed + s_idx)
        fcols = [c for c in df.columns if c.startswith("x")]
        train, cal, test = chronological_split(
            df, cfg["train_fraction"], cfg["calibration_fraction"])
        for name, method in _make_methods(cfg, seed).items():
            method.fit(train[fcols].to_numpy(), train["y"].to_numpy(),
                       cal[fcols].to_numpy(), cal["y"].to_numpy())
            pred = sequential_evaluate(method, test[fcols].to_numpy(), test["y"].to_numpy())
            m = interval_metrics(pred, cfg["alpha"])
            m.update({"seed": seed, "scenario": scenario, "method": name})
            rows.append(m)
    return rows


def _one_seed_real(cfg, seed, df, features, target, date_col):
    train, cal, test = chronological_split(
        df, cfg["train_fraction"], cfg["calibration_fraction"])
    rows, preds = [], {}
    dates = test[date_col].astype(str).tolist() if date_col else None
    for name, method in _make_methods(cfg, seed).items():
        method.fit(train[features].to_numpy(), train[target].to_numpy(),
                   cal[features].to_numpy(), cal[target].to_numpy())
        pred = sequential_evaluate(method, test[features].to_numpy(),
                                   test[target].to_numpy(), dates=dates)
        m = interval_metrics(pred, cfg["alpha"])
        m.update({"seed": seed, "scenario": cfg["name"], "method": name})
        rows.append(m)
        preds[name] = pred
    return rows, preds


# ----------------------------------------------------------------------------
def _aggregate(long_df, metrics):
    """mean/std/sem/95% CI per (scenario, method, metric) across seeds."""
    out = []
    for (scen, method), g in long_df.groupby(["scenario", "method"]):
        n = len(g)
        rec = {"scenario": scen, "method": method, "n_seeds": n}
        for metric in metrics:
            if metric not in g:
                continue
            vals = g[metric].to_numpy(dtype=float)
            mean, sd = float(np.mean(vals)), float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = sd / np.sqrt(n) if n > 1 else 0.0
            if n > 1:
                h = stats.t.ppf(0.975, n - 1) * sem
            else:
                h = 0.0
            rec[f"{metric}_mean"] = mean
            rec[f"{metric}_std"] = sd
            rec[f"{metric}_sem"] = sem
            rec[f"{metric}_ci_low"] = mean - h
            rec[f"{metric}_ci_high"] = mean + h
        out.append(rec)
    return pd.DataFrame(out)


def _block_bootstrap_real(preds, alpha, block=50, B=1000, seed=0):
    """Moving-block bootstrap CIs on coverage/width/score for fixed real data."""
    rng = np.random.default_rng(seed)
    out = []
    for name, pred in preds.items():
        v = pred.replace([np.inf, -np.inf], np.nan).dropna(subset=["y", "lower", "upper"])
        y = v["y"].to_numpy(); lo = v["lower"].to_numpy(); up = v["upper"].to_numpy()
        cov = ((y >= lo) & (y <= up)).astype(float)
        wid = up - lo
        sco = interval_score(y, lo, up, alpha)
        n = len(y)
        if n < block:
            block = max(2, n // 5)
        starts_max = n - block
        cc, ww, ss = [], [], []
        nblocks = int(np.ceil(n / block))
        for _ in range(B):
            idx = []
            for _ in range(nblocks):
                s0 = rng.integers(0, starts_max + 1)
                idx.extend(range(s0, s0 + block))
            idx = np.array(idx[:n])
            cc.append(cov[idx].mean()); ww.append(wid[idx].mean()); ss.append(sco[idx].mean())
        ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
        cl, ch = ci(cc); wl, wh = ci(ww); sl, sh = ci(ss)
        out.append({"method": name,
                    "coverage_ci_low": cl, "coverage_ci_high": ch,
                    "mean_width_ci_low": wl, "mean_width_ci_high": wh,
                    "mean_interval_score_ci_low": sl, "mean_interval_score_ci_high": sh})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
METRICS = ["coverage", "coverage_error", "mean_width", "median_width",
           "mean_interval_score", "abstention_rate", "fallback_rate"]


def main(config_path, kind, seeds, seed0):
    cfg = load_yaml(config_path)
    set_seed(cfg.get("seed", 42))
    out_dir = ensure_dir(cfg["output_dir"])
    seed_list = list(range(seed0, seed0 + seeds))

    all_rows = []
    if kind == "synthetic":
        for i, sd in enumerate(seed_list):
            print(f"[MC synthetic] seed {sd} ({i+1}/{len(seed_list)})", flush=True)
            all_rows.extend(_one_seed_synthetic(cfg, sd))
        boot = None
    else:
        df = pd.read_csv(cfg["data"]["path"])
        date_col = cfg["data"].get("date_column")
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col]); df = df.sort_values(date_col)
        target = cfg["data"]["target_column"]; features = cfg["data"]["feature_columns"]
        df = df.dropna(subset=features + [target]).reset_index(drop=True)
        last_preds = None
        for i, sd in enumerate(seed_list):
            print(f"[MC real] seed {sd} ({i+1}/{len(seed_list)})", flush=True)
            rows, preds = _one_seed_real(cfg, sd, df, features, target, date_col)
            all_rows.extend(rows); last_preds = preds
        boot = _block_bootstrap_real(last_preds, cfg["alpha"])

    long_df = pd.DataFrame(all_rows)
    long_path = out_dir / "mc_runs.csv"
    long_df.to_csv(long_path, index=False)
    summary = _aggregate(long_df, METRICS)
    summary_path = out_dir / "mc_summary.csv"
    summary.to_csv(summary_path, index=False)
    if boot is not None:
        boot.to_csv(out_dir / "mc_block_bootstrap.csv", index=False)

    print("\n=== Monte-Carlo summary (mean [95% CI]) ===")
    for _, r in summary.iterrows():
        print(f"{r['scenario']:>18s} | {r['method']:<16s} "
              f"cov={r['coverage_mean']:.4f} "
              f"[{r['coverage_ci_low']:.4f},{r['coverage_ci_high']:.4f}]  "
              f"score={r['mean_interval_score_mean']:.3f} "
              f"[{r['mean_interval_score_ci_low']:.3f},{r['mean_interval_score_ci_high']:.3f}]")
    print(f"\nWrote:\n  {long_path}\n  {summary_path}")
    if boot is not None:
        print(f"  {out_dir/'mc_block_bootstrap.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--kind", choices=["synthetic", "real"], required=True)
    p.add_argument("--seeds", type=int, default=20, help="number of seeds")
    p.add_argument("--seed0", type=int, default=42, help="first seed")
    p.add_argument("--no-online", action="store_true", help="exclude FACI/SAOCP/Conformal-PID baselines")
    a = p.parse_args()
    INCLUDE_ONLINE = not a.no_online
    main(a.config, a.kind, a.seeds, a.seed0)
