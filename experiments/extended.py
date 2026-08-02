"""Extended experiments: ablation, base models, horizons, sensitivity, runtime.

All output goes to the single ``results/`` folder alongside the main pipeline;
there is no separate reviewer/ subfolder.

Sub-commands
------------
euro-area     Download and run the TRUE euro-area HICP aggregate (Eurostat geo=EA20).
              The previous 'european_inflation' config used geo=RO by mistake.
ablation      Component ablation on the REAL panel (not only synthetic).
models        Repeat the comparison with several base forecasters
              (gradient boosting, random forest, ridge, ARIMA-style AR, naive drift).
horizons      Multi-step-ahead evaluation (h = 1, 3, 6, 12).
sensitivity   Extra sensitivity: ESS floor, number of regimes, density-ratio estimator.
runtime       Runtime/memory benchmark with hardware metadata.

Examples
--------
    python -m experiments.extended euro-area
    python -m experiments.extended ablation --seeds 20
    python -m experiments.extended models --seeds 10
    python -m experiments.extended horizons --seeds 10
    python -m experiments.extended sensitivity
    python -m experiments.extended runtime
"""
from __future__ import annotations
import argparse, glob, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from dracp import DRACPConfig, DRACPRegressor
from dracp.metrics import interval_metrics
from dracp.utils import chronological_split, load_yaml, set_seed
from experiments.common import sequential_evaluate, build_methods

OUT = Path(os.environ.get("RESULTS_ROOT", "results")); OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- helpers
def load_series(cfg):
    df = pd.read_csv(cfg["data"]["path"])
    dc = cfg["data"].get("date_column")
    if dc:
        df[dc] = pd.to_datetime(df[dc]); df = df.sort_values(dc)
    tgt = cfg["data"]["target_column"]; feats = cfg["data"]["feature_columns"]
    df = df.dropna(subset=feats + [tgt]).reset_index(drop=True)
    return df, feats, tgt, dc

def real_configs(limit=None):
    names = []
    for man in ("configs/economic_manifest.json", "configs/eu_inflation_manifest.json"):
        if Path(man).exists():
            names += [d["name"] for d in json.loads(Path(man).read_text())]
    for extra in ("euro_area_inflation", "electricity"):
        if Path(f"configs/{extra}.yaml").exists():
            names.append(extra)
    names = [n for n in names if Path(f"data/processed/{n}.csv").exists()]
    return names[:limit] if limit else names

# ------------------------------------------------------------------ euro-area
def cmd_euro_area(args):
    """Download the true euro-area aggregate and run it."""
    raw = "data/raw/inflation_EA20.csv"
    print("Downloading Eurostat HICP for the euro area (geo=EA20)...")
    subprocess.check_call([sys.executable, "-m", "real_data.download_inflation",
                           "--output", raw, "--geo", "EA20"])
    subprocess.check_call([sys.executable, "-m", "real_data.prepare_inflation",
                           "--input", raw, "--output", "data/processed/euro_area_inflation.csv",
                           "--date-column", "date", "--target-column", "inflation"])
    subprocess.check_call([sys.executable, "-m", "experiments.run_real",
                           "--config", "configs/euro_area_inflation.yaml"])
    subprocess.check_call([sys.executable, "-m", "experiments.montecarlo",
                           "--config", "configs/euro_area_inflation.yaml",
                           "--kind", "real", "--seeds", str(args.seeds)])
    print("\nDone. New series: results/euro_area_inflation/")
    print("NOTE: the old 'results/inflation' folder is the Romanian series mislabelled")
    print("      as euro-area; it duplicates results/inflation_RO and should be dropped.")

# ------------------------------------------------------------------- ablation
def cmd_ablation(args):
    """Component ablation on the real panel."""
    variants = {
        "dracp":             {},
        "dracp_no_ratio":    {"use_ratio": False},
        "dracp_no_local":    {"use_local": False},
        "dracp_no_regime":   {"use_regime": False},
        "dracp_no_adaptive": {"use_adaptive_alpha": False},
        "dracp_no_temporal": {"use_temporal": False},
    }
    rows = []
    names = real_configs(args.limit)
    print(f"Real-panel ablation over {len(names)} series x {args.seeds} seeds")
    for name in names:
        cfg = load_yaml(f"configs/{name}.yaml")
        df, feats, tgt, dc = load_series(cfg)
        for seed in range(42, 42 + args.seeds):
            set_seed(seed)
            tr, cal, te = chronological_split(df, cfg["train_fraction"], cfg["calibration_fraction"])
            from sklearn.ensemble import HistGradientBoostingRegressor
            base = HistGradientBoostingRegressor(
                max_iter=cfg["model"].get("max_iter", 200),
                learning_rate=cfg["model"].get("learning_rate", 0.05), random_state=seed)
            for vname, over in variants.items():
                c = DRACPConfig(alpha=cfg["alpha"], random_state=seed,
                                **{**cfg["dracp"], "use_faci_control": True, **over})
                m = DRACPRegressor(c, base_model=base)
                m.fit(tr[feats].to_numpy(), tr[tgt].to_numpy(),
                      cal[feats].to_numpy(), cal[tgt].to_numpy())
                pred = sequential_evaluate(m, te[feats].to_numpy(), te[tgt].to_numpy())
                met = interval_metrics(pred, cfg["alpha"])
                met.update(series=name, seed=seed, variant=vname); rows.append(met)
        print(f"  {name} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"real_ablation.csv", index=False)
    print(f"\nWrote {OUT/'real_ablation.csv'}")

# --------------------------------------------------------------------- models
def _base_models(seed):
    """Several point forecasters, all with fit/predict."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.dummy import DummyRegressor
    from sklearn.neural_network import MLPRegressor
    return {
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                           random_state=seed),
        "random_forest":     RandomForestRegressor(n_estimators=200, random_state=seed),
        "ridge_ar":          RidgeCV(alphas=np.logspace(-3, 3, 13)),   # linear AR model
        "mlp":               MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                                          random_state=seed),
        "naive":             DummyRegressor(strategy="mean"),
    }

def cmd_models(args):
    """Repeat the method comparison across several base forecasters."""
    rows = []
    names = real_configs(args.limit)
    print(f"Base-model comparison over {len(names)} series x {args.seeds} seeds")
    for name in names:
        cfg = load_yaml(f"configs/{name}.yaml")
        df, feats, tgt, dc = load_series(cfg)
        for seed in range(42, 42 + args.seeds):
            set_seed(seed)
            tr, cal, te = chronological_split(df, cfg["train_fraction"], cfg["calibration_fraction"])
            for mname, model in _base_models(seed).items():
                # point-forecast accuracy of the base model itself
                try:
                    model.fit(tr[feats].to_numpy(), tr[tgt].to_numpy())
                    rmse = float(np.sqrt(np.mean((te[tgt].to_numpy()
                                 - model.predict(te[feats].to_numpy()))**2)))
                except Exception as exc:
                    print(f"    {mname} failed on {name}: {exc}"); continue
                methods = build_methods(cfg, seed, include_online=True)
                for meth, obj in methods.items():
                    # swap in this base model
                    if hasattr(obj, "base_model"): obj.base_model = model
                    if hasattr(obj, "model"):      obj.model = model
                    try:
                        obj.fit(tr[feats].to_numpy(), tr[tgt].to_numpy(),
                                cal[feats].to_numpy(), cal[tgt].to_numpy())
                        pred = sequential_evaluate(obj, te[feats].to_numpy(), te[tgt].to_numpy())
                        met = interval_metrics(pred, cfg["alpha"])
                    except Exception as exc:
                        print(f"    {meth}/{mname} failed on {name}: {exc}"); continue
                    met.update(series=name, seed=seed, base_model=mname,
                               method=meth, base_rmse=rmse)
                    rows.append(met)
        print(f"  {name} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"base_models.csv", index=False)
    print(f"\nWrote {OUT/'base_models.csv'}")

# ------------------------------------------------------------------- horizons
def cmd_horizons(args):
    """Multi-step-ahead evaluation: shift the target by h before splitting."""
    rows = []
    names = real_configs(args.limit)
    horizons = [int(h) for h in args.horizons.split(",")]
    print(f"Multi-step evaluation h={horizons} over {len(names)} series x {args.seeds} seeds")
    for name in names:
        cfg = load_yaml(f"configs/{name}.yaml")
        df0, feats, tgt, dc = load_series(cfg)
        for h in horizons:
            df = df0.copy()
            df[tgt] = df[tgt].shift(-(h-1))          # predict y_{t+h} from information at t
            df = df.dropna(subset=[tgt]).reset_index(drop=True)
            if len(df) < 60: continue
            for seed in range(42, 42 + args.seeds):
                set_seed(seed)
                tr, cal, te = chronological_split(df, cfg["train_fraction"], cfg["calibration_fraction"])
                for meth, obj in build_methods(cfg, seed, include_online=True,
                                               horizon=h).items():
                    obj.fit(tr[feats].to_numpy(), tr[tgt].to_numpy(),
                            cal[feats].to_numpy(), cal[tgt].to_numpy())
                    pred = sequential_evaluate(obj, te[feats].to_numpy(), te[tgt].to_numpy())
                    met = interval_metrics(pred, cfg["alpha"])
                    met.update(series=name, seed=seed, horizon=h, method=meth); rows.append(met)
        print(f"  {name} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"horizons.csv", index=False)
    print(f"\nWrote {OUT/'horizons.csv'}")

# ---------------------------------------------------------------- sensitivity
def cmd_sensitivity(args):
    """ESS floor, number of regimes, and density-ratio estimator."""
    from synthetic import generate_scenario
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    grids = {
        "min_ess":   [10, 20, 30, 50, 80],
        "n_regimes": [1, 2, 3, 4, 5, 6],
        "ratio_clip_max": [2, 5, 10, 20, 1e9],
        "max_weight": [0.05, 0.10, 0.25, 0.50, 1.0],
    }
    base = dict(calibration_window=400, target_window=80, temporal_decay=0.995,
                bandwidth=2.0, adaptive_bandwidth=True, bandwidth_grid=(0.5,1,2,4,8),
                min_ess=30, max_weight=0.25, ratio_clip_min=0.05, ratio_clip_max=20.0,
                n_regimes=3, regime_beta=1.0, alpha_step=0.01, fallback="global",
                use_faci_control=True)
    for key, values in grids.items():
        for v in values:
            set_seed(42)
            df = generate_scenario("combined_shift", 3000, 5, 48)
            fc = [c for c in df.columns if c.startswith("x")]
            tr, cal, te = chronological_split(df, 0.45, 0.25)
            cfg = DRACPConfig(alpha=0.1, random_state=42, **{**base, key: v})
            m = DRACPRegressor(cfg, base_model=HistGradientBoostingRegressor(
                max_iter=180, learning_rate=0.05, random_state=42))
            m.fit(tr[fc].to_numpy(), tr["y"].to_numpy(), cal[fc].to_numpy(), cal["y"].to_numpy())
            pred = sequential_evaluate(m, te[fc].to_numpy(), te["y"].to_numpy())
            met = interval_metrics(pred, 0.1)
            met.update(parameter=key, value=float(v), mean_ess=float(pred["ess"].mean()))
            rows.append(met); print(f"  {key}={v}: cov={met['coverage']:.3f} "
                                   f"score={met['mean_interval_score']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT/"sensitivity_extra.csv", index=False)
    print(f"\nWrote {OUT/'sensitivity_extra.csv'}")

# -------------------------------------------------------------------- runtime
def cmd_runtime(args):
    """Runtime and peak memory per prediction, with hardware metadata."""
    import tracemalloc
    from synthetic import generate_scenario
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    set_seed(42)
    df = generate_scenario("combined_shift", 3000, 5, 48)
    fc = [c for c in df.columns if c.startswith("x")]
    tr, cal, te = chronological_split(df, 0.45, 0.25)
    cfg_yaml = {"alpha": 0.1, "model": {"max_iter": 180, "learning_rate": 0.05},
                "dracp": dict(calibration_window=400, target_window=80, temporal_decay=0.995,
                              bandwidth=2.0, adaptive_bandwidth=True,
                              bandwidth_grid=(0.5,1,2,4,8), min_ess=30, max_weight=0.25,
                              ratio_clip_min=0.05, ratio_clip_max=20.0, n_regimes=3,
                              regime_beta=1.0, alpha_step=0.01, fallback="global")}
    for meth, obj in build_methods(cfg_yaml, 42, include_online=True).items():
        # start tracing before fitting so the peak includes the fitted model and
        # the calibration buffer, not just per-prediction allocations
        tracemalloc.start()
        obj.fit(tr[fc].to_numpy(), tr["y"].to_numpy(), cal[fc].to_numpy(), cal["y"].to_numpy())
        X = te[fc].to_numpy(); Y = te["y"].to_numpy()
        t0 = time.perf_counter()
        for i in range(len(X)):
            r = obj.predict_one(X[i]); obj.update_one(X[i], float(Y[i]), r)
        el = time.perf_counter() - t0
        cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        rows.append({"method": meth, "ms_per_prediction": 1000*el/len(X),
                     "peak_memory_mb": peak/1e6, "n_predictions": len(X)})
        print(f"  {meth:<16} {1000*el/len(X):7.2f} ms/pred  peak {peak/1e6:6.1f} MB", flush=True)
    meta = {"platform": platform.platform(), "processor": platform.processor(),
            "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    pd.DataFrame(rows).to_csv(OUT/"runtime.csv", index=False)
    (OUT/"runtime_hardware.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {OUT/'runtime.csv'} and runtime_hardware.json")
    print(json.dumps(meta, indent=2))

CMDS = {"euro-area": cmd_euro_area, "ablation": cmd_ablation, "models": cmd_models,
        "horizons": cmd_horizons, "sensitivity": cmd_sensitivity, "runtime": cmd_runtime}

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=list(CMDS))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="use only the first N series")
    p.add_argument("--horizons", default="1,3,6,12")
    a = p.parse_args()
    CMDS[a.command](a)

if __name__ == "__main__":
    main()
