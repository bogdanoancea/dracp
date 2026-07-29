from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from dracp.metrics import interval_metrics, rolling_metrics
from dracp.utils import ensure_dir

def sequential_evaluate(method, x_test, y_test, dates=None):
    rows = []
    start = time.perf_counter()
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        result = method.predict_one(x)
        error = method.update_one(x, float(y), result)
        result.update({"i": i, "y": float(y), "error": error})
        if dates is not None:
            result["date"] = dates[i]
        rows.append(result)
    elapsed = time.perf_counter() - start
    df = pd.DataFrame(rows)
    df["latency_ms"] = 1000 * elapsed / max(len(df), 1)
    return df

def save_run(predictions, result_dir, method_name, scenario, alpha):
    result_dir = ensure_dir(result_dir)
    pred_path = result_dir / f"{scenario}__{method_name}__predictions.csv"
    predictions.to_csv(pred_path, index=False)
    metrics = interval_metrics(predictions, alpha)
    metrics.update({"method": method_name, "scenario": scenario})
    roll = rolling_metrics(predictions)
    roll.to_csv(result_dir / f"{scenario}__{method_name}__rolling.csv", index=False)
    return metrics

def write_metadata(path, config):
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
    }
    Path(path).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def build_methods(cfg, seed, include_online=True, include_ablations=False):
    """Construct competing methods. The headline ``dracp`` is the regime-aware,
    self-tuning, weighted conformal method (the paper's contribution). FACI, SAOCP,
    Conformal-PID and the classical CP methods are BASELINES. ``include_ablations``
    adds the weight-component ablations and the optional online selection hedge
    ``dracp_select`` (a deployment robustness variant, not the headline method)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from dracp import (DRACPConfig, DRACPRegressor, SplitConformalRegressor,
                       RollingConformalRegressor, AdaptiveConformalRegressor,
                       FACIRegressor, SAOCPRegressor, ConformalPIDRegressor)
    from dracp.aggregate import build_unified_dracp
    alpha = cfg["alpha"]
    base = HistGradientBoostingRegressor(
        max_iter=cfg["model"].get("max_iter", 200),
        learning_rate=cfg["model"].get("learning_rate", 0.05),
        random_state=seed,
    )
    dcfg = DRACPConfig(alpha=alpha, random_state=seed, **cfg["dracp"])
    win = dcfg.calibration_window
    methods = {
        "split_cp": SplitConformalRegressor(alpha, base),
        "rolling_cp": RollingConformalRegressor(alpha, win, base),
        "adaptive_cp": AdaptiveConformalRegressor(alpha, win, dcfg.alpha_step, base),
    }
    if include_online:
        methods["faci"] = FACIRegressor(alpha, win, model=base)
        methods["saocp"] = SAOCPRegressor(alpha, win, model=base)
        methods["conformal_pid"] = ConformalPIDRegressor(alpha, win, model=base)
    # headline: regime-aware self-tuning weighted DRACP (the contribution)
    methods["dracp"] = DRACPRegressor(
        DRACPConfig(alpha=alpha, random_state=seed,
                    **{**cfg["dracp"], "use_faci_control": True}), base_model=base)
    if include_ablations:
        # weight-component ablations of the headline method
        for variant in cfg.get("ablations", []):
            vcfg = DRACPConfig(alpha=alpha, random_state=seed, **cfg["dracp"], )
            vcfg.use_faci_control = True
            for k, v in variant["changes"].items():
                setattr(vcfg, k, v)
            methods[variant["name"]] = DRACPRegressor(vcfg, base_model=base)
        # optional online selection hedge (deployment robustness variant)
        methods["dracp_select"] = build_unified_dracp(cfg, seed, base, dcfg, mode="select")
    return methods
