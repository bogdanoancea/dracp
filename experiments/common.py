from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from dracp.metrics import interval_metrics, rolling_metrics
from dracp.utils import ensure_dir

def sequential_evaluate(method, x_test, y_test, dates=None, horizon=1):
    """Sequential evaluation with feedback delayed to match the forecast horizon.

    At horizon ``h`` the target attached to origin ``i`` is realised ``h - 1``
    rows later in the shifted frame. Updating the calibration set immediately
    after each prediction -- which is correct at ``h = 1`` -- would therefore
    reveal an outcome that a real-time forecaster could not yet have seen, and
    the leak grows with the horizon.

    Feedback is instead queued and released only once the target has actually
    become observable: the observation for origin ``i`` enters the calibration
    set no earlier than row ``i + h - 1``. At ``h = 1`` the queue is released
    immediately and the behaviour is identical to the previous implementation,
    so one-step results are unchanged.
    """
    rows = []
    pending = []            # (release_row, x, y, result)
    h = max(1, int(horizon))
    start = time.perf_counter()
    for i, (x, y) in enumerate(zip(x_test, y_test)):
        while pending and pending[0][0] <= i:
            _, px, py, pres = pending.pop(0)
            method.update_one(px, py, pres)
        result = method.predict_one(x)
        error = int(not (result["lower"] <= float(y) <= result["upper"]))
        pending.append((i + h - 1, x, float(y), result))
        result.update({"i": i, "y": float(y), "error": error})
        if dates is not None:
            result["date"] = dates[i]
        rows.append(result)
    for _, px, py, pres in pending:      # flush so state is consistent at the end
        method.update_one(px, py, pres)
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


def build_methods(cfg, seed, include_online=True, include_ablations=False,
                  horizon=1):
    """Construct competing methods. The headline ``dracp`` is the regime-aware,
    self-tuning, weighted conformal method (the paper's contribution). FACI, SAOCP,
    Conformal-PID and the classical CP methods are BASELINES. ``include_ablations``
    adds the weight-component ablations and the optional online selection hedge
    ``dracp_select`` (a deployment robustness variant, not the headline method).

    ``horizon`` > 1 additionally registers the two multi-step variants:
    ``dracp_h`` (horizon-aware weighting: target-aligned regimes, softened decay,
    inflated bandwidth/ESS, damped controller) and ``dracp_hedge`` (the
    no-regret hedge over the horizon-aware core, FACI and conformal PID). The
    plain ``dracp`` entry is retained at every horizon so that the horizon-aware
    variants can be compared against the horizon-unaware method directly. At
    ``horizon=1`` nothing changes and the h=1 results reproduce exactly."""
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
        # Conditional-scale ablation. The headline method normalises residuals by
        # a fitted conditional scale while the classical baselines use raw
        # absolute residuals, so part of any efficiency gain could come from the
        # normalisation rather than from the weights or the controller. This
        # variant replaces the fitted scale with a constant, isolating that
        # contribution.
        from sklearn.dummy import DummyRegressor
        methods["dracp_no_scale"] = DRACPRegressor(
            DRACPConfig(alpha=alpha, random_state=seed,
                        **{**cfg["dracp"], "use_faci_control": True}),
            base_model=base, scale_model=DummyRegressor(strategy="constant", constant=1.0))
        # weight-component ablations of the headline method
        for variant in cfg.get("ablations", []):
            vcfg = DRACPConfig(alpha=alpha, random_state=seed, **cfg["dracp"], )
            vcfg.use_faci_control = True
            for k, v in variant["changes"].items():
                setattr(vcfg, k, v)
            methods[variant["name"]] = DRACPRegressor(vcfg, base_model=base)
        # optional online selection hedge (deployment robustness variant)
        methods["dracp_select"] = build_unified_dracp(cfg, seed, base, dcfg, mode="select")
    if int(horizon) > 1:
        from dracp.aggregate import build_horizon_dracp
        methods["dracp_h"] = DRACPRegressor(
            DRACPConfig(alpha=alpha, random_state=seed,
                        **{**cfg["dracp"], "use_faci_control": True,
                           "horizon": int(horizon), "horizon_adapt": True}),
            base_model=base)
        methods["dracp_hedge"] = build_horizon_dracp(
            cfg, seed, base, dcfg, int(horizon), mode="select")
    return methods
