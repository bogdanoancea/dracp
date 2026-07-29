from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from dracp import (
    DRACPConfig, DRACPRegressor, SplitConformalRegressor,
    RollingConformalRegressor, AdaptiveConformalRegressor,
)
from dracp.utils import chronological_split, ensure_dir, load_yaml, set_seed
from synthetic import generate_scenario
from .common import sequential_evaluate, save_run, write_metadata
from reporting.plots import make_all_plots

def make_methods(cfg):
    alpha = cfg["alpha"]
    window = cfg["dracp"]["calibration_window"]
    base = HistGradientBoostingRegressor(
        max_iter=cfg["model"].get("max_iter", 150),
        learning_rate=cfg["model"].get("learning_rate", 0.05),
        random_state=cfg["seed"],
    )
    dr_cfg = DRACPConfig(alpha=alpha, random_state=cfg["seed"], **cfg["dracp"])
    methods = {
        "split_cp": SplitConformalRegressor(alpha, base),
        "rolling_cp": RollingConformalRegressor(alpha, window, base),
        "adaptive_cp": AdaptiveConformalRegressor(
            alpha, window, dr_cfg.alpha_step, base
        ),
        "dracp": DRACPRegressor(dr_cfg, base_model=base),
    }
    for variant in cfg.get("ablations", []):
        vcfg = DRACPConfig(alpha=alpha, random_state=cfg["seed"], **cfg["dracp"])
        for key, value in variant["changes"].items():
            setattr(vcfg, key, value)
        methods[variant["name"]] = DRACPRegressor(vcfg, base_model=base)
    return methods

def main(config_path):
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    result_dir = ensure_dir(cfg["output_dir"])
    metrics = []

    for s_idx, scenario in enumerate(cfg["scenarios"]):
        df = generate_scenario(
            scenario, cfg["n_samples"], cfg["n_features"],
            cfg["seed"] + s_idx
        )
        feature_cols = [c for c in df.columns if c.startswith("x")]
        train, cal, test = chronological_split(
            df, cfg["train_fraction"], cfg["calibration_fraction"]
        )
        methods = make_methods(cfg)
        for name, method in methods.items():
            method.fit(
                train[feature_cols].to_numpy(), train["y"].to_numpy(),
                cal[feature_cols].to_numpy(), cal["y"].to_numpy(),
            )
            pred = sequential_evaluate(
                method, test[feature_cols].to_numpy(), test["y"].to_numpy()
            )
            pred["regime_true"] = test["regime_true"].to_numpy()
            metrics.append(save_run(
                pred, result_dir, name, scenario, cfg["alpha"]
            ))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(result_dir / "metrics.csv", index=False)
    make_all_plots(result_dir, metrics_df)
    write_metadata(result_dir / "run_metadata.json", cfg)
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
