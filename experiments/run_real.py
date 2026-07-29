from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from dracp import (
    DRACPConfig, DRACPRegressor, SplitConformalRegressor,
    RollingConformalRegressor, AdaptiveConformalRegressor,
)
from dracp.utils import chronological_split, ensure_dir, load_yaml, set_seed
from .common import sequential_evaluate, save_run, write_metadata, build_methods
from reporting.plots import make_all_plots

def main(config_path):
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    df = pd.read_csv(cfg["data"]["path"])
    date_col = cfg["data"].get("date_column")
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
    target = cfg["data"]["target_column"]
    features = cfg["data"]["feature_columns"]
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    train, cal, test = chronological_split(
        df, cfg["train_fraction"], cfg["calibration_fraction"]
    )

    methods = build_methods(cfg, cfg["seed"], include_online=True,
                            include_ablations=bool(cfg.get("ablations")))

    result_dir = ensure_dir(cfg["output_dir"])
    metrics = []
    for name, method in methods.items():
        method.fit(
            train[features].to_numpy(), train[target].to_numpy(),
            cal[features].to_numpy(), cal[target].to_numpy(),
        )
        dates = test[date_col].astype(str).tolist() if date_col else None
        pred = sequential_evaluate(
            method, test[features].to_numpy(),
            test[target].to_numpy(), dates=dates
        )
        metrics.append(save_run(
            pred, result_dir, name, cfg["name"], cfg["alpha"]
        ))
    mdf = pd.DataFrame(metrics)
    mdf.to_csv(result_dir / "metrics.csv", index=False)
    make_all_plots(result_dir, mdf)
    write_metadata(result_dir / "run_metadata.json", cfg)
    print(mdf.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
