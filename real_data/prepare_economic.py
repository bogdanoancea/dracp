"""Generic feature preparation for an economic time series.

Reads a two-column series (date, value), optionally applies a transform
(level / log / diff / pct_change / log_diff) to obtain the modelling target,
and builds autoregressive lag features, rolling summaries and seasonal
encodings suitable for the DRACP real-data pipeline.

Example (monthly unemployment rate, model the level):
    python -m real_data.prepare_economic \
        --input data/raw/UNRATE.csv --output data/processed/us_unemployment.csv \
        --target unemployment --freq M --transform level

Example (industrial production, model month-on-month log growth):
    python -m real_data.prepare_economic \
        --input data/raw/INDPRO.csv --output data/processed/us_indpro.csv \
        --target indpro --freq M --transform log_diff
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def apply_transform(s: pd.Series, transform: str) -> pd.Series:
    if transform == "level":
        return s
    if transform == "diff":
        return s.diff()
    if transform == "pct_change":
        return s.pct_change() * 100.0
    if transform == "log":
        return np.log(s)
    if transform == "log_diff":
        return np.log(s).diff() * 100.0
    raise ValueError(f"unknown transform: {transform}")

def make_features(df, target, freq, lags, roll):
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    for w in roll:
        df[f"{target}_rolling_mean_{w}"] = df[target].shift(1).rolling(w).mean()
        df[f"{target}_rolling_std_{w}"] = df[target].shift(1).rolling(w).std()
    fu = freq.upper()
    if fu.startswith("M"):
        period, unit = 12, df["date"].dt.month
    elif fu.startswith("Q"):
        period, unit = 4, df["date"].dt.quarter
    elif fu.startswith("W"):
        period, unit = 52, df["date"].dt.isocalendar().week.astype(int)
    elif fu.startswith("D"):
        period, unit = 7, df["date"].dt.dayofweek  # weekly seasonality
    else:
        period, unit = 12, df["date"].dt.month
    df["season_sin"] = np.sin(2 * np.pi * unit / period)
    df["season_cos"] = np.cos(2 * np.pi * unit / period)
    return df

def feature_columns(target, lags, roll):
    cols = [f"{target}_lag_{l}" for l in lags]
    for w in roll:
        cols += [f"{target}_rolling_mean_{w}", f"{target}_rolling_std_{w}"]
    cols += ["season_sin", "season_cos"]
    return cols

def main(args):
    df = pd.read_csv(args.input)
    # accept either (date,value) or custom column names
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    val_col = "value" if "value" in df.columns else df.columns[1]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if args.max_rows and args.max_rows > 0:
        df = df.tail(int(args.max_rows)).reset_index(drop=True)
    df[args.target] = apply_transform(pd.to_numeric(df[val_col], errors="coerce"),
                                      args.transform)
    lags = [int(x) for x in args.lags.split(",")]
    roll = [int(x) for x in args.roll.split(",")]
    df = make_features(df, args.target, args.freq, lags, roll)
    cols = feature_columns(args.target, lags, roll)
    keep = ["date", args.target] + cols
    df = df[keep].dropna().reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")
    print("feature_columns:", cols)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target", required=True, help="name for the modelling target")
    p.add_argument("--freq", default="M", help="D (daily), W (weekly), M (monthly) or Q (quarterly)")
    p.add_argument("--transform", default="level",
                   choices=["level", "diff", "pct_change", "log", "log_diff"])
    p.add_argument("--lags", default="1,2,3,6,12")
    p.add_argument("--roll", default="3,12")
    p.add_argument("--max-rows", type=int, default=0, help="keep only the last N raw observations (0 = all)")
    main(p.parse_args())
