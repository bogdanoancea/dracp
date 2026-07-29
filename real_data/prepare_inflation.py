from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def make_lag_features(df, target, lags=(1,2,3,6,12)):
    for lag in lags:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    df[f"{target}_rolling_mean_3"] = df[target].shift(1).rolling(3).mean()
    df[f"{target}_rolling_mean_12"] = df[target].shift(1).rolling(12).mean()
    df["month_sin"] = np.sin(2*np.pi*df["date"].dt.month/12)
    df["month_cos"] = np.cos(2*np.pi*df["date"].dt.month/12)
    return df

def main(args):
    df = pd.read_csv(args.input)
    df = df.rename(columns={args.date_column: "date", args.target_column: "inflation"})
    df["date"] = pd.to_datetime(df["date"])
    df["inflation"] = pd.to_numeric(df["inflation"], errors="coerce")
    df = df.sort_values("date").drop_duplicates("date")
    df = make_lag_features(df, "inflation").dropna().reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--date-column", default="date")
    p.add_argument("--target-column", default="inflation")
    main(p.parse_args())
