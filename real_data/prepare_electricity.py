from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main(args):
    df = pd.read_csv(args.input, sep=";", decimal=",", index_col=0)
    df.index = pd.to_datetime(df.index)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if args.aggregate == "sum":
        load = numeric.sum(axis=1)
    else:
        load = numeric.mean(axis=1)
    hourly = load.resample("h").mean().rename("load").to_frame()
    hourly["hour_sin"] = np.sin(2*np.pi*hourly.index.hour/24)
    hourly["hour_cos"] = np.cos(2*np.pi*hourly.index.hour/24)
    hourly["dow_sin"] = np.sin(2*np.pi*hourly.index.dayofweek/7)
    hourly["dow_cos"] = np.cos(2*np.pi*hourly.index.dayofweek/7)
    for lag in [1,2,3,24,48,168]:
        hourly[f"load_lag_{lag}"] = hourly["load"].shift(lag)
    hourly["load_roll_24"] = hourly["load"].shift(1).rolling(24).mean()
    hourly["load_roll_168"] = hourly["load"].shift(1).rolling(168).mean()
    hourly = hourly.dropna().reset_index().rename(columns={"index":"date"})
    if args.max_rows:
        hourly = hourly.tail(args.max_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(args.output, index=False)
    print(f"Wrote {len(hourly)} hourly rows to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--aggregate", choices=["mean","sum"], default="mean")
    p.add_argument("--max-rows", type=int, default=20000)
    main(p.parse_args())
