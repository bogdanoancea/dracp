"""Download one or more FRED economic time series as CSV.

Uses the public fredgraph CSV endpoint (no API key required):
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES

Examples:
    python -m real_data.download_fred --series UNRATE --output data/raw/UNRATE.csv
    python -m real_data.download_fred --series INDPRO CPIAUCSL GS10 \
        --output-dir data/raw/fred
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import requests

BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

def download_series(series_id: str, timeout: int = 120) -> pd.DataFrame:
    r = requests.get(BASE, params={"id": series_id}, timeout=timeout)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    # FRED CSV has columns like ['DATE','UNRATE'] or ['observation_date','UNRATE']
    date_col = df.columns[0]
    val_col = df.columns[1]
    df = df.rename(columns={date_col: "date", val_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def main(args):
    if args.output_dir:
        outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    for sid in args.series:
        df = download_series(sid)
        if args.output and len(args.series) == 1:
            out = Path(args.output)
        else:
            out = Path(args.output_dir or "data/raw") / f"{sid}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"{sid}: wrote {len(df)} rows ({df['date'].min().date()} -> "
              f"{df['date'].max().date()}) to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--series", nargs="+", required=True, help="FRED series id(s)")
    p.add_argument("--output", default=None, help="output CSV (single series)")
    p.add_argument("--output-dir", default=None, help="output dir (multiple series)")
    main(p.parse_args())
