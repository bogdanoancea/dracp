from __future__ import annotations
import argparse, gzip, io
from pathlib import Path
import pandas as pd
import requests

DEFAULT_BULK_URL = (
    "https://ec.europa.eu/eurostat/api/dissemination/"
    "statistics/1.0/data/prc_hicp_manr"
)

def main(args):
    params = {
        "geo": args.geo,
        "coicop": args.coicop,
        "unit": args.unit,
        "lang": "en",
    }
    response = requests.get(args.url, params=params, timeout=120)
    response.raise_for_status()
    js = response.json()
    time_index = js["dimension"]["time"]["category"]["index"]
    ordered = sorted(time_index.items(), key=lambda kv: kv[1])
    values = js.get("value", {})
    rows = []
    for label, position in ordered:
        value = values.get(str(position))
        if value is not None:
            rows.append({"date": label, "inflation": value})
    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/raw/inflation.csv")
    p.add_argument("--geo", default="RO")
    p.add_argument("--coicop", default="CP00")
    p.add_argument("--unit", default="RCH_A")
    p.add_argument("--url", default=DEFAULT_BULK_URL)
    main(p.parse_args())
