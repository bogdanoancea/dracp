"""Download HICP annual inflation for all EU-27 countries from Eurostat.
Writes data/raw/inflation_<GEO>.csv for each. Reuses the single-country loader.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from .download_inflation import main as dl_one

EU27 = ["AT","BE","BG","CY","CZ","DE","DK","EE","EL","ES","FI","FR","HR","HU",
        "IE","IT","LT","LU","LV","MT","NL","PL","PT","RO","SE","SI","SK"]

def main(args):
    geos = args.geos.split(",") if args.geos else EU27
    for g in geos:
        out = Path(args.output_dir)/f"inflation_{g}.csv"
        try:
            ns = argparse.Namespace(output=str(out), geo=g, coicop="CP00",
                                    unit="RCH_A", url=
                "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_manr")
            dl_one(ns)
        except Exception as e:
            print(f"[warn] {g}: {e}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="data/raw")
    p.add_argument("--geos", default="", help="comma-separated Eurostat geos; default EU-27")
    main(p.parse_args())
