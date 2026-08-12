#!/usr/bin/env python3
"""Download and prepare the 48 real series -- data only, no experiments.

The `real` stage of run.py interleaves data preparation with the one-step
experiments. If you already have valid results/ from a previous run and only
need to regenerate data/processed/ (for example to re-run the multi-horizon or
ablation stages), running that stage would waste hours recomputing results you
already trust. This script performs only the download and preparation steps.

    python prepare_data.py                 # everything missing
    python prepare_data.py --only fred     # just the FRED series
    python prepare_data.py --force         # re-prepare even if present

Existing files are skipped unless --force is given, so it is safe to re-run
after an interruption.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"


def run(args, what):
    print(f"  $ {' '.join(str(a) for a in args)}", flush=True)
    try:
        subprocess.run([str(a) for a in args], check=True, cwd=ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"    FAILED ({what}): exit {e.returncode}", file=sys.stderr)
        return False


def have(name, force):
    p = PROC / f"{name}.csv"
    if p.exists() and not force:
        print(f"  skip {name} (already present)")
        return True
    return False


def do_fred(py, force):
    man = json.loads((ROOT / "configs/economic_manifest.json").read_text())
    print(f"\n== FRED economic and financial series ({len(man)}) ==")
    ok = 0
    for d in man:
        if have(d["name"], force):
            ok += 1
            continue
        raw = RAW / "fred" / f"{d['fred_id']}.csv"
        raw.parent.mkdir(parents=True, exist_ok=True)
        if not raw.exists():
            if not run([py, "-m", "real_data.download_fred", "--series", d["fred_id"],
                        "--output", str(raw)], d["name"]):
                continue
        if run([py, "-m", "real_data.prepare_economic", "--input", str(raw),
                "--output", f"data/processed/{d['name']}.csv", "--target", d["target"],
                "--freq", d["freq"], "--transform", d["transform"], "--lags", d["lags"],
                "--roll", d["roll"], "--max-rows", str(d.get("max_rows", 0))], d["name"]):
            ok += 1
    return ok


def do_eu(py, force):
    man_p = ROOT / "configs/eu_inflation_manifest.json"
    if not man_p.exists():
        print("\n== EU inflation: manifest missing, skipped ==")
        return 0
    man = json.loads(man_p.read_text())
    print(f"\n== EU-27 HICP inflation ({len(man)}) ==")
    todo = [d for d in man if not have(d["name"], force)]
    if todo:
        run([py, "-m", "real_data.download_inflation_eu", "--output-dir", "data/raw"], "eu bulk")
    ok = len(man) - len(todo)
    for d in todo:
        raw = RAW / f"{d['name']}.csv"
        if not raw.exists():
            print(f"    no raw file for {d['name']}, skipped", file=sys.stderr)
            continue
        if run([py, "-m", "real_data.prepare_inflation", "--input", str(raw),
                "--output", f"data/processed/{d['name']}.csv",
                "--date-column", "date", "--target-column", "inflation"], d["name"]):
            ok += 1
    return ok


def do_euro_area(py, force):
    print("\n== Euro-area HICP aggregate ==")
    if have("euro_area_inflation", force):
        return 1
    raw = RAW / "inflation_EA20.csv"
    if not raw.exists():
        if not run([py, "-m", "real_data.download_inflation", "--output", str(raw),
                    "--geo", "EA20"], "euro area"):
            return 0
    return int(run([py, "-m", "real_data.prepare_inflation", "--input", str(raw),
                    "--output", "data/processed/euro_area_inflation.csv",
                    "--date-column", "date", "--target-column", "inflation"], "euro area"))


def do_electricity(py, force):
    print("\n== Electricity demand (UCI) ==")
    if have("electricity", force):
        return 1
    src = next((p for p in (RAW / "electricity").rglob("LD2011_2014.txt")), None)
    if src is None:
        if not run([py, "-m", "real_data.download_electricity",
                    "--output-dir", "data/raw/electricity"], "electricity"):
            return 0
        src = next((p for p in (RAW / "electricity").rglob("LD2011_2014.txt")), None)
    if src is None:
        print("    source file not found after download", file=sys.stderr)
        return 0
    return int(run([py, "-m", "real_data.prepare_electricity", "--input", str(src),
                    "--output", "data/processed/electricity.csv",
                    "--aggregate", "mean", "--max-rows", "20000"], "electricity"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["fred", "eu", "euro-area", "electricity"], default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    PROC.mkdir(parents=True, exist_ok=True)
    before = len(list(PROC.glob("*.csv")))
    print(f"Preparing data into {PROC}")
    print(f"  csv files present before: {before}")
    steps = {"fred": do_fred, "eu": do_eu, "euro-area": do_euro_area,
             "electricity": do_electricity}
    for name, fn in steps.items():
        if a.only in (None, name):
            fn(py, a.force)
    after = sorted(p.stem for p in PROC.glob("*.csv"))
    print(f"\nDone. {len(after)} series in data/processed/ (was {before}).")
    if len(after) < 48:
        print("Expected 48. Missing series will simply be skipped by the experiments;")
        print("check the errors above if the count is well short.")
    return 0 if after else 1


if __name__ == "__main__":
    sys.exit(main())
