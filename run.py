#!/usr/bin/env python3
"""DRACP experimental pipeline -- single cross-platform entry point.

Works on Windows, macOS and Linux. Creates its own virtual environment, installs
the package, and runs the requested stages.

    python run.py                       # tier 'main' (what the paper needs)
    python run.py --tier quick          # smoke test, minutes
    python run.py --tier full           # everything, including sweeps
    python run.py montecarlo report     # only selected stages

Tiers
    quick   setup + synthetic + report                    (minutes)
    main    + real panel, repeated fits, horizons, extended experiments
    full    + hyperparameter sweeps and the 5-forecaster grid

Stages
    setup       create venv, install package, run unit tests
    synthetic   full synthetic benchmark          -> results/synthetic
    real        download/prepare/run electricity, inflation, EU-27 and FRED series
    sweeps      hyperparameter sensitivity sweeps -> results/sweeps_all.csv
    montecarlo  repeated fits with paired bootstrap intervals for every dataset
    extended    real-panel ablation, base-forecaster grid, multi-horizon
                (h = 1, 3, 6, 12 including the horizon-aware variants),
                extra sensitivity, runtime benchmark
    report      aggregation, Diebold-Mariano tests, paper tables and figures

All results are written to a single results/ folder; there is no separate
reviewer/ subfolder.

Environment variables
    SEEDS=20                 Monte-Carlo seeds
    SKIP_DOWNLOADS=0         reuse already-downloaded raw data
    NO_ONLINE=0              drop FACI/SAOCP/Conformal-PID baselines
    RUN_TESTS=1              run pytest during setup
    MAX_ELECTRICITY_ROWS=20000
    INFLATION_GEO=EA20
    HORIZONS=1,3,6,12        horizons for the extended stage
    MODEL_SEEDS=10           seeds for the base-forecaster grid
    PYTHON_BIN=python3       interpreter used to create the venv
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALL_STAGES = ["setup", "synthetic", "real", "sweeps", "montecarlo",
              "extended", "report"]

# Tiers group the stages into three run sizes. The point is that you can
# validate the pipeline in minutes before committing hours to it.
TIERS = {
    # smoke test: does everything import, run and write output?
    "quick": ["setup", "synthetic", "report"],
    # what the paper needs: the full 48-series panel with repeated fits,
    # the multi-horizon evaluation, and all reporting
    "main":  ["setup", "synthetic", "real", "montecarlo", "extended", "report"],
    # everything, including hyperparameter sweeps and the 5-forecaster grid
    "full":  ALL_STAGES,
}

def env(name, default):
    return os.environ.get(name, default)

SEEDS       = env("SEEDS", "20")
SKIP_DL     = env("SKIP_DOWNLOADS", "0") == "1"
NO_ONLINE   = env("NO_ONLINE", "0") == "1"
RUN_TESTS   = env("RUN_TESTS", "1") == "1"
MAX_ELEC    = env("MAX_ELECTRICITY_ROWS", "20000")
INFL_GEO    = env("INFLATION_GEO", "EA20")
HORIZONS    = env("HORIZONS", "1,3,6,12")
MC_SEEDS_H  = env("HORIZON_SEEDS", "10")
MODEL_SEEDS = env("MODEL_SEEDS", "10")
VENV        = ROOT / env("VENV_DIR", ".venv")
ONLINE_FLAG = ["--no-online"] if NO_ONLINE else []

def venv_python(venv: Path) -> Path:
    """Interpreter inside a venv (Windows uses Scripts/python.exe)."""
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

LOG = None
def section(title: str):
    line = "=" * 60
    msg = f"\n{line}\n# {title}\n{line}"
    print(msg, flush=True)
    if LOG: LOG.write(msg + "\n"); LOG.flush()

def run(args, check=True, allow_fail=False):
    printable = " ".join(str(a) for a in args)
    print(f"$ {printable}", flush=True)
    if LOG: LOG.write(f"$ {printable}\n"); LOG.flush()
    try:
        subprocess.run([str(a) for a in args], check=check, cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        if allow_fail:
            print(f"  (non-fatal: exit {exc.returncode})", flush=True)
            return False
        raise
    return True

def ensure_venv() -> Path:
    py = venv_python(VENV)
    if not py.exists():
        if VENV.exists():
            print(f"Existing '{VENV.name}' has no usable interpreter; recreating.")
            shutil.rmtree(VENV, ignore_errors=True)
        base = env("PYTHON_BIN", sys.executable)
        if shutil.which(base) is None and not Path(base).exists():
            sys.exit(f"ERROR: interpreter '{base}' not found; set PYTHON_BIN.")
        subprocess.run([base, "-m", "venv", str(VENV)], check=True)
    return py

# --------------------------------------------------------------------- stages
def stage_setup(py):
    section("SETUP: environment + install")
    run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel"])
    run([py, "-m", "pip", "install", "--quiet", "-e", ".[experiments]"])
    run([py, "-m", "pip", "install", "--quiet", "pytest"])
    run([py, "--version"])
    if RUN_TESTS:
        section("SETUP: unit tests")
        run([py, "-m", "pytest", "-q"])

def stage_synthetic(py):
    section("SYNTHETIC: full benchmark")
    run([py, "-m", "experiments.run_synthetic", "--config", "configs/synthetic.yaml"])
    run([py, "-m", "reporting.create_tables", "--results-dir", "results/synthetic"])

def stage_real(py):
    section("REAL: inflation (Eurostat)")
    raw = ROOT / f"data/raw/inflation_{INFL_GEO}.csv"
    if not SKIP_DL or not raw.exists():
        run([py, "-m", "real_data.download_inflation", "--output", str(raw),
             "--geo", INFL_GEO], allow_fail=True)
    if raw.exists():
        run([py, "-m", "real_data.prepare_inflation", "--input", str(raw),
             "--output", "data/processed/inflation.csv",
             "--date-column", "date", "--target-column", "inflation"])
        run([py, "-m", "experiments.run_real", "--config", "configs/inflation.yaml"])
        run([py, "-m", "reporting.create_tables", "--results-dir", "results/inflation"])

    section("REAL: EU-27 inflation (Eurostat)")
    if not SKIP_DL:
        run([py, "-m", "real_data.download_inflation_eu", "--output-dir", "data/raw"],
            allow_fail=True)
    run([py, "-m", "real_data.prepare_and_run_eu"], allow_fail=True)

    section("REAL: electricity (UCI)")
    if not SKIP_DL:
        run([py, "-m", "real_data.download_electricity",
             "--output-dir", "data/raw/electricity"], allow_fail=True)
    src = next((p for p in (ROOT/"data/raw/electricity").rglob("LD2011_2014.txt")), None)
    if src:
        run([py, "-m", "real_data.prepare_electricity", "--input", str(src),
             "--output", "data/processed/electricity.csv",
             "--aggregate", "mean", "--max-rows", MAX_ELEC])
        run([py, "-m", "experiments.run_real", "--config", "configs/electricity.yaml"])
        run([py, "-m", "reporting.create_tables", "--results-dir", "results/electricity"])
    else:
        print("  electricity source not found; skipping")

    section("REAL: FRED economic/financial series")
    man = json.loads((ROOT/"configs/economic_manifest.json").read_text())
    for d in man:
        raw = ROOT/f"data/raw/fred/{d['fred_id']}.csv"
        if not (SKIP_DL and raw.exists()):
            if not run([py, "-m", "real_data.download_fred", "--series", d["fred_id"],
                        "--output", str(raw)], allow_fail=True):
                continue
        run([py, "-m", "real_data.prepare_economic", "--input", str(raw),
             "--output", f"data/processed/{d['name']}.csv", "--target", d["target"],
             "--freq", d["freq"], "--transform", d["transform"], "--lags", d["lags"],
             "--roll", d["roll"], "--max-rows", str(d.get("max_rows", 0))])
        run([py, "-m", "experiments.run_real", "--config", f"configs/{d['name']}.yaml"])
        run([py, "-m", "reporting.create_tables", "--results-dir", f"results/{d['name']}"])

    section("REAL: combine single-run metrics")
    combine(py)

def combine(py):
    code = (
        "import glob,pathlib,pandas as pd;"
        "rows=[];"
        "[rows.append(pd.read_csv(p).assign(experiment=pathlib.Path(p).parent.name))"
        " for p in sorted(glob.glob('results/*/metrics.csv'))"
        " if pathlib.Path(p).parent.name!='synthetic_smoke'];"
        "pd.concat(rows,ignore_index=True,sort=False)"
        ".to_csv('results/all_experiments_metrics.csv',index=False) if rows else None;"
        "print('wrote results/all_experiments_metrics.csv' if rows else 'no metrics found')"
    )
    run([py, "-c", code], allow_fail=True)

def stage_sweeps(py):
    section("SWEEPS: DRACP hyperparameter sensitivity")
    (ROOT/"results/sweeps_all.csv").unlink(missing_ok=True)
    run([py, "-m", "experiments.sensitivity"])

def stage_montecarlo(py):
    section(f"MONTE-CARLO: synthetic ({SEEDS} seeds)")
    run([py, "-m", "experiments.montecarlo", "--config", "configs/synthetic.yaml",
         "--kind", "synthetic", "--seeds", SEEDS] + ONLINE_FLAG)
    section(f"MONTE-CARLO: real datasets ({SEEDS} seeds)")
    names = ["inflation", "electricity"]
    names += [d["name"] for d in json.loads((ROOT/"configs/economic_manifest.json").read_text())]
    eu = ROOT/"configs/eu_inflation_manifest.json"
    if eu.exists():
        names += [d["name"] for d in json.loads(eu.read_text())]
    for name in names:
        if not (ROOT/f"data/processed/{name}.csv").exists():
            print(f"  skip {name} (no processed data)"); continue
        run([py, "-m", "experiments.montecarlo", "--config", f"configs/{name}.yaml",
             "--kind", "real", "--seeds", SEEDS] + ONLINE_FLAG, allow_fail=True)

def stage_extended(py):
    """Ablation, base-forecaster grid, horizons, sensitivity, runtime."""
    have_real = list((ROOT/"results").glob("*/*__dracp__predictions.csv"))
    if not have_real:
        print("  skip extended (needs the real stage first)")
        return
    section("EXTENDED: euro-area HICP aggregate")
    run([py, "-m", "experiments.extended", "euro-area"], allow_fail=True)
    section("EXTENDED: component ablation on the real panel")
    run([py, "-m", "experiments.extended", "ablation", "--seeds", SEEDS], allow_fail=True)
    section(f"EXTENDED: multi-horizon evaluation (h = {HORIZONS})")
    run([py, "-m", "experiments.extended", "horizons", "--seeds", MC_SEEDS_H,
         "--horizons", HORIZONS], allow_fail=True)
    section("EXTENDED: base-forecaster grid")
    run([py, "-m", "experiments.extended", "models", "--seeds", MODEL_SEEDS], allow_fail=True)
    section("EXTENDED: extra sensitivity")
    run([py, "-m", "experiments.extended", "sensitivity"], allow_fail=True)
    section("EXTENDED: runtime benchmark")
    run([py, "-m", "experiments.extended", "runtime"], allow_fail=True)


def stage_report(py):
    section("REPORT: Monte-Carlo aggregation + error-bar figures")
    if list((ROOT/"results").glob("*/mc_summary.csv")):
        run([py, "-m", "reporting.mc_report"], allow_fail=True)
    else:
        print("  skip mc_report (no mc_summary.csv yet)")
    section("REPORT: Diebold-Mariano tests (per series)")
    if list((ROOT/"results").glob("*/*__dracp__predictions.csv")):
        run([py, "-m", "reporting.dm_test", "--results-root", "results",
             "--alpha", "0.10"], allow_fail=True)
    else:
        print("  skip dm_test (needs the real stage)")
    section("REPORT: extended experiments summary")
    run([py, "-m", "reporting.extended_report", "--results-root", "results"],
        allow_fail=True)
    section("REPORT: paper tables and figures")
    if (ROOT/"results/mc_all_summary.csv").exists():
        run([py, "-m", "reporting.paper_outputs", "--results-root", "results",
             "--out", "results/paper"], allow_fail=True)
        run([py, "-m", "reporting.diagnostic_figures", "--results-root", "results",
             "--out", "results/paper/figures"], allow_fail=True)
    else:
        print("  skip paper_outputs (run the montecarlo stage first)")

STAGES = {"setup": stage_setup, "synthetic": stage_synthetic, "real": stage_real,
          "sweeps": stage_sweeps, "montecarlo": stage_montecarlo,
          "extended": stage_extended, "report": stage_report}

def main(argv):
    global LOG
    tier = None
    argv = list(argv)
    if argv and argv[0] in ("--tier", "-t"):
        if len(argv) < 2:
            sys.exit("--tier requires a value: " + ", ".join(TIERS))
        tier, argv = argv[1], argv[2:]
    elif argv and argv[0].startswith("--tier="):
        tier, argv = argv[0].split("=", 1)[1], argv[1:]
    if tier is not None and tier not in TIERS:
        sys.exit(f"Unknown tier {tier!r}. Available: {', '.join(TIERS)}")
    stages = argv or TIERS[tier or "main"]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        sys.exit(f"Unknown stage(s): {', '.join(unknown)}\nAvailable: {', '.join(ALL_STAGES)}")
    for d in ("data/raw/fred", "data/processed", "results", "logs"):
        (ROOT/d).mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = ROOT/f"logs/run_{stamp}.log"
    LOG = open(logpath, "w", encoding="utf-8")

    if "setup" in stages:
        py = ensure_venv()
    else:
        py = venv_python(VENV)
        if not py.exists():
            sys.exit(f"No usable venv at '{VENV.name}'; run the 'setup' stage first "
                     f"(python run.py setup ...).")
    t0 = time.time()
    try:
        for s in ALL_STAGES:
            if s in stages:
                STAGES[s](py)
    except subprocess.CalledProcessError as exc:
        print(f"\nERROR: a stage failed (exit {exc.returncode}). See {logpath}")
        return exc.returncode
    section("PIPELINE COMPLETE")
    print(f"Stages run: {' '.join(s for s in ALL_STAGES if s in stages)}")
    if tier: print(f"Tier: {tier}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
    print("Key outputs:")
    for p in ("results/all_experiments_metrics.csv", "results/mc_all_summary.csv",
              "results/dm_results.csv", "results/sweeps_all.csv",
              "results/horizons.csv", "results/real_ablation.csv",
              "results/base_models.csv", "results/sensitivity_extra.csv",
              "results/runtime.csv",
              "results/paper/tables/", "results/paper/figures/", "results/mc_figures/"):
        print(f"  {p}")
    print(f"Log: {logpath}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
