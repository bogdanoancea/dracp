#!/usr/bin/env bash
# ============================================================================
# DRACP -- single entry point for the entire experimental pipeline.
#
#   chmod +x run.sh
#   ./run.sh                 # run every stage, in order
#   ./run.sh montecarlo report   # run only selected stages
#
# Stages (canonical order):
#   setup       create venv, install package, (optionally) run unit tests
#   synthetic   full synthetic benchmark  -> results/synthetic
#   real        download+prepare+run electricity, inflation, and the FRED
#               economic/financial series (single run)  -> results/<name>
#   sweeps      DRACP hyperparameter sensitivity sweeps -> results/sweeps_all.csv
#   montecarlo  20-seed Monte-Carlo with error bars for every dataset
#   report      analysis tables + all figures + MC aggregation/plots
#
# Environment overrides:
#   SEEDS=20                Monte-Carlo seeds                (default 20)
#   SKIP_DOWNLOADS=0        reuse already-downloaded raw data
#   NO_ONLINE=0             drop FACI/SAOCP/Conformal-PID baselines
#   RUN_TESTS=1             run pytest during setup
#   MAX_ELECTRICITY_ROWS=20000
#   INFLATION_GEO=RO
#   PYTHON_BIN=python3
#
# Runtime note: 'montecarlo' is by far the heaviest stage (DRACP and dracp_asym
# dominate, and the daily series add up). Start with e.g. SEEDS=5 to gauge
# wall-clock, then scale up. Every stage is safe to re-run.
# ============================================================================
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"; VENV_DIR="${VENV_DIR:-.venv}"
SEEDS="${SEEDS:-20}"; SKIP_DOWNLOADS="${SKIP_DOWNLOADS:-0}"; NO_ONLINE="${NO_ONLINE:-0}"
RUN_TESTS="${RUN_TESTS:-1}"; MAX_ELECTRICITY_ROWS="${MAX_ELECTRICITY_ROWS:-20000}"
INFLATION_GEO="${INFLATION_GEO:-RO}"
ONLINE_FLAG=""; [[ "$NO_ONLINE" == "1" ]] && ONLINE_FLAG="--no-online"

ALL_STAGES=(setup synthetic real sweeps montecarlo report)
if [[ $# -gt 0 ]]; then STAGES=("$@"); else STAGES=("${ALL_STAGES[@]}"); fi
want() { for s in "${STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

mkdir -p data/raw/fred data/processed results logs
TS="$(date +%Y%m%d_%H%M%S)"; LOG="logs/run_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
section(){ echo; echo "============================================================"; echo "# $1"; echo "============================================================"; }
trap 'echo; echo "ERROR: stage failed (exit $?). See $LOG"; exit 1' ERR

# ---------------------------------------------------------------- setup
if want setup; then
  section "SETUP: environment + install"
  [[ -d "$VENV_DIR" ]] || "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --quiet --upgrade pip setuptools wheel
  python -m pip install --quiet -e .
  python -m pip install --quiet pytest
  python --version
  [[ "$RUN_TESTS" == "1" ]] && { section "SETUP: unit tests"; python -m pytest -q; }
else
  # still need the venv active for later stages
  # shellcheck disable=SC1091
  [[ -d "$VENV_DIR" ]] && source "$VENV_DIR/bin/activate" || { echo "No venv; run the 'setup' stage first."; exit 1; }
fi

# ---------------------------------------------------------------- synthetic
if want synthetic; then
  section "SYNTHETIC: full benchmark"
  python -m experiments.run_synthetic --config configs/synthetic.yaml
  python -m reporting.create_tables --results-dir results/synthetic
fi

# ---------------------------------------------------------------- real (single run)
if want real; then
  section "REAL: inflation (Eurostat)"
  if [[ "$SKIP_DOWNLOADS" != "1" || ! -f data/raw/inflation_${INFLATION_GEO}.csv ]]; then
    python -m real_data.download_inflation --output "data/raw/inflation_${INFLATION_GEO}.csv" --geo "$INFLATION_GEO" || true
  fi
  if [[ -f data/raw/inflation_${INFLATION_GEO}.csv ]]; then
    python -m real_data.prepare_inflation --input "data/raw/inflation_${INFLATION_GEO}.csv" \
      --output data/processed/inflation.csv --date-column date --target-column inflation
    python -m experiments.run_real --config configs/inflation.yaml
    python -m reporting.create_tables --results-dir results/inflation
  fi

  section "REAL: EU-27 inflation (Eurostat)"
  if [[ "$SKIP_DOWNLOADS" != "1" ]]; then
    python -m real_data.download_inflation_eu --output-dir data/raw || true
  fi
  python -m real_data.prepare_and_run_eu || true

  section "REAL: electricity (UCI)"
  if [[ "$SKIP_DOWNLOADS" != "1" ]]; then
    python -m real_data.download_electricity --output-dir data/raw/electricity || true
  fi
  ESRC="$(find data/raw/electricity -type f -name 'LD2011_2014.txt' 2>/dev/null | head -n1 || true)"
  if [[ -n "$ESRC" ]]; then
    python -m real_data.prepare_electricity --input "$ESRC" --output data/processed/electricity.csv \
      --aggregate mean --max-rows "$MAX_ELECTRICITY_ROWS"
    python -m experiments.run_real --config configs/electricity.yaml
    python -m reporting.create_tables --results-dir results/electricity
  fi

  section "REAL: FRED economic/financial series"
  SKIP_DOWNLOADS="$SKIP_DOWNLOADS" python - <<'PY'
import json, os, subprocess, sys, pathlib
man=json.load(open("configs/economic_manifest.json"))
skip=os.environ.get("SKIP_DOWNLOADS","0")=="1"
for d in man:
    raw=f"data/raw/fred/{d['fred_id']}.csv"; proc=f"data/processed/{d['name']}.csv"
    if not (skip and pathlib.Path(raw).exists()):
        subprocess.check_call([sys.executable,"-m","real_data.download_fred","--series",d["fred_id"],"--output",raw])
    subprocess.check_call([sys.executable,"-m","real_data.prepare_economic",
        "--input",raw,"--output",proc,"--target",d["target"],"--freq",d["freq"],
        "--transform",d["transform"],"--lags",d["lags"],"--roll",d["roll"],
        "--max-rows",str(d.get("max_rows",0))])
    subprocess.check_call([sys.executable,"-m","experiments.run_real","--config",f"configs/{d['name']}.yaml"])
    subprocess.check_call([sys.executable,"-m","reporting.create_tables","--results-dir",f"results/{d['name']}"])
print("All FRED datasets done.")
PY

  section "REAL: combine single-run metrics"
  python - <<'PY'
import glob, pandas as pd, pathlib
rows=[]
for p in sorted(glob.glob("results/*/metrics.csv")):
    exp=pathlib.Path(p).parent.name
    if exp=="synthetic_smoke": continue
    d=pd.read_csv(p); d.insert(0,"experiment",exp); rows.append(d)
if rows:
    pd.concat(rows,ignore_index=True,sort=False).to_csv("results/all_experiments_metrics.csv",index=False)
    print("Wrote results/all_experiments_metrics.csv")
PY
fi

# ---------------------------------------------------------------- sweeps
if want sweeps; then
  section "SWEEPS: DRACP hyperparameter sensitivity"
  rm -f results/sweeps_all.csv
  python -m experiments.sensitivity
fi

# ---------------------------------------------------------------- montecarlo
if want montecarlo; then
  section "MONTE-CARLO: synthetic (${SEEDS} seeds)"
  python -m experiments.montecarlo --config configs/synthetic.yaml --kind synthetic --seeds "$SEEDS" $ONLINE_FLAG
  section "MONTE-CARLO: real datasets (${SEEDS} seeds)"
  REALCFGS=($(python -m experiments.list_real_configs))
  for cfg in "${REALCFGS[@]}"; do
    [[ -f "data/processed/${cfg}.csv" ]] || { echo "skip ${cfg} (no processed data)"; continue; }
    python -m experiments.montecarlo --config "configs/${cfg}.yaml" --kind real --seeds "$SEEDS" $ONLINE_FLAG
  done
fi

# ---------------------------------------------------------------- report
if want report; then
  section "REPORT: Monte-Carlo aggregation + error-bar figures"
  ls results/*/mc_summary.csv >/dev/null 2>&1 && python -m reporting.mc_report || echo "skip mc_report (no mc_summary.csv yet)"

  section "REPORT: Diebold-Mariano tests (per series)"
  ls results/*/*__dracp__predictions.csv >/dev/null 2>&1 && python -m reporting.dm_test --results-root results --alpha 0.10 || echo "skip dm_test (needs the real stage)"

  # regenerates every table and figure of the paper (uses dm_results.csv if present)
  section "REPORT: paper tables and figures"
  [[ -f results/mc_all_summary.csv ]] && python -m reporting.paper_outputs --results-root results --out results/paper || echo "skip paper_outputs (run the montecarlo stage first)"
fi

section "PIPELINE COMPLETE"
echo "Stages run: ${STAGES[*]}"
echo "Key outputs:"
echo "  results/all_experiments_metrics.csv   (single-run metrics)"
echo "  results/mc_all_summary.csv            (Monte-Carlo means + CIs)"
echo "  results/*/mc_block_bootstrap.csv      (bootstrap CIs, real datasets)"
echo "  results/mc_figures/                   (error-bar figures)"
echo "  results/paper/tables/, results/paper/figures/  (paper tables and figures)"
echo "  results/dm_results.csv, results/sweeps_all.csv"
echo "Log: $LOG"
