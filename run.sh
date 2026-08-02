#!/usr/bin/env bash
# DRACP -- single entry point for every experiment.
#
#   ./run.sh                  # tier 'main': what the paper needs
#   ./run.sh --tier quick     # smoke test (minutes) -- run this first
#   ./run.sh --tier full      # everything, including hyperparameter sweeps
#   ./run.sh montecarlo report    # individual stages
#
# All results land in a single results/ directory.
#
# Environment variables (all optional):
#   SEEDS=20            repeated fits per series
#   HORIZON_SEEDS=10    seeds for the multi-horizon stage
#   MODEL_SEEDS=10      seeds for the base-forecaster grid
#   HORIZONS=1,3,6,12   horizons to evaluate
#   SKIP_DOWNLOADS=1    reuse already-downloaded raw data
#   PYTHON_BIN=python3  interpreter used to create the venv
#   VENV_DIR=.venv      virtual-environment directory
set -Eeuo pipefail

cd "$(dirname "$0")"

if [[ ! -f experiments/common.py ]]; then
  echo "ERROR: run this from the repository root (experiments/common.py not found)." >&2
  exit 1
fi

# --- locate a usable interpreter -------------------------------------------
PYBIN="${PYTHON_BIN:-}"
if [[ -z "$PYBIN" ]]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PYBIN="$c"; break; fi
  done
fi
if [[ -z "$PYBIN" ]]; then
  echo "ERROR: no python3/python on PATH. Set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-.venv}"
VENV_PY="$VENV_DIR/bin/python"
[[ -x "$VENV_PY" ]] || VENV_PY="$VENV_DIR/Scripts/python.exe"   # Git Bash on Windows

# A venv copied between machines or folders keeps an absolute interpreter path
# and silently stops working. Validate rather than trusting the directory.
if [[ -d "$VENV_DIR" && ! -x "$VENV_PY" ]]; then
  echo "Existing '$VENV_DIR' has no usable interpreter; recreating."
  rm -rf "$VENV_DIR"
fi

trap 'rc=$?; echo; echo "run.sh failed (exit $rc) at line $LINENO." >&2; exit $rc' ERR

export PYTHON_BIN="$PYBIN"
echo "Interpreter: $("$PYBIN" --version 2>&1)  ($PYBIN)"
exec "$PYBIN" run.py "$@"
