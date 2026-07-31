#!/usr/bin/env bash
# ============================================================================
# Run the reviewer-requested experiments.
#
#   chmod +x run_reviewer.sh
#   ./run_reviewer.sh                    # all steps, quickest first
#   ./run_reviewer.sh euro-area runtime  # only selected steps
#
# Steps (in default order):
#   euro-area    true euro-area HICP aggregate (fixes the geo=RO bug)   minutes
#   runtime      runtime + peak memory + hardware metadata              minutes
#   sensitivity  ESS floor, regimes, clipping, weight cap               ~30-45 min
#   ablation     component ablation on the REAL panel                   hours
#   horizons     multi-step evaluation h = 1,3,6,12                     hours
#   models       5 base forecasters x 7 conformal methods               longest
#
# Environment overrides:
#   SEEDS=20         seeds for euro-area / ablation
#   MODEL_SEEDS=10   seeds for models / horizons (kept lower: they are heavy)
#   LIMIT=           restrict heavy steps to the first N series, e.g. LIMIT=12
#   HORIZONS=1,3,6,12
#   PYTHON_BIN=python3   (macOS has no bare 'python'; python3 is the default)
#
# Each step runs independently: if one fails the others still run, and a summary
# is printed at the end. Re-running a step overwrites only its own output.
# ============================================================================
set -Eeuo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SEEDS="${SEEDS:-20}"
MODEL_SEEDS="${MODEL_SEEDS:-10}"
LIMIT="${LIMIT:-}"
HORIZONS="${HORIZONS:-1,3,6,12}"

# --- interpreter: prefer the project venv, never rely on a bare 'python' -----
PYBIN="$VENV_DIR/bin/python"
[[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]] && PYBIN="$VENV_DIR/Scripts/python.exe"
if [[ ! -x "$PYBIN" ]]; then
  echo "No usable virtual environment at '$VENV_DIR'."
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "ERROR: '$PYTHON_BIN' not found. On macOS use python3 (or set PYTHON_BIN)."; exit 1; }
  echo "Creating it with $PYTHON_BIN ..."
  rm -rf "$VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$PYBIN" -m pip install --quiet --upgrade pip
  "$PYBIN" -m pip install --quiet -e ".[experiments]"
fi

# --- sanity: must be run from the repository root ---------------------------
for f in experiments/common.py experiments/reviewer_experiments.py configs/economic_manifest.json; do
  [[ -f "$f" ]] || { echo "ERROR: '$f' not found. Run this from the dracp repository root."; exit 1; }
done
"$PYBIN" -c "import dracp" 2>/dev/null || {
  echo "ERROR: the 'dracp' package is not importable in $VENV_DIR."
  echo "       Run: $PYBIN -m pip install -e \".[experiments]\""; exit 1; }

ALL=(euro-area runtime sensitivity ablation horizons models)
if [[ $# -gt 0 ]]; then STEPS=("$@"); else STEPS=("${ALL[@]}"); fi

mkdir -p logs results/reviewer
TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY=()

run_step () {
  local name="$1"; shift
  local log="logs/reviewer_${name}_${TS}.log"
  echo
  echo "============================================================"
  echo "# ${name}   (log: ${log})"
  echo "============================================================"
  local t0=$SECONDS
  if "$PYBIN" -m experiments.reviewer_experiments "$name" "$@" 2>&1 | tee "$log"; then
    SUMMARY+=("OK      ${name}  ($(( (SECONDS-t0)/60 )) min)")
  else
    SUMMARY+=("FAILED  ${name}  -- see ${log}")
    echo "  (continuing with the remaining steps)"
  fi
}

want () { for s in "${STEPS[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

want euro-area   && run_step euro-area   --seeds "$SEEDS"
want runtime     && run_step runtime
want sensitivity && run_step sensitivity
if want ablation; then
  if [[ -n "$LIMIT" ]]; then run_step ablation --seeds "$SEEDS" --limit "$LIMIT"
  else                       run_step ablation --seeds "$SEEDS"; fi
fi
if want horizons; then
  if [[ -n "$LIMIT" ]]; then run_step horizons --seeds "$MODEL_SEEDS" --horizons "$HORIZONS" --limit "$LIMIT"
  else                       run_step horizons --seeds "$MODEL_SEEDS" --horizons "$HORIZONS"; fi
fi
if want models; then
  if [[ -n "$LIMIT" ]]; then run_step models --seeds "$MODEL_SEEDS" --limit "$LIMIT"
  else                       run_step models --seeds "$MODEL_SEEDS"; fi
fi

echo
echo "============================================================"
echo "# SUMMARY"
echo "============================================================"
printf '%s\n' "${SUMMARY[@]}"
echo
echo "Outputs to send back:"
echo "  results/euro_area_inflation/mc_summary.csv"
echo "  results/reviewer/real_ablation.csv"
echo "  results/reviewer/base_models.csv"
echo "  results/reviewer/horizons.csv"
echo "  results/reviewer/sensitivity_extra.csv"
echo "  results/reviewer/runtime.csv   results/reviewer/runtime_hardware.json"
