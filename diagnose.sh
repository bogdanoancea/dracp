#!/usr/bin/env bash
# Explains why an experiment produced no output. Run from the repository root.
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
echo "repo:        $(pwd)"
echo "interpreter: $PY"
echo
echo "--- where does 'dracp' import from? -----------------------------------"
"$PY" - <<'PYEOF'
import importlib.util, sys, pathlib
spec = importlib.util.find_spec("dracp")
if spec is None:
    print("  NOT INSTALLED. Run:  pip install -e '.[experiments]'")
    sys.exit()
loc = pathlib.Path(spec.origin).parent
here = pathlib.Path.cwd() / "dracp"
print(f"  resolves to: {loc}")
if loc.resolve() == here.resolve():
    print("  OK - editable install pointing at this repo; your edits take effect.")
else:
    print("  PROBLEM - this is NOT your repo copy (probably the PyPI release).")
    print("  Your local fixes are being ignored. Fix with:")
    print("      pip uninstall -y dracp && pip install -e '.[experiments]'")
try:
    from dracp.core import DRACPConfig
    print("  horizon support:", "yes" if "horizon" in DRACPConfig.__dataclass_fields__ else "NO - stale copy")
except Exception as e:
    print("  import failed:", e)
PYEOF
echo
echo "--- is the processed data present? ------------------------------------"
if [ -d data/processed ]; then
  n=$(ls data/processed/*.csv 2>/dev/null | wc -l | tr -d ' ')
  echo "  data/processed: $n csv files"
  [ "$n" -eq 0 ] && echo "  PROBLEM - no processed data. Copy it from your previous run."
else
  echo "  PROBLEM - data/processed does not exist."
  echo "  Copy it from your backup:  cp -r ../results_backup_*/../data/processed data/"
fi
echo
echo "--- how many series will the experiments actually use? ----------------"
"$PY" - <<'PYEOF'
import sys, pathlib
sys.path.insert(0, ".")
try:
    from experiments.extended import real_configs
    n = real_configs()
    print(f"  usable series: {len(n)}")
    if n: print(f"    first few: {n[:5]}")
except Exception as e:
    print("  could not evaluate:", type(e).__name__, e)
PYEOF
