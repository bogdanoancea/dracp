#!/usr/bin/env bash
# Force-copies the files changed in v0.3.1 over an existing checkout.
# Unlike rsync it does not compare timestamps, so it cannot silently skip a file.
#   ./install_over.sh /tmp/newpkg/dracp-v2
set -Eeuo pipefail
SRC="${1:?usage: ./install_over.sh /path/to/extracted/dracp-v2}"
DST="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC/experiments/common.py" ] || { echo "ERROR: $SRC does not look like the package root"; exit 1; }

FILES=(
  dracp/bootstrap.py dracp/regimes.py dracp/core.py dracp/aggregate.py
  experiments/common.py experiments/extended.py experiments/montecarlo.py
  reporting/extended_report.py reporting/paper_outputs.py reporting/diagnostic_figures.py
  reporting/mc_report.py reporting/plots.py
  run.py run.sh run.bat verify_install.sh
)
for f in "${FILES[@]}"; do
  if [ -f "$SRC/$f" ]; then
    mkdir -p "$DST/$(dirname "$f")"
    cp -f "$SRC/$f" "$DST/$f"
    printf '  copied  %s\n' "$f"
  else
    printf '  absent  %s (skipped)\n' "$f"
  fi
done
# remove the file that was renamed, so a stale import cannot shadow the new one
[ -f "$DST/experiments/reviewer_experiments.py" ] && rm -f "$DST/experiments/reviewer_experiments.py" && echo "  removed experiments/reviewer_experiments.py (renamed to extended.py)"
find "$DST" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "  cleared __pycache__"
echo
chmod +x "$DST/verify_install.sh" 2>/dev/null || true
"$DST/verify_install.sh"
