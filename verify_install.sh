#!/usr/bin/env bash
# Confirms the working copy has every v0.3.1 fix. Run from the repo root.
cd "$(dirname "$0")"
fail=0
chk () {  # name, file, pattern, expected-minimum
  # NB: grep -c prints 0 AND exits non-zero on no-match, so capture without
  # appending a fallback, then default an empty result to 0.
  n=$(grep -c "$3" "$2" 2>/dev/null || true)
  [ -z "$n" ] && n=0
  if [ "$n" -ge "$4" ]; then printf '  OK    %s\n' "$1"
  else printf '  MISS  %s\n          %s: found %s, need >=%s\n' "$1" "$2" "$n" "$4"; fail=1; fi
}
echo "Checking DRACP v0.3.1 install:"
chk "delayed-feedback evaluator"     experiments/common.py         'horizon=1'                 1
chk "scale ablation (build_methods)" experiments/common.py         'dracp_no_scale'            1
chk "scale ablation (ablation cmd)"  experiments/extended.py       'dracp_no_scale'            2
chk "paired bootstrap module"        dracp/bootstrap.py            'politis_white'             1
chk "regime transition matrix"       dracp/regimes.py              'transition_'               1
chk "horizon-aware config"           dracp/core.py                 'horizon_adapt'             1
chk "comparable horizon ranks"       reporting/extended_report.py  'defined at every horizon'  1
chk "label_column fix"               reporting/extended_report.py  'def label_column'          1
chk "Type 42 fonts"                  reporting/paper_outputs.py    'pdf.fonttype'              1
echo
if [ "$fail" -eq 0 ]; then
  echo "All checks passed - safe to run the experiments."
else
  echo "INSTALL INCOMPLETE - do not start the long runs yet."
  echo "Run ./install_over.sh /path/to/extracted/dracp-v2 to force-copy the changed files."
  exit 1
fi
