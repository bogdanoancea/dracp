"""Diebold--Mariano tests of interval-score equality between DRACP and each
baseline, per real dataset, from the saved single-run predictions.

For each dataset it reads results/<dataset>/<scenario>__<method>__predictions.csv,
computes the per-step interval (Winkler) score, forms the loss differential
d_t = score^{base}_t - score^{DRACP}_t, and reports the Diebold--Mariano (1995)
statistic with a Newey--West (HAC) long-run variance and the Harvey--Leybourne--
Newbold (1997) small-sample correction. A positive statistic favours DRACP.

    python -m reporting.dm_test --results-root results --alpha 0.1
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

def interval_score(y, lo, up, alpha):
    y, lo, up = map(lambda z: np.asarray(z, float), (y, lo, up))
    return (up - lo) + (2/alpha)*(lo - y)*(y < lo) + (2/alpha)*(y - up)*(y > up)

def dm_stat(d, h=1):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; T = len(d)
    if T < 8: return np.nan, np.nan
    dbar = d.mean()
    def gamma(k): return np.sum((d[k:]-dbar)*(d[:T-k]-dbar))/T
    lrv = gamma(0) + 2*sum((1-k/h)*gamma(k) for k in range(1, h))
    if lrv <= 0: return np.nan, np.nan
    dm = dbar/np.sqrt(lrv/T)
    corr = np.sqrt((T + 1 - 2*h + h*(h-1)/T)/T)   # Harvey-Leybourne-Newbold
    dm *= corr
    p = 2*(1 - stats.t.cdf(abs(dm), df=T-1))
    return float(dm), float(p)

def main(root, alpha, target="dracp"):
    root = Path(root)
    baselines = ["split_cp","rolling_cp","adaptive_cp","faci","saocp","conformal_pid"]
    rows = []
    for d in sorted(glob.glob(str(root/"*"))):
        d = Path(d)
        if not d.is_dir(): continue
        preds = glob.glob(str(d/"*__{}__predictions.csv".format(target)))
        if not preds: continue
        scen = os.path.basename(preds[0]).split("__")[0]
        def load(m):
            p = d/f"{scen}__{m}__predictions.csv"
            if not p.exists(): return None
            df = pd.read_csv(p)
            return interval_score(df["y"], df["lower"], df["upper"], alpha)
        st = load(target)
        if st is None: continue
        h = max(1, int(len(st)**(1/3)))
        for b in baselines:
            sb = load(b)
            if sb is None: continue
            n = min(len(st), len(sb))
            dm, p = dm_stat(sb[:n] - st[:n], h=h)
            rows.append({"dataset": d.name, "comparison": f"{b} vs {target}",
                         "dm_stat": dm, "p_value": p, "n": n})
    out = pd.DataFrame(rows)
    out.to_csv(root/"dm_results.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nWrote {root/'dm_results.csv'}  (positive statistic favours {target})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--target", default="dracp")
    a = ap.parse_args()
    main(a.results_root, a.alpha, a.target)
