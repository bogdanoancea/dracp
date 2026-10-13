"""Collect Monte-Carlo summaries across experiments and draw error-bar figures.

Reads every results/<exp>/mc_summary.csv, concatenates them into
results/mc_all_summary.csv, and renders coverage / interval-score error-bar
plots (mean with 95% CI whiskers) into results/mc_figures/.
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ORDER = ["split_cp", "rolling_cp", "adaptive_cp", "faci", "saocp", "conformal_pid", "dracp"]
NICE = {"split_cp":"Split","rolling_cp":"Rolling","adaptive_cp":"Adaptive","faci":"FACI","saocp":"SAOCP","conformal_pid":"Conf-PID","dracp":"DRACP"}
COL = {"split_cp":"#888888","rolling_cp":"#1f77b4","adaptive_cp":"#2ca02c","faci":"#9467bd","saocp":"#8c564b","conformal_pid":"#e377c2","dracp":"#d62728"}

def main(results_root="results"):
    root = Path(results_root)
    frames = []
    for f in glob.glob(str(root / "*" / "mc_summary.csv")):
        d = pd.read_csv(f); d["experiment"] = Path(f).parent.name; frames.append(d)
    if not frames:
        raise SystemExit("No mc_summary.csv files found. Run experiments.montecarlo first.")
    alld = pd.concat(frames, ignore_index=True)
    alld.to_csv(root / "mc_all_summary.csv", index=False)
    print(f"Wrote {root/'mc_all_summary.csv'} ({len(alld)} rows)")

    figdir = root / "mc_figures"; figdir.mkdir(exist_ok=True)
    for metric, nom in [("coverage", 0.9), ("mean_interval_score", None)]:
        for scen, g in alld.groupby("scenario"):
            g = g[g["method"].isin(ORDER)].copy()
            g["ord"] = g["method"].map({m: i for i, m in enumerate(ORDER)})
            g = g.sort_values("ord")
            if g.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 3.6))
            x = np.arange(len(g))
            means = g[f"{metric}_mean"].to_numpy()
            lo = g[f"{metric}_ci_low"].to_numpy(); hi = g[f"{metric}_ci_high"].to_numpy()
            err = np.vstack([means - lo, hi - means])
            cols = [COL[m] for m in g["method"]]
            ax.bar(x, means, color=cols, alpha=0.8, yerr=err, capsize=4,
                   error_kw=dict(ecolor="black", lw=1))
            ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in g["method"]], rotation=20, ha="right", fontsize=8)
            if nom is not None:
                ax.axhline(nom, ls="--", color="k", lw=1)
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"{scen}: {metric.replace('_',' ')} (mean, 95% CI)")
            fig.tight_layout()
            out = figdir / f"{scen}__{metric}.pdf"
            fig.savefig(out, dpi=160); plt.close(fig)
    print(f"Wrote error-bar figures to {figdir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", default="results")
    main(p.parse_args().results_root)
