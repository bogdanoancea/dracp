"""Summaries for the extended experiments, written to results/paper/tables.

Reads the flat files the extended stage writes into ``results/`` (horizons.csv,
real_ablation.csv, base_models.csv, sensitivity_extra.csv, runtime.csv) and
emits both a readable console summary and LaTeX table fragments.

Two reporting conventions are applied deliberately:

* Aggregation across series uses the **mean of per-series percentage changes**,
  never the ratio of pooled means. Interval scores are scale-dependent, so a
  pooled mean is dominated by whichever series has the largest units (hourly
  electricity demand, in this panel).
* Base-forecaster results are broken out **by group** as well as pooled, because
  the pooled figure conceals a large difference between the inflation panel and
  the rest.

    python -m reporting.extended_report --results-root results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HICP_PREFIX = "inflation_"
FINANCIAL = {"dgs10_daily", "term_spread", "usd_eur", "vix", "wti_oil",
             "sp500_daily", "fed_funds_daily"}


def group_of(series: str) -> str:
    if series.startswith(HICP_PREFIX) or series == "euro_area_inflation":
        return "HICP inflation"
    if series in FINANCIAL:
        return "Financial"
    return "US macro & energy"


def pct_change(a: pd.Series, b: pd.Series) -> float:
    """Mean per-series percentage change of ``a`` relative to ``b``."""
    m = (b != 0) & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float((100.0 * (a[m] - b[m]) / b[m]).mean())


def _ranks(df: pd.DataFrame, value: str, within: list[str]) -> pd.DataFrame:
    return (df.pivot_table(index=within, columns="method", values=value)
              .rank(axis=1))


# ------------------------------------------------------------------ horizons
def report_horizons(root: Path, out: Path) -> str:
    f = root / "horizons.csv"
    if not f.exists():
        return "  (no horizons.csv)"
    d = pd.read_csv(f)
    piv = d.pivot_table(index=["series", "horizon"], columns="method",
                        values="mean_interval_score")
    ranks = piv.rank(axis=1)
    lines = []
    order = [c for c in ["dracp", "dracp_h", "dracp_hedge", "faci", "conformal_pid",
                         "adaptive_cp", "saocp", "rolling_cp", "split_cp"]
             if c in piv.columns]
    lines.append("Average interval-score rank by horizon (1 = best):")
    rk = ranks.groupby("horizon").mean()[order].round(2)
    lines.append(rk.to_string())

    # the question the horizon experiment exists to answer
    lines.append("")
    lines.append("Horizon-aware vs horizon-unaware DRACP, and vs FACI")
    lines.append("(mean per-series % change; negative favours the first named):")
    rows = []
    for h, g in piv.groupby("horizon"):
        rec = {"horizon": h, "n_series": len(g)}
        if "dracp_h" in g and "dracp" in g:
            rec["dracp_h vs dracp"] = pct_change(g["dracp_h"], g["dracp"])
        if "dracp_h" in g and "faci" in g:
            rec["dracp_h vs faci"] = pct_change(g["dracp_h"], g["faci"])
        if "dracp_hedge" in g and "faci" in g:
            rec["hedge vs faci"] = pct_change(g["dracp_hedge"], g["faci"])
        if "dracp" in g and "faci" in g:
            rec["dracp vs faci"] = pct_change(g["dracp"], g["faci"])
        rows.append(rec)
    tab = pd.DataFrame(rows).set_index("horizon").round(2)
    lines.append(tab.to_string())

    (out / "horizons_rank.tex").write_text(
        "\n".join(" & ".join([str(i)] + [f"{v:.2f}" for v in r]) + r" \\"
                  for i, r in zip(rk.index, rk.to_numpy())), encoding="utf-8")
    tab.to_csv(out / "horizons_summary.csv")
    return "\n".join(lines)


# ------------------------------------------------------------- base models
def report_models(root: Path, out: Path) -> str:
    f = root / "base_models.csv"
    if not f.exists():
        return "  (no base_models.csv)"
    d = pd.read_csv(f)
    piv = d.pivot_table(index=["series", "base_model"], columns="method",
                        values="mean_interval_score")
    rmse = d.groupby(["series", "base_model"]).base_rmse.first()
    piv = piv.join(rmse).reset_index()
    piv["group"] = piv.series.map(group_of)
    best = piv.groupby("series").base_rmse.transform("min")
    piv["rmse_ratio"] = piv.base_rmse / best

    lines = ["Base-forecaster grid (mean per-series % change, negative favours DRACP):"]
    pooled = []
    for bm, g in piv.groupby("base_model"):
        pooled.append({"base_model": bm, "rmse_ratio": g.rmse_ratio.median(),
                       "dracp vs faci": pct_change(g["dracp"], g["faci"]),
                       "n": len(g)})
    pooled = pd.DataFrame(pooled).sort_values("rmse_ratio").set_index("base_model")
    lines.append(pooled.round(2).to_string())

    lines.append("")
    lines.append("By group -- the pooled figure hides large differences:")
    rows = []
    for bm, g in piv.groupby("base_model"):
        rec = {"base_model": bm}
        for grp, gg in g.groupby("group"):
            rec[f"{grp} RMSE"] = gg.rmse_ratio.median()
            rec[f"{grp} %"] = pct_change(gg["dracp"], gg["faci"])
        rows.append(rec)
    bygroup = pd.DataFrame(rows).set_index("base_model")
    lines.append(bygroup.round(2).to_string())
    bygroup.to_csv(out / "base_models_by_group.csv")
    pooled.to_csv(out / "base_models_pooled.csv")
    return "\n".join(lines)


# ---------------------------------------------------------------- ablation
def report_ablation(root: Path, out: Path) -> str:
    f = root / "real_ablation.csv"
    if not f.exists():
        return "  (no real_ablation.csv)"
    d = pd.read_csv(f)
    piv = d.pivot_table(index="series", columns="method", values="mean_interval_score")
    if "dracp" not in piv:
        return "  (real_ablation.csv has no 'dracp' column)"
    rows = []
    for m in piv.columns:
        if m == "dracp":
            continue
        rows.append({"removed": m,
                     "% worse without it": pct_change(piv[m], piv["dracp"]),
                     "series hurt": int((piv[m] > piv["dracp"]).sum()),
                     "n": int(piv[m].notna().sum())})
    t = pd.DataFrame(rows).sort_values("% worse without it", ascending=False).set_index("removed")
    t.to_csv(out / "real_ablation_summary.csv")
    return ("Component ablation on the real panel\n"
            "(positive = removing the component makes the score worse):\n"
            + t.round(2).to_string())


# --------------------------------------------------------------- bootstrap
def report_bootstrap(root: Path, out: Path) -> str:
    files = sorted(root.glob("*/mc_block_bootstrap.csv"))
    if not files:
        return "  (no mc_block_bootstrap.csv)"
    rows = []
    for f in files:
        d = pd.read_csv(f)
        d["series"] = f.parent.name
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    col = "d_mean_interval_score"
    if col not in d.columns:
        return "  (bootstrap files predate the paired implementation; re-run montecarlo)"
    sig = d[d[f"{col}_p"] < 0.05]
    n_series = d.series.nunique()
    blocks = d.get("block_length_score")
    lines = [f"Paired block bootstrap over {n_series} series "
             f"({int(d.n_fits.max())} fits pooled)."]
    if blocks is not None:
        lines.append(f"  data-driven block length: median {int(blocks.median())}, "
                     f"range {int(blocks.min())}-{int(blocks.max())}")
    lines.append(f"  paired differences significant at 5%: {len(sig)} of {len(d.dropna(subset=[col]))}")
    better = int((sig[col] > 0).sum()); worse = int((sig[col] < 0).sum())
    lines.append(f"    reference better in {better}, worse in {worse}")
    d.to_csv(out / "bootstrap_all_series.csv", index=False)
    return "\n".join(lines)


def main(root="results", out=None):
    root = Path(root)
    out = Path(out) if out else root / "paper" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    blocks = [
        ("MULTI-HORIZON", report_horizons(root, out)),
        ("BASE FORECASTERS", report_models(root, out)),
        ("REAL-PANEL ABLATION", report_ablation(root, out)),
        ("PAIRED BOOTSTRAP", report_bootstrap(root, out)),
    ]
    text = []
    for title, body in blocks:
        text.append("=" * 70)
        text.append(title)
        text.append("=" * 70)
        text.append(body)
        text.append("")
    s = "\n".join(text)
    print(s)
    (out / "extended_summary.txt").write_text(s, encoding="utf-8")
    print(f"Wrote {out/'extended_summary.txt'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.results_root, a.out)
