"""Regenerate every table and figure of the DRACP paper from experiment output.

Reads
    <results>/mc_all_summary.csv      Monte-Carlo summary (all methods, all series)
    <results>/dm_results.csv          Diebold-Mariano tests (optional)
    <results>/<dataset>/*__<m>__predictions.csv   per-step predictions (optional,
                                      used for the inflation case study)

Writes
    <out>/tables/*.tex                LaTeX table bodies, \input-able
    <out>/figures/*.pdf               figures used in the paper

Usage
    python -m reporting.paper_outputs --results-root results --out results/paper
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- setup
METHODS = ["split_cp", "rolling_cp", "adaptive_cp", "faci", "saocp", "conformal_pid", "dracp"]
NICE = {"split_cp": "Split", "rolling_cp": "Rolling", "adaptive_cp": "Adaptive",
        "faci": "FACI", "saocp": "SAOCP", "conformal_pid": "C-PID", "dracp": "DRACP"}
SYN = ["stationary", "covariate_shift", "conditional_drift", "abrupt_drift",
       "gradual_drift", "regime_switching", "combined_shift"]
SYNL = dict(zip(SYN, ["Stationary", "Covariate", "Conditional", "Abrupt", "Gradual",
                      "Regime", "Combined"]))
FINANCIAL = ["us_10y_treasury", "us_pce_inflation", "dgs10_daily", "term_spread",
             "usd_eur", "vix", "wti_oil", "sp500_daily", "fed_funds_daily"]
FOLDER = {"electricity_demand": "electricity", "european_inflation": "inflation"}
LABEL = {"electricity_demand": "Electricity", "european_inflation": "Euro-area HICP",
         "us_unemployment": "US unemployment", "us_industrial_production": "US ind. production",
         "us_cpi_inflation": "US CPI", "us_retail_sales": "US retail sales",
         "us_nonfarm_payrolls": "US payrolls", "us_capacity_util": "US capacity util.",
         "us_housing_starts": "US housing starts", "us_m2": "US M2", "us_ppi": "US PPI",
         "us_consumer_sentiment": "US sentiment", "us_pce_inflation": "US PCE",
         "us_10y_treasury": "US 10Y yield", "dgs10_daily": "US 10Y (d)",
         "term_spread": "Term spread (d)", "usd_eur": "USD/EUR (d)", "vix": "VIX (d)",
         "wti_oil": "WTI oil (d)", "sp500_daily": "S\\&P 500 (d)",
         "fed_funds_daily": "Fed funds (d)"}


def label(s):
    if s.startswith("inflation_"):
        return "HICP " + s.split("_", 1)[1]
    return LABEL.get(s, s.replace("_", " "))


def fmt(v, coverage=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.3f}" if (coverage or abs(v) < 1) else f"{v:.2f}"


def interval_score(y, lo, up, alpha=0.1):
    y, lo, up = (np.asarray(z, float) for z in (y, lo, up))
    return (up - lo) + (2/alpha)*(lo - y)*(y < lo) + (2/alpha)*(y - up)*(y > up)


# --------------------------------------------------------------------------- core
class PaperOutputs:
    def __init__(self, results_root="results", out="results/paper", alpha=0.1):
        self.root = Path(results_root)
        self.tab = Path(out)/"tables"; self.fig = Path(out)/"figures"
        self.tab.mkdir(parents=True, exist_ok=True); self.fig.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.S = pd.read_csv(self.root/"mc_all_summary.csv")
        dm = self.root/"dm_results.csv"
        self.DM = pd.read_csv(dm) if dm.exists() else None
        if self.DM is not None:
            self.DM["base"] = self.DM["comparison"].str.replace(" vs dracp", "", regex=False)
        self.methods = [m for m in METHODS if m in set(self.S["method"])]
        allscen = sorted(self.S["scenario"].unique())
        self.real = [s for s in allscen if s not in SYN]
        self.eu = [s for s in self.real if s.startswith("inflation_")]
        self.fin = [s for s in self.real if s in FINANCIAL]
        self.macro = [s for s in self.real if s not in self.eu and s not in self.fin]
        self.stats = {}

    def M(self, scen, meth, metric="mean_interval_score"):
        r = self.S[(self.S.scenario == scen) & (self.S.method == meth)]
        return float(r[f"{metric}_mean"].iloc[0]) if len(r) else np.nan

    def _matrix(self, series, metric="mean_interval_score"):
        m = np.array([[self.M(s, me, metric) for me in self.methods] for s in series])
        ok = ~np.isnan(m).any(axis=1)
        return m[ok], [s for s, o in zip(series, ok) if o]

    def _write(self, name, body):
        (self.tab/f"{name}.tex").write_text(body + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- tables
    def table_rank(self):
        mat, series = self._matrix(self.real)
        ranks = np.array([stats.rankdata(r) for r in mat]); avg = ranks.mean(0)
        best = [(ranks[:, j] == 1).sum() for j in range(len(self.methods))]
        cov = np.array([[abs(self.M(s, me, "coverage") - (1-self.alpha))
                         for me in self.methods] for s in series]).mean(0)
        wr = np.array([stats.rankdata([self.M(s, me, "mean_width") for me in self.methods])
                       for s in series]).mean(0)
        order = np.argsort(avg)
        rows = [f"{NICE[self.methods[i]]} & {avg[i]:.2f} & {int(best[i])} & {cov[i]:.3f} & {wr[i]:.2f} \\\\"
                for i in order]
        self._write("rank", "\n".join(rows))
        fr = stats.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
        k, N = len(self.methods), len(series)
        cd = 2.949*np.sqrt(k*(k+1)/(6*N))
        di = self.methods.index("dracp")
        self.stats.update(n_series=N, friedman_p=float(fr.pvalue), nemenyi_cd=float(cd),
                          dracp_rank=float(avg[di]), dracp_best=int(best[di]),
                          dracp_cov_err=float(cov[di]))
        self._rank_data = (avg, order, cd, ranks)
        return avg, order, cd, ranks

    def table_groups(self):
        rows = []
        for name, grp in [("EU HICP inflation", self.eu), ("US macro \\& energy", self.macro),
                          ("Financial returns", self.fin)]:
            mat, series = self._matrix(grp)
            if not len(mat):
                continue
            r = np.array([stats.rankdata(x) for x in mat]); a = r.mean(0)
            w = int((r[:, self.methods.index("dracp")] == 1).sum())
            rows.append(f"{name} & {len(series)} & " +
                        " & ".join(f"{a[self.methods.index(m)]:.2f}" for m in self.methods) +
                        f" & {w}/{len(series)} \\\\")
            key = name.split()[0].lower().strip("\\&")
            self.stats[f"{key}_rank"] = float(a[self.methods.index("dracp")])
            self.stats[f"{key}_best"] = w
            self.stats[f"{key}_n"] = len(series)
        self._write("groups", "\n".join(rows))

    def table_series(self, metric, name):
        def block(grp):
            out = []
            for s in grp:
                vals = [self.M(s, me, metric) for me in self.methods]
                if all(np.isnan(v) for v in vals):
                    continue
                if metric == "coverage":
                    bi = int(np.nanargmin([abs(v-(1-self.alpha)) for v in vals]))
                else:
                    bi = int(np.nanargmin(vals))
                cells = [("\\textbf{%s}" % fmt(v, metric == "coverage")) if j == bi
                         else fmt(v, metric == "coverage") for j, v in enumerate(vals)]
                out.append(f"{label(s)} & " + " & ".join(cells) + " \\\\")
            return "\n".join(out)
        parts = [block(self.eu), block(self.macro), block(self.fin)]
        self._write(name, "\n\\midrule\n".join(p for p in parts if p))

    def table_synthetic(self, metric, name):
        cols = [m for m in ["split_cp", "rolling_cp", "adaptive_cp", "faci", "dracp"]
                if m in self.methods]
        rows = []
        for s in SYN:
            vals = [self.M(s, me, metric) for me in cols]
            if all(np.isnan(v) for v in vals):
                continue
            if metric == "coverage":
                bi = int(np.nanargmin([abs(v-(1-self.alpha)) for v in vals]))
            else:
                bi = int(np.nanargmin(vals))
            cells = [("\\textbf{%s}" % fmt(v, metric == "coverage")) if j == bi
                     else fmt(v, metric == "coverage") for j, v in enumerate(vals)]
            rows.append(f"{SYNL[s]} & " + " & ".join(cells) + " \\\\")
        self._write(name, "\n".join(rows))

    def table_ablation(self):
        variants = [("dracp", "Full DRACP"), ("dracp_no_ratio", "-- density ratio"),
                    ("dracp_no_local", "-- localization"), ("dracp_no_regime", "-- regime weighting"),
                    ("dracp_no_adaptive", "-- adaptive control"), ("dracp_select", "+ selection hedge")]
        rows = []
        for v, lab in variants:
            rel = [100*(self.M(s, v)-self.M(s, "dracp"))/self.M(s, "dracp")
                   for s in SYN if np.isfinite(self.M(s, v))]
            if not rel:
                continue
            cerr = np.nanmean([abs(self.M(s, v, "coverage")-(1-self.alpha)) for s in SYN])
            rows.append(f"{lab} & {np.mean(rel):+.1f} & {cerr:.3f} \\\\")
        self._write("ablation", "\n".join(rows))

    def table_dm(self):
        if self.DM is None:
            return
        bases = [m for m in self.methods if m != "dracp"]
        def stars(p):
            return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""
        def row(s):
            ds = FOLDER.get(s, s)
            sub = self.DM[self.DM.dataset == ds].set_index("base")
            if not len(sub):
                return None
            return f"{label(s)} & " + " & ".join(
                f"{sub.loc[b,'dm_stat']:.1f}{stars(sub.loc[b,'p_value'])}" if b in sub.index else "--"
                for b in bases) + " \\\\"
        allrows = [r for r in (row(s) for s in self.real) if r]
        self._write("dm_full", "\n".join(allrows))
        # restrict to the real series: synthetic aggregates and smoke tests are
        # not part of the per-series real-data evidence
        real_ds = {FOLDER.get(x, x) for x in self.real}
        DMr = self.DM[self.DM.dataset.isin(real_ds)]
        counts = []
        for b in bases:
            sub = DMr[DMr.base == b]
            w = int(((sub.p_value < 0.05) & (sub.dm_stat > 0)).sum())
            l = int(((sub.p_value < 0.05) & (sub.dm_stat < 0)).sum())
            counts.append(f"{NICE[b]} & {len(sub)} & {w} & {l} \\\\")
        self._write("dm_counts", "\n".join(counts))
        self.stats["dm_better"] = int(((DMr.p_value < 0.05) & (DMr.dm_stat > 0)).sum())
        self.stats["dm_worse"] = int(((DMr.p_value < 0.05) & (DMr.dm_stat < 0)).sum())
        self.stats["dm_tests"] = int(len(DMr))

    # --------------------------------------------------------------- figures
    def figure_rank(self):
        avg, order, cd, ranks = self._rank_data
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        oo = order[::-1]
        colors = ["#d62728" if self.methods[i] == "dracp" else "#4c78a8" for i in oo]
        ax.barh([NICE[self.methods[i]] for i in oo], [avg[i] for i in oo], color=colors)
        for pos, i in enumerate(oo):
            ax.text(avg[i]+0.04, pos, f"{avg[i]:.2f}", va="center", fontsize=8)
        ax.axvline(avg[order[0]]+cd, ls=":", color="grey")
        ax.set_xlabel(f"Average interval-score rank (1=best) over {self.stats['n_series']} real series")
        fig.tight_layout(); fig.savefig(self.fig/"rank48.pdf"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        ax.hist(ranks[:, self.methods.index("dracp")],
                bins=np.arange(0.5, len(self.methods)+1.5, 1), color="#d62728", rwidth=.85)
        ax.set_xlabel("DRACP rank on a series"); ax.set_ylabel("# series")
        ax.set_xticks(range(1, len(self.methods)+1))
        fig.tight_layout(); fig.savefig(self.fig/"rank_hist.pdf"); plt.close(fig)

    def figure_groups(self):
        groups = [("EU HICP", self.eu), ("US macro/energy", self.macro), ("Financial", self.fin)]
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        x = np.arange(len(groups)); w = 0.11
        for j, m in enumerate(self.methods):
            vals = []
            for _, g in groups:
                mat, _ = self._matrix(g)
                r = np.array([stats.rankdata(v) for v in mat]).mean(0) if len(mat) else np.full(len(self.methods), np.nan)
                vals.append(r[self.methods.index(m)])
            ax.bar(x+(j-len(self.methods)/2)*w, vals, w, label=NICE[m],
                   color="#d62728" if m == "dracp" else None)
        ax.set_xticks(x); ax.set_xticklabels([n for n, _ in groups])
        ax.set_ylabel("Avg. rank (1=best)"); ax.legend(fontsize=7, ncol=len(self.methods), loc="upper center")
        fig.tight_layout(); fig.savefig(self.fig/"rank_groups.pdf"); plt.close(fig)

    def figure_eu_improvement(self):
        if not self.eu or "faci" not in self.methods:
            return
        imp = [(label(s), 100*(self.M(s, "faci")-self.M(s, "dracp"))/self.M(s, "faci"))
               for s in self.eu if np.isfinite(self.M(s, "dracp"))]
        if not imp:
            return
        imp.sort(key=lambda t: t[1])
        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ax.barh([t[0] for t in imp], [t[1] for t in imp],
                color=["#d62728" if v > 0 else "#888" for _, v in imp])
        ax.axvline(0, color="k", lw=.8)
        ax.set_xlabel("Interval-score improvement of DRACP over FACI (\\%)")
        fig.tight_layout(); fig.savefig(self.fig/"eu_improvement.pdf"); plt.close(fig)

    def figure_case_study(self, countries=("DE", "RO")):
        """Intervals through the 2021-23 inflation surge (needs per-step predictions)."""
        def load(folder, m):
            g = glob.glob(str(self.root/folder/f"*__{m}__predictions.csv"))
            return pd.read_csv(g[0]) if g else None
        avail = [c for c in countries if load(f"inflation_{c}", "dracp") is not None]
        if not avail:
            return
        fig, axes = plt.subplots(2, len(avail), figsize=(4.8*len(avail), 5.0), squeeze=False)
        for col, c in enumerate(avail):
            f = f"inflation_{c}"
            for row, m in enumerate(["dracp", "faci"]):
                d = load(f, m)
                ax = axes[row][col]
                if d is None:
                    continue
                t = pd.to_datetime(d["date"]) if "date" in d else np.arange(len(d))
                y = d["y"].to_numpy()
                colr = "#d62728" if m == "dracp" else "#9467bd"
                ax.fill_between(t, d["lower"], d["upper"], color=colr, alpha=.25)
                ax.plot(t, y, "k-", lw=.9)
                cov = ((y >= d["lower"]) & (y <= d["upper"])).mean()
                sc = interval_score(y, d["lower"], d["upper"], self.alpha).mean()
                ax.set_title(f"{c}: {NICE.get(m,m)} (coverage {cov:.2f}, score {sc:.2f})", fontsize=9)
                if "date" in d:
                    ax.axvspan(pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-01"),
                               color="grey", alpha=.12)
                ax.tick_params(labelsize=7)
        fig.suptitle("Prediction intervals through the 2021--23 inflation surge (shaded)", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(self.fig/"case_inflation.pdf"); plt.close(fig)

    def figure_case_width_alpha(self, country="DE"):
        """Interval width and the self-tuned significance level around the break."""
        def load(folder, m):
            g = glob.glob(str(self.root/folder/f"*__{m}__predictions.csv"))
            return pd.read_csv(g[0]) if g else None
        d = load(f"inflation_{country}", "dracp"); b = load(f"inflation_{country}", "faci")
        if d is None:
            return
        t = pd.to_datetime(d["date"]) if "date" in d else np.arange(len(d))
        fig, ax = plt.subplots(1, 2, figsize=(9.0, 2.8))
        ax[0].plot(t, d["upper"]-d["lower"], color="#d62728", label="DRACP")
        if b is not None:
            ax[0].plot(t, b["upper"]-b["lower"], color="#9467bd", lw=.9, label="FACI")
        ax[0].set_ylabel("Interval width"); ax[0].legend(fontsize=7); ax[0].tick_params(labelsize=7)
        ax[0].set_title(f"Width adapts at the break ({country})", fontsize=9)
        if "alpha_t" in d:
            ax[1].plot(t, d["alpha_t"], color="#2ca02c")
            ax[1].axhline(self.alpha, ls="--", color="k", lw=.8)
            ax[1].set_ylabel(r"self-tuned $\alpha_t$"); ax[1].tick_params(labelsize=7)
            ax[1].set_title(f"Online significance level ({country})", fontsize=9)
        if "date" in d:
            for a_ in ax:
                a_.axvspan(pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-01"),
                           color="grey", alpha=.12)
        fig.tight_layout(); fig.savefig(self.fig/"case_width_alpha.pdf"); plt.close(fig)

    # ------------------------------------------------------------------ run
    def run(self):
        self.table_rank(); self.table_groups(); self.table_ablation(); self.table_dm()
        self.table_series("mean_interval_score", "series_score")
        self.table_series("coverage", "series_coverage")
        self.table_series("mean_width", "series_width")
        self.table_synthetic("coverage", "syn_coverage")
        self.table_synthetic("mean_interval_score", "syn_score")
        self.table_synthetic("mean_width", "syn_width")
        self.figure_rank(); self.figure_groups(); self.figure_eu_improvement()
        self.figure_case_study(); self.figure_case_width_alpha()
        (self.tab.parent/"key_statistics.json").write_text(json.dumps(self.stats, indent=2))
        print(f"tables  -> {self.tab}")
        print(f"figures -> {self.fig}")
        print(json.dumps(self.stats, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="results")
    p.add_argument("--out", default="results/paper")
    p.add_argument("--alpha", type=float, default=0.1)
    a = p.parse_args()
    PaperOutputs(a.results_root, a.out, a.alpha).run()


if __name__ == "__main__":
    main()
