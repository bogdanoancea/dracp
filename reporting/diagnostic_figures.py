"""Diagnostic, sensitivity and prediction figures for the DRACP paper.

Companion to :mod:`reporting.paper_outputs`, which produces the ranking and
case-study figures. This module produces the remaining figures of the appendix:
weighting diagnostics, hyperparameter sensitivity, cost scaling and example
prediction intervals.

    python -m reporting.diagnostic_figures --results-root results \
        --out results/paper/figures
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out", default="results/paper/figures")
    a = ap.parse_args()
    R = a.results_root
    FIG = a.out
    os.makedirs(FIG, exist_ok=True)
    SW = os.path.join(R, "sweeps_all.csv")
    _run(R, FIG, SW)

def _summary(R):
    """Per-scenario metrics from the Monte-Carlo summary (replaces the former
    analysis.json produced by an earlier, now-removed script)."""
    import pandas as _pd, numpy as _np
    S = _pd.read_csv(os.path.join(R, "mc_all_summary.csv"))
    out = {"per_scenario": {}, "ablation_combined": {}}
    for scen, g in S.groupby("scenario"):
        out["per_scenario"][scen] = {
            r["method"]: {"cov": float(r["coverage_mean"]),
                          "mw": float(r["mean_width_mean"]),
                          "score": float(r["mean_interval_score_mean"])}
            for _, r in g.iterrows()}
    cs = out["per_scenario"].get("combined_shift", {})
    full = cs.get("dracp", {}).get("score")
    for m, v in cs.items():
        if m.startswith("dracp"):
            out["ablation_combined"][m] = {
                "cov": v["cov"], "mw": v["mw"], "score": v["score"],
                "rel": (100*(v["score"]-full)/full) if full else 0.0}
    return out


def _run(R, FIG, SW):
    ANA = _summary(R)
    plt.rcParams.update({"figure.dpi":150,"font.size":11,"axes.grid":True,"grid.alpha":0.3,
                         "savefig.bbox":"tight","axes.spilnes.top" if False else "axes.grid":True})




    NICE={"split_cp":"Split CP","rolling_cp":"Rolling CP","adaptive_cp":"Adaptive CP","dracp":"DRACP"}
    COL={"split_cp":"#888888","rolling_cp":"#1f77b4","adaptive_cp":"#2ca02c","dracp":"#d62728"}
    def L(scen,m,dset="synthetic"): return pd.read_csv(f"{R}/{dset}/{scen}__{m}__predictions.csv")
    def Lr(scen,m,dset="synthetic"): return pd.read_csv(f"{R}/{dset}/{scen}__{m}__rolling.csv")
    def iscore(y,l,u,a=0.1):
        y,l,u=map(lambda z:np.asarray(z,float),(y,l,u)); return (u-l)+(2/a)*(l-y)*(y<l)+(2/a)*(y-u)*(y>u)
    def save(fig,name): fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig); print("wrote",name)
    sw=pd.read_csv(SW)

    # synthetic_time_series : regime_switching y with regime shading
    d=L("regime_switching","dracp"); 
    fig,ax=plt.subplots(figsize=(9,3.2))
    ax.plot(d["y"].values,lw=0.7,color="#333")
    rt=d["regime_true"].values; cols=["#e8f0fe","#fdecea","#eafaf0"]
    start=0
    for i in range(1,len(rt)+1):
        if i==len(rt) or rt[i]!=rt[start]:
            ax.axvspan(start,i,color=cols[int(rt[start])%3],alpha=0.6,lw=0); start=i
    ax.set_xlabel("Time index"); ax.set_ylabel("y"); ax.set_title("Synthetic series (regime switching) with true-regime shading")
    save(fig,"synthetic_time_series")

    # synthetic_predictions : combined_shift dracp interval subset
    d=L("combined_shift","dracp"); s=slice(0,300)
    fig,ax=plt.subplots(figsize=(9,3.4))
    x=np.arange(len(d))[s]
    ax.fill_between(x,d["lower"].values[s],d["upper"].values[s],color=COL["dracp"],alpha=0.25,label="DRACP 90% interval")
    ax.plot(x,d["y"].values[s],".",ms=3,color="#222",label="y")
    ax.plot(x,d["prediction"].values[s],lw=0.8,color=COL["dracp"],label="prediction")
    ax.set_xlabel("Time index"); ax.set_ylabel("y"); ax.legend(loc="upper left",fontsize=9)
    ax.set_title("DRACP predictions and intervals (combined shift)")
    save(fig,"synthetic_predictions")

    # rolling_coverage : combined_shift all methods
    fig,ax=plt.subplots(figsize=(9,3.4))
    for m in ["split_cp","rolling_cp","adaptive_cp","dracp"]:
        r=Lr("combined_shift",m); ax.plot(r["rolling_coverage"].values,label=NICE[m],color=COL[m],lw=1.2)
    ax.axhline(0.9,ls="--",color="k",lw=1); ax.set_ylim(0.5,1.02)
    ax.set_xlabel("Prediction index"); ax.set_ylabel("Rolling coverage (window 100)"); ax.legend(fontsize=9,ncol=2)
    ax.set_title("Rolling coverage under combined shift")
    save(fig,"rolling_coverage")

    # rolling_coverage_abrupt_drift
    fig,ax=plt.subplots(figsize=(9,3.4))
    for m in ["split_cp","rolling_cp","adaptive_cp","dracp"]:
        r=Lr("abrupt_drift",m); ax.plot(r["rolling_coverage"].values,label=NICE[m],color=COL[m],lw=1.2)
    ax.axhline(0.9,ls="--",color="k",lw=1); ax.set_ylim(0.3,1.02)
    ax.set_xlabel("Prediction index"); ax.set_ylabel("Rolling coverage"); ax.legend(fontsize=9,ncol=2)
    ax.set_title("Rolling coverage under abrupt drift")
    save(fig,"rolling_coverage_abrupt_drift")

    # coverage_vs_drift : coverage by scenario (grouped bars)
    SCEN=["stationary","covariate_shift","conditional_drift","abrupt_drift","gradual_drift","regime_switching","combined_shift"]
    lab=[s.replace("_","\n") for s in SCEN]
    fig,ax=plt.subplots(figsize=(10,3.8)); w=0.2; x=np.arange(len(SCEN))
    for k,m in enumerate(["split_cp","rolling_cp","adaptive_cp","dracp"]):
        vals=[ANA["per_scenario"][s][m]["cov"] for s in SCEN]
        ax.bar(x+(k-1.5)*w,vals,w,label=NICE[m],color=COL[m])
    ax.axhline(0.9,ls="--",color="k"); ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=8)
    ax.set_ylabel("Empirical coverage"); ax.set_ylim(0.4,1.0); ax.legend(fontsize=9,ncol=4,loc="lower center")
    ax.set_title("Coverage across scenarios of increasing shift complexity")
    save(fig,"coverage_vs_drift")

    # interval_width_time : combined_shift methods rolling width
    fig,ax=plt.subplots(figsize=(9,3.4))
    for m in ["split_cp","rolling_cp","adaptive_cp","dracp"]:
        r=Lr("combined_shift",m); ax.plot(r["rolling_width"].values,label=NICE[m],color=COL[m],lw=1.2)
    ax.set_xlabel("Prediction index"); ax.set_ylabel("Rolling mean width"); ax.legend(fontsize=9,ncol=2)
    ax.set_title("Interval width over time (combined shift)")
    save(fig,"interval_width_time")

    # interval_width_regime_switching with regime shading (dracp)
    d=Lr("regime_switching","dracp")
    fig,ax=plt.subplots(figsize=(9,3.4))
    rt=d["regime_true"].values; cols=["#e8f0fe","#fdecea","#eafaf0"]; start=0
    for i in range(1,len(rt)+1):
        if i==len(rt) or rt[i]!=rt[start]:
            ax.axvspan(start,i,color=cols[int(rt[start])%3],alpha=0.6,lw=0); start=i
    ax.plot(d["width"].values,color=COL["dracp"],lw=0.8)
    ax.set_xlabel("Prediction index"); ax.set_ylabel("Interval width")
    ax.set_title("DRACP interval width under regime switching (shading = true regime)")
    save(fig,"interval_width_regime_switching")

    # width_distribution : boxplot of widths by method combined_shift
    fig,ax=plt.subplots(figsize=(7,3.6))
    data=[(L("combined_shift",m)["upper"]-L("combined_shift",m)["lower"]).values for m in ["split_cp","rolling_cp","adaptive_cp","dracp"]]
    bp=ax.boxplot(data,showfliers=False,patch_artist=True)
    ax.set_xticks(range(1,len(data)+1))
    ax.set_xticklabels([NICE[m] for m in ["split_cp","rolling_cp","adaptive_cp","dracp"]])
    for patch,m in zip(bp["boxes"],["split_cp","rolling_cp","adaptive_cp","dracp"]): patch.set_facecolor(COL[m]); patch.set_alpha(0.6)
    ax.set_ylabel("Interval width"); ax.set_title("Interval-width distribution (combined shift)")
    save(fig,"width_distribution")

    # density_ratio_hist : histogram of estimated ratio proxy (covariate_shift dracp)
    d=L("covariate_shift","dracp"); rr=d["b_ratio_proxy"].replace([np.inf,-np.inf],np.nan).dropna()
    fig,ax=plt.subplots(figsize=(6.5,3.4)); ax.hist(rr,bins=40,color=COL["dracp"],alpha=0.7)
    ax.set_xlabel("Density-ratio perturbation proxy $b_{ratio}$"); ax.set_ylabel("Count")
    ax.set_title("Distribution of density-ratio contribution (covariate shift)")
    save(fig,"density_ratio_hist")

    # density_ratio_scatter / density_ratio_estimation : proxy vs x1 (covariate shift)
    d=L("covariate_shift","dracp")
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    ax.scatter(np.arange(len(d)),d["b_ratio_proxy"].values,s=6,alpha=0.4,color=COL["dracp"])
    ax.set_xlabel("Time index"); ax.set_ylabel("$b_{ratio}$ proxy")
    ax.set_title("Density-ratio contribution over the covariate-shift stream")
    save(fig,"density_ratio_scatter")
    fig,ax=plt.subplots(figsize=(6.5,3.6))
    ax.scatter(d["b_kernel_proxy"].values,d["b_ratio_proxy"].values,s=6,alpha=0.4,color=COL["adaptive_cp"])
    ax.set_xlabel("Kernel/local proxy"); ax.set_ylabel("Density-ratio proxy")
    ax.set_title("Estimated density-ratio vs local-kernel contributions (covariate shift)")
    save(fig,"density_ratio_estimation")

    # local_weights : local_weight_mean over time (combined_shift dracp)
    d=L("combined_shift","dracp")
    fig,ax=plt.subplots(figsize=(7,3.2)); ax.plot(d["local_weight_mean"].values,color=COL["adaptive_cp"],lw=0.8)
    ax.set_xlabel("Time index"); ax.set_ylabel("Mean localized weight")
    ax.set_title("Localized calibration weight over time (combined shift)")
    save(fig,"local_weights")

    # effective_sample_size : ess vs bandwidth (sweep)
    b=sw[sw.sweep=="bandwidth"].sort_values("value")
    fig,ax=plt.subplots(figsize=(6.5,3.4)); ax.plot(b["value"],b["mean_ess"],"o-",color=COL["dracp"])
    ax.set_xscale("log"); ax.set_xlabel("Localization bandwidth $h$"); ax.set_ylabel("Mean effective sample size")
    ax.set_title("Effective calibration sample size vs bandwidth")
    save(fig,"effective_sample_size")

    # regime_probabilities : predicted regime + true regime over time (regime_switching)
    d=L("regime_switching","dracp")
    fig,ax=plt.subplots(figsize=(9,3.0))
    ax.plot(d["regime_true"].values,label="true regime",lw=1.0,color="#333")
    ax.plot(d["regime"].values,label="predicted regime",lw=0.8,alpha=0.7,color=COL["dracp"])
    ax.set_yticks([0,1,2]); ax.set_xlabel("Time index"); ax.set_ylabel("Regime"); ax.legend(fontsize=9)
    ax.set_title("Estimated vs true latent regime (regime switching)")
    save(fig,"regime_probabilities")

    # regime_transition_matrix : estimated from predicted regime sequence
    d=L("regime_switching","dracp"); seq=d["regime"].values.astype(int); K=3
    T=np.zeros((K,K))
    for a,bb in zip(seq[:-1],seq[1:]): T[a,bb]+=1
    T=T/np.clip(T.sum(1,keepdims=True),1,None)
    fig,ax=plt.subplots(figsize=(4.6,4)); im=ax.imshow(T,cmap="Blues",vmin=0,vmax=1)
    for i in range(K):
        for j in range(K): ax.text(j,i,f"{T[i,j]:.2f}",ha="center",va="center",color="black")
    ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xlabel("To regime"); ax.set_ylabel("From regime")
    ax.set_title("Estimated regime transition matrix"); fig.colorbar(im,fraction=0.046)
    save(fig,"regime_transition_matrix")

    # residual_histogram : dracp errors combined_shift (error col = |y-pred|? check) use y-prediction
    d=L("combined_shift","dracp"); res=(d["y"]-d["prediction"]).values
    fig,ax=plt.subplots(figsize=(6.5,3.4)); ax.hist(res,bins=40,color=COL["dracp"],alpha=0.7,density=True)
    xs=np.linspace(res.min(),res.max(),200); import math
    mu,sd=res.mean(),res.std(); ax.plot(xs,np.exp(-(xs-mu)**2/(2*sd**2))/(sd*np.sqrt(2*np.pi)),"k--",lw=1)
    ax.set_xlabel("Residual $y-\\hat y$"); ax.set_ylabel("Density"); ax.set_title("Residual distribution (DRACP, combined shift)")
    save(fig,"residual_histogram")

    # qq_plot : normalized residuals
    from scipy import stats
    d=L("combined_shift","dracp"); res=((d["y"]-d["prediction"])/d["scale"].replace(0,np.nan)).dropna().values
    fig,ax=plt.subplots(figsize=(5,4.2)); stats.probplot(res,dist="norm",plot=ax)
    ax.set_title("Normal Q--Q plot of standardized residuals (DRACP)")
    save(fig,"qq_plot")

    # adaptive_alpha : alpha_t over time (combined_shift dracp)
    d=L("combined_shift","dracp")
    fig,ax=plt.subplots(figsize=(8,3.0)); ax.plot(d["alpha_t"].values,color=COL["dracp"],lw=0.9)
    ax.axhline(0.1,ls="--",color="k",lw=1); ax.set_xlabel("Time index"); ax.set_ylabel(r"Adaptive $\alpha_t$")
    ax.set_title("Adaptive significance level over time (combined shift)")
    save(fig,"adaptive_alpha")

    # ablation_barplot : interval score by variant (combined_shift)
    abl=ANA["ablation_combined"]; order=["dracp","dracp_no_ratio","dracp_no_local","dracp_no_regime","dracp_no_adaptive"]
    names={"dracp":"Full","dracp_no_ratio":"No ratio","dracp_no_local":"No local","dracp_no_regime":"No regime","dracp_no_adaptive":"No adaptive"}
    fig,ax=plt.subplots(figsize=(7,3.4))
    vals=[abl[m]["score"] for m in order]; cov=[abl[m]["cov"] for m in order]
    bars=ax.bar([names[m] for m in order],vals,color="#4c78a8",alpha=0.85)
    ax.set_ylabel("Mean interval score"); ax.set_title("Ablation: interval score under combined shift")
    for b_,c in zip(bars,cov): ax.text(b_.get_x()+b_.get_width()/2,b_.get_height()+0.05,f"cov {c:.3f}",ha="center",fontsize=8)
    save(fig,"ablation_barplot")

    # bandwidth_sensitivity
    b=sw[sw.sweep=="bandwidth"].sort_values("value")
    fig,ax=plt.subplots(figsize=(6.5,3.4))
    ax.plot(b["value"],b["coverage"],"o-",label="coverage",color=COL["dracp"])
    ax.axhline(0.9,ls="--",color="k",lw=1); ax.set_xscale("log"); ax.set_xlabel("Bandwidth $h$"); ax.set_ylabel("Coverage")
    ax2=ax.twinx(); ax2.plot(b["value"],b["mean_width"],"s--",color=COL["adaptive_cp"],label="width"); ax2.set_ylabel("Mean width")
    ax.set_title("Sensitivity to localization bandwidth (combined shift)")
    save(fig,"bandwidth_sensitivity")

    # clipping_sensitivity
    c=sw[sw.sweep=="clipping"].sort_values("value")
    fig,ax=plt.subplots(figsize=(6.5,3.4))
    xl=[str(int(v)) if v<1e8 else "none" for v in c["value"]]
    ax.plot(xl,c["mean_interval_score"],"o-",color=COL["dracp"])
    ax.set_xlabel("Upper clip $r_{max}$"); ax.set_ylabel("Mean interval score")
    ax.set_title("Sensitivity to density-ratio clipping (combined shift)")
    save(fig,"clipping_sensitivity")

    # forgetting_factor
    f=sw[sw.sweep=="forgetting"].sort_values("value")
    fig,ax=plt.subplots(figsize=(6.5,3.4))
    ax.plot(f["value"],f["coverage"],"o-",color=COL["dracp"],label="coverage"); ax.axhline(0.9,ls="--",color="k")
    ax2=ax.twinx(); ax2.plot(f["value"],f["mean_interval_score"],"s--",color=COL["adaptive_cp"]); ax2.set_ylabel("Interval score")
    ax.set_xlabel(r"Forgetting factor $\lambda_{time}$"); ax.set_ylabel("Coverage")
    ax.set_title("Sensitivity to temporal forgetting (combined shift)")
    save(fig,"forgetting_factor")

    # runtime_scaling : latency vs calibration buffer (sweep)
    cs=sw[sw.sweep=="calibration_size"].sort_values("value")
    fig,ax=plt.subplots(figsize=(6.5,3.4)); ax.plot(cs["value"],cs["latency_ms"],"o-",color=COL["dracp"])
    ax.set_xlabel("Calibration buffer size"); ax.set_ylabel("Latency (ms/prediction)")
    ax.set_title("DRACP runtime vs calibration-buffer size")
    save(fig,"runtime_scaling")

    # memory_scaling : analytic O(Bd) illustration using buffer sizes
    B=np.array([100,250,500,1000,2000,4000]); d_dim=5
    fig,ax=plt.subplots(figsize=(6.5,3.4)); ax.plot(B,B*d_dim*8/1024,"o-",color=COL["dracp"])
    ax.set_xlabel("Calibration buffer size $B$"); ax.set_ylabel("Calibration memory (KB)")
    ax.set_title(r"Memory footprint scales as $\mathcal{O}(Bd)$")
    save(fig,"memory_scaling")

    # electricity_predictions
    d=L("electricity_demand","dracp","electricity"); s=slice(0,400)
    fig,ax=plt.subplots(figsize=(9,3.4)); x=np.arange(len(d))[s]
    ax.fill_between(x,d["lower"].values[s],d["upper"].values[s],color=COL["dracp"],alpha=0.25,label="DRACP interval")
    ax.plot(x,d["y"].values[s],lw=0.7,color="#222",label="load")
    ax.set_xlabel("Hourly index"); ax.set_ylabel("Electricity load"); ax.legend(fontsize=9)
    ax.set_title("DRACP intervals on electricity demand (first 400 test hours)")
    save(fig,"electricity_predictions")

    # inflation_predictions
    d=L("european_inflation","dracp","inflation")
    fig,ax=plt.subplots(figsize=(8,3.4)); x=np.arange(len(d))
    ax.fill_between(x,d["lower"].values,d["upper"].values,color=COL["dracp"],alpha=0.25,label="DRACP interval")
    ax.plot(x,d["y"].values,"o-",ms=3,lw=0.8,color="#222",label="inflation")
    ax.set_xlabel("Month index"); ax.set_ylabel("Inflation"); ax.legend(fontsize=9)
    ax.set_title("DRACP intervals on European inflation")
    save(fig,"inflation_predictions")

    print("ALL FIGURES DONE")


if __name__ == "__main__":
    main()
