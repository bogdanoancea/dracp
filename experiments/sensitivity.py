"""Reproducible DRACP hyperparameter-sensitivity sweeps (combined-shift scenario).

Full-scale by default (n=3000, matching configs/synthetic.yaml). Resumable:
appends each result to results/sweeps_all.csv and skips rows already computed,
so it is safe to re-run or interrupt. Override via environment variables:

    SWEEP_N=3000            # samples in the synthetic scenario
    SWEEP_CALWIN=400        # base DRACP calibration_window
    SWEEP_MAXITER=180       # base predictor max_iter
    SWEEP_BUDGET=1e9        # seconds of wall-clock per invocation (default: all)
"""
from __future__ import annotations
import os, time, csv
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from dracp import DRACPConfig, DRACPRegressor
from dracp.utils import chronological_split, set_seed
from dracp.metrics import interval_metrics
from synthetic import generate_scenario
from experiments.common import sequential_evaluate

OUT = Path("results"); OUT.mkdir(parents=True, exist_ok=True)
CSVP = OUT/"sweeps_all.csv"
ALPHA=0.10; SCEN="combined_shift"; SCEN_IDX=6; BASE_SEED=42
N       = int(float(os.environ.get("SWEEP_N", 3000)))
CALWIN  = int(float(os.environ.get("SWEEP_CALWIN", 400)))
MAXITER = int(float(os.environ.get("SWEEP_MAXITER", 180)))
BUDGET  = float(os.environ.get("SWEEP_BUDGET", 1e9))
D = 5
DR_BASE=dict(calibration_window=CALWIN,target_window=80,temporal_decay=0.995,
             bandwidth=2.0,adaptive_bandwidth=True,
             bandwidth_grid=(0.5,1.0,2.0,4.0,8.0),min_ess=30,max_weight=0.25,
             ratio_clip_min=0.05,ratio_clip_max=20.0,n_regimes=3,
             regime_beta=1.0,alpha_step=0.01,fallback="global",
             use_faci_control=True)
FIELDS=["sweep","value","coverage","mean_width","median_width",
        "mean_interval_score","mean_ess","mean_regime_entropy","latency_ms"]

JOBS=[]
for v in [0.5,1.0,2.0,4.0,8.0]: JOBS.append(("bandwidth","bandwidth",v,{"adaptive_bandwidth":False}))
for v in [0.90,0.95,0.97,0.99,1.0]: JOBS.append(("forgetting","temporal_decay",v,{}))
for v in [0.001,0.005,0.01,0.05,0.1]: JOBS.append(("learning_rate","alpha_step",v,{}))
for v in [2.0,5.0,10.0,20.0,1e9]: JOBS.append(("clipping","ratio_clip_max",v,{}))
for v in [100,250,500,1000,2000]: JOBS.append(("calibration_size","calibration_window",v,{}))
for v in [1,2,3,4,5]: JOBS.append(("regimes","n_regimes",v,{}))

def done_keys():
    if not CSVP.exists(): return set()
    d=pd.read_csv(CSVP); return set(zip(d["sweep"], d["value"].astype(float)))

def get_data():
    set_seed(BASE_SEED)
    df=generate_scenario(SCEN,N,D,BASE_SEED+SCEN_IDX)
    fcols=[c for c in df.columns if c.startswith("x")]
    tr,cal,te=chronological_split(df,0.45,0.25)
    return fcols,tr,cal,te

def eval_one(key,value,extra):
    fcols,tr,cal,te=get_data()
    ov={key:value}; ov.update(extra)
    cfg=DRACPConfig(alpha=ALPHA,random_state=BASE_SEED,**{**DR_BASE,**ov})
    bm=HistGradientBoostingRegressor(max_iter=MAXITER,learning_rate=0.05,random_state=BASE_SEED)
    m=DRACPRegressor(cfg,base_model=bm)
    m.fit(tr[fcols].to_numpy(),tr["y"].to_numpy(),cal[fcols].to_numpy(),cal["y"].to_numpy())
    pred=sequential_evaluate(m,te[fcols].to_numpy(),te["y"].to_numpy())
    mt=interval_metrics(pred,ALPHA)
    return dict(coverage=mt["coverage"],mean_width=mt["mean_width"],
                median_width=mt["median_width"],mean_interval_score=mt["mean_interval_score"],
                mean_ess=float(pred["ess"].mean()),
                mean_regime_entropy=float(pred["regime_entropy"].mean()),
                latency_ms=float(pred["latency_ms"].iloc[0]))

def main():
    if not CSVP.exists():
        with open(CSVP,"w",newline="") as f: csv.DictWriter(f,FIELDS).writeheader()
    done=done_keys(); start=time.time(); n=0
    for name,key,value,extra in JOBS:
        if (name,float(value)) in done: continue
        if time.time()-start>BUDGET: break
        r=eval_one(key,value,extra); r["sweep"]=name; r["value"]=value
        with open(CSVP,"a",newline="") as f: csv.DictWriter(f,FIELDS).writerow(r)
        n+=1
        print(f"{name}={value}: cov={r['coverage']:.4f} width={r['mean_width']:.3f} "
              f"score={r['mean_interval_score']:.3f} ess={r['mean_ess']:.1f}",flush=True)
    total=len(JOBS); have=len(done_keys())
    print(f"progress {have}/{total} (+{n} this run) -> {CSVP}",flush=True)

if __name__=="__main__":
    main()
