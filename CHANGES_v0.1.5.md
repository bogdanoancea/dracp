# dracp 0.1.5 - bootstrap correctness, consolidated pipeline, multi-step horizons

Three changes.

---

## 1. The bootstrap (`dracp/bootstrap.py`, new)

The previous resampler had four defects. All are fixed.

| | before | now |
|---|---|---|
| block length | fixed at 50 | Politis-White (2004) automatic rule, per series, capped at n/4 |
| blocks | non-circular (ends under-sampled) | circular |
| pairing | each method drew its own indices | one common index set per replicate |
| repeated fits | only the last fit used | all fits pooled; fit index drawn per replicate |

---

## 2. Consolidated experiments

   Everything writes to a single `results/`.
* New **`reporting/extended_report.py`** summarises horizons, base forecasters
  (pooled *and by group*), the real-panel ablation, and the bootstrap into
  `results/paper/tables/extended_summary.txt`. It is run automatically by the
  `report` stage.

Two reporting conventions are enforced there, both deliberate: aggregation uses
the **mean of per-series percentage changes** (never a ratio of pooled means,
because interval scores are scale-dependent and electricity demand would
dominate), and base-forecaster results are broken out **by group**, because the
pooled number hides a very large HICP-vs-rest difference.

### One entry point, three tiers

```bash
./run.sh --tier quick    # smoke test, minutes — run this first
./run.sh                 # tier 'main': the 48-series panel + horizons + reporting
./run.sh --tier full     # adds hyperparameter sweeps and the 5-forecaster grid
./run.sh montecarlo report   # individual stages still work
```

`run.bat` mirrors this on Windows. `INFLATION_GEO` now defaults to `EA20`, not
`RO` — that default was the source of the euro-area/Romania mix-up.

---

## 3. Multi-step horizons

At `h = 1` nothing below is active, by design.


### Measured on real series (VIX, fed funds, reconstructed from saved runs)

Horizon-aware beat horizon-unaware DRACP in **5 of 5** cells:

| series | h | unaware | aware | improvement |
|---|---|---|---|---|
| fed funds | 6 | 0.0390 | 0.0267 | 31.7% |
| fed funds | 12 | 0.0198 | 0.0191 | 3.4% |
| VIX | 3 | 45.00 | 43.40 | 3.6% |
| VIX | 6 | 43.58 | 42.12 | 3.4% |
| VIX | 12 | 53.54 | 47.53 | 11.2% |

End-to-end through the pipeline at h = 6: `dracp` 44.14 → `dracp_h` 41.88 →
`dracp_hedge` 40.89, with the hedge overtaking FACI (41.43).

**Read this with the right caveat.** These are *financial* series - DRACP's
deliberate negative control, where it loses even at h = 1. The mechanism is
clearly working (it never failed to improve on the horizon-unaware version), and
the hedge bounds the downside to a few percent instead of 15-45%. But whether
horizon-awareness turns h = 3, 6, 12 into a *win* has to be settled on the HICP
panel, whose processed data isn't available here. That is what your run will
tell us, and it is the one result I would not pre-judge.

### New config fields

```python
DRACPConfig(
    horizon=1,                    # forecast horizon; 1 disables everything below
    horizon_adapt=True,           # master switch for the h>1 adjustments
    horizon_target_regime=True,   # align regime weights to the target date
    horizon_max_ess_frac=0.5,     # cap on the inflated ESS floor
)
```

`build_methods(..., horizon=h)` registers `dracp_h` and `dracp_hedge`
automatically when `h > 1`.

---

## Verification performed

* `pytest` - 7 passed.
* h = 1 regression against 0.1.x on two independent datasets - interval endpoints
  identical to machine precision.
* Bootstrap validated on AR(1) data with a known 0.3 shift: correct block
  selection, paired CI `[0.298, 0.307]`, p = 0.
* Full pipeline smoke run: synthetic, real Monte-Carlo with bootstrap, horizons
  at h = 1 and 6, extended reporting.
