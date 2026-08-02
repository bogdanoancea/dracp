# dracp 0.1.4 — bootstrap correctness, consolidated pipeline, multi-step horizons



---

## 1. The bootstrap (`dracp/bootstrap.py`, new)

The previous resampler had four defects. All are fixed, and each mattered.

| | before | now |
|---|---|---|
| block length | fixed at 50 | Politis-White (2004) automatic rule, per series, capped at n/4 |
| blocks | non-circular (ends under-sampled) | circular |
| pairing | each method drew its own indices | one common index set per replicate |
| repeated fits | only the last fit used | all fits pooled; fit index drawn per replicate |

On an 84-observation monthly series a block of
50 leaves 34 distinct starts, so replicates overlap almost completely and the
interval comes out far too narrow. Measured on German HICP, the DRACP-FACI
differential CI was `[+0.73, +1.91]` at block 50 but `[-0.83, +2.76]` at block
12 - the fixed choice manufactured significance that the data does not support.

Unpaired marginal intervals cannot be compared for
overlap to test a difference; doing so is conservative and improper. The new
output includes explicit paired columns (`d_*`, `d_*_ci_low/high`, `d_*_p`) so
differences are tested directly. In the smoke run the marginals spanned ~17
units while the paired difference CI spanned ~10 - the pairing recovers real
precision.

Output file `mc_block_bootstrap.csv` gains: `block_length_*`, `n_fits`,
`reference`, and the paired difference columns.

---

## 2. Consolidated experiments

Everything writes to a single `results/`.
Override with `RESULTS_ROOT` if you want it elsewhere.
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

`run.bat` mirrors this on Windows. `INFLATION_GEO` now defaults to `EA20`.

---

## 3. Multi-step horizons

At `h = 1` nothing below is active, by design.

**Diagnosis.** The one-step design weights calibration points by their relevance
*at the forecast origin*. At horizon `h` the quantity that matters is the state
at the **target** date, and origin information decays toward irrelevance as `h`
grows. The regime model made this worst: it was a plain GMM with no transition
structure, so it could only ever describe the present.

**Fixes**, all gated on `horizon > 1`:

1. **Target-aligned regime weighting.** `CausalRegimeEstimator` now estimates a
   first-order transition matrix by soft counts, `P[j,k] ∝ Σ_t π_t[j] π_{t+1}[k]`.
   Calibration rows are weighted by the regime that actually prevailed at their
   target date (`π_{i+h}`, known and causal); the test point uses the forecast
   `π_t P^h`. A useful side effect: as `h` grows `P^h` approaches the stationary
   distribution, so the regime factor switches *itself* off at long horizons
   rather than contributing noise.
2. **Softened temporal decay**, `decay^(1/h)` — overlapping targets carry roughly
   `T/h` independent observations, so one-step memory discards too much.
3. **Inflated bandwidth grid (`×√h`) and ESS floor (`×h`, capped)** — `Y_{t+h}|X_t`
   is more diffuse, so tight localization buys bias without relevance.
4. **Damped controller step (`/√h`)** — overlapping errors autocorrelate the
   coverage indicator and an undamped controller over-reacts to runs of misses.
5. **`dracp_hedge`** — a no-regret selection hedge over {horizon-aware core,
   FACI, conformal PID}. This is the safety net: whatever the weighting achieves,
   cumulative interval score stays within a vanishing term of the best candidate.

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

**Read this with the right caveat.** These are *financial* series — DRACP's
deliberate negative control, where it loses even at h = 1. The mechanism is
clearly working (it never failed to improve on the horizon-unaware version), and
the hedge bounds the downside to a few percent instead of 15–45%. But whether
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
* h = 1 regression against 0.1.x on two independent datasets — interval endpoints
  identical to machine precision.
* Bootstrap validated on AR(1) data with a known 0.3 shift: correct block
  selection, paired CI `[0.298, 0.307]`, p = 0.
* Full pipeline smoke run: synthetic, real Monte-Carlo with bootstrap, horizons
  at h = 1 and 6, extended reporting.
