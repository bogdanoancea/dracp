# dracp — Dynamic Regime-Aware Conformal Prediction

[![tests](https://github.com/bogdanoancea/dracp/actions/workflows/tests.yml/badge.svg)](https://github.com/bogdanoancea/dracp/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21675706.svg)](https://doi.org/10.5281/zenodo.21675706)
[![PyPI](https://img.shields.io/pypi/v/dracp.svg)](https://pypi.org/project/dracp/)

Distribution-free prediction intervals for forecasting when several distribution
shifts act **at the same time** — covariate shift, concept drift, local
heterogeneity and latent regimes.

Classical conformal prediction guarantees finite-sample coverage under
exchangeability, which sequential economic data routinely violate. Existing
adaptive extensions each correct a single departure. **DRACP** composes four
mechanisms inside one weighted-conformal calibration step:

- **density-ratio weighting** — corrects a shifted covariate distribution;
- **localized kernel weighting** — restores approximate conditional validity,
  with an automatic floor on the effective sample size;
- **regime-aware weighting** — up-weights calibration points from the same
  latent regime as the query;
- **self-tuning online control** — adapts the significance level from realised
  coverage, aggregating adaptive-conformal experts over several learning rates.

It wraps **any** point-prediction model: conformal validity does not depend on
the model, so keep whatever forecaster you already trust.

## Install

```bash
pip install dracp
```

or from source:

```bash
git clone https://github.com/bogdanoancea/dracp.git
cd dracp
pip install -e .
```

Requires Python ≥ 3.10 and the standard scientific stack (NumPy, pandas, SciPy,
scikit-learn). Reproducing the paper's experiments additionally needs
`pip install -e ".[experiments]"`.

## Quick start

```python
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from dracp import ConformalForecaster

# X, y in chronological order
cf = ConformalForecaster(
        model=RandomForestRegressor(n_estimators=200, random_state=0),
        alpha=0.10)                       # nominal coverage 90%
cf.fit(X_train, y_train)

# Sequential intervals: outcomes are revealed one step at a time,
# so calibration adapts online (recommended for forecasting).
lower, upper = cf.predict_interval(X_test, y_test)

print(cf.score(X_test, y_test))           # coverage, mean width, interval score
cf.predict_frame(X_test, y_test).head()   # tidy DataFrame
```

Omit `y` in `predict_interval` for fixed (non-adaptive) calibration.
A complete runnable script is in [`example_usage.py`](example_usage.py).

### Useful options

Any configuration field can be passed as a keyword argument:

```python
ConformalForecaster(model=..., alpha=0.1,
                    calibration_window=500,   # rolling calibration buffer
                    n_regimes=3,              # latent regimes
                    bandwidth=2.0,            # localization kernel
                    use_faci_control=True,    # self-tuning controller (default)
                    asymmetric=True)          # two-sided quantiles for skewed errors
```

## Baselines included

For benchmarking, the package also implements split, rolling and adaptive (ACI)
conformal prediction, and the recent **FACI**, **SAOCP** and **conformal PID**
online procedures, all behind the same interface.

## Reproducing the paper

The repository ships the full experimental pipeline. Raw data and results are not
version-controlled; a single command downloads everything and regenerates every
table and figure:

```bash
chmod +x run.sh
SEEDS=20 ./run.sh          # setup, synthetic, real, sweeps, montecarlo, report
./run.sh setup real        # or run individual stages
```

The `report` stage writes the paper's tables and figures to `results/paper/`:

```bash
python -m reporting.paper_outputs --results-root results --out results/paper
```

Hyperparameter-sensitivity sweeps (the sensitivity appendix) are produced by

```bash
python -m experiments.sensitivity        # resumable; writes results/sweeps_all.csv
```

The paper-output step regenerates every table body (`results/paper/tables/*.tex`, ready to `\input`)
and figure (`results/paper/figures/*.pdf`) reported in the paper, together with
`key_statistics.json` containing the headline numbers (average ranks, win counts,
Friedman/Nemenyi statistics, Diebold--Mariano tallies).

Data sources are public: Eurostat (EU-27 HICP inflation), FRED (US
macroeconomic and financial series) and UCI (electricity demand).

## Citing

If you use this software, please cite both the software (see
[`CITATION.cff`](CITATION.cff), or the archived release DOI) and the paper:

> Oancea, B. (2026). *Dynamic Regime-Aware Conformal Prediction for Economic
> Forecasting under Multiple Distribution Shifts.*

## License

MIT — see [LICENSE](LICENSE).
