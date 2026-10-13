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

**macOS / Linux**

```bash
chmod +x run.sh
SEEDS=20 ./run.sh          # all stages
./run.sh setup real        # or selected stages
```

**Windows** (PowerShell or Command Prompt)

```bat
run.bat
run.bat setup real
```

**Any platform**, directly:

```bash
python run.py                    # all stages
python run.py montecarlo report  # selected stages
```

The pipeline is implemented once, in `run.py`; `run.sh` and `run.bat` are thin
wrappers, so behaviour is identical everywhere. Stages are `setup`, `synthetic`,
`real`, `sweeps`, `montecarlo`, `report`. Settings are environment variables
(`SEEDS`, `SKIP_DOWNLOADS`, `NO_ONLINE`, `RUN_TESTS`, `MAX_ELECTRICITY_ROWS`,
`INFLATION_GEO`, `PYTHON_BIN`); on Windows set them with
`set SEEDS=20` (cmd) or `$env:SEEDS=20` (PowerShell).

The script creates its own virtual environment and calls that interpreter
explicitly, so it does not require a `python` on your `PATH` (macOS ships only
`python3`). Include the `setup` stage whenever the environment is missing; a
`.venv` left behind by an interrupted install is detected and rebuilt. Override
the interpreter with `PYTHON_BIN=python3.11 ./run.sh` if needed.

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

If you use this software, please cite the archived release and the paper.

**Software** (DOI: [10.5281/zenodo.21675706](https://doi.org/10.5281/zenodo.21675706)):

```bibtex
@software{oancea_dracp_2026,
  author    = {Oancea, Bogdan},
  title     = {dracp: Dynamic Regime-Aware Conformal Prediction},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.1.0},
  doi       = {10.5281/zenodo.21675706},
  url       = {https://doi.org/10.5281/zenodo.21675706}
}
```

**Paper:**

> Oancea, B. (2026). *Dynamic Regime-Aware Conformal Prediction for Economic
> Forecasting under Multiple Distribution Shifts.*

GitHub also renders a "Cite this repository" button from [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
