import numpy as np
from dracp.quantiles import weighted_quantile_with_test_mass
from dracp.diagnostics import effective_sample_size
from dracp import DRACPConfig, DRACPRegressor

def test_quantile_finite():
    q = weighted_quantile_with_test_mass(
        np.array([1.,2.,3.,4.]), np.ones(4), alpha=0.3, test_weight=1.0
    )
    assert np.isfinite(q)

def test_ess():
    assert np.isclose(effective_sample_size(np.ones(10)), 10)

def test_dracp_smoke():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(250,3))
    y = x[:,0] + rng.normal(size=250)
    model = DRACPRegressor(DRACPConfig(
        calibration_window=80, target_window=20, min_ess=10,
        n_regimes=2, bandwidth_grid=(1.0,2.0,4.0)
    ))
    model.fit(x[:120], y[:120], x[120:200], y[120:200])
    result = model.predict_one(x[200])
    assert "lower" in result and "upper" in result
