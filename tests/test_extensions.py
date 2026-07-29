import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from dracp import (DRACPConfig, DRACPRegressor, FACIRegressor, SAOCPRegressor,
                   ConformalPIDRegressor)
from dracp.metrics import interval_metrics

def _data(n=400, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ np.linspace(1, 2, d) + rng.normal(scale=1.0, size=n)
    tr, cal, te = slice(0, n//2), slice(n//2, 3*n//4), slice(3*n//4, n)
    return X, y, tr, cal, te

def _run(method, X, y, tr, cal, te):
    method.fit(X[tr], y[tr], X[cal], y[cal])
    rows = []
    for i in range(te.start, te.stop):
        r = method.predict_one(X[i])
        method.update_one(X[i], float(y[i]), r)
        r.update({"y": float(y[i])}); rows.append(r)
    import pandas as pd
    return interval_metrics(pd.DataFrame(rows), 0.1)

def test_online_baselines_run_and_cover():
    X, y, tr, cal, te = _data()
    for cls in (FACIRegressor, SAOCPRegressor, ConformalPIDRegressor):
        m = cls(alpha=0.1, window=150,
                model=HistGradientBoostingRegressor(max_iter=60, random_state=0))
        met = _run(m, X, y, tr, cal, te)
        assert 0.0 <= met["coverage"] <= 1.0
        assert np.isfinite(met["mean_width"])
        assert met["coverage"] > 0.6  # broadly calibrated on iid data

def test_dracp_asymmetric_valid_intervals():
    X, y, tr, cal, te = _data()
    cfg = DRACPConfig(alpha=0.1, calibration_window=150, n_regimes=2,
                      bandwidth_grid=(1.0, 2.0), asymmetric=True, random_state=0)
    m = DRACPRegressor(cfg, base_model=HistGradientBoostingRegressor(max_iter=60, random_state=0))
    met = _run(m, X, y, tr, cal, te)
    assert np.isfinite(met["mean_width"]) and met["mean_width"] > 0
    assert met["coverage"] > 0.6

def test_pid_controller_reduces_to_aci():
    from dracp.controller import PIDAlphaController, AdaptiveAlphaController
    pid = PIDAlphaController(0.1, 0.05, ki=0.0, kd=0.0)
    aci = AdaptiveAlphaController(0.1, 0.05)
    for e in [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]:
        pid.update(e); aci.update(e)
    assert abs(pid.used_alpha - aci.used_alpha) < 1e-9

def test_faci_controller_valid_and_dracp_faci_runs():
    import numpy as np
    from dracp.controller import FACIController
    c = FACIController(0.1)
    for e in [1,0,0,1,0,0,0,1,0,0,0,0]:
        a = c.update(e)
        assert 1e-6 <= a <= 1-1e-6
    from sklearn.ensemble import HistGradientBoostingRegressor
    from dracp import DRACPConfig, DRACPRegressor
    X, y, tr, cal, te = _data()
    cfg = DRACPConfig(alpha=0.1, calibration_window=150, n_regimes=2,
                      bandwidth_grid=(1.0,2.0), use_faci_control=True, random_state=0)
    m = DRACPRegressor(cfg, base_model=HistGradientBoostingRegressor(max_iter=60, random_state=0))
    met = _run(m, X, y, tr, cal, te)
    assert np.isfinite(met["mean_width"]) and met["coverage"] > 0.6
