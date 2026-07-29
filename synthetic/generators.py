from __future__ import annotations
import numpy as np
import pandas as pd

def generate_scenario(name: str, n: int = 3000, d: int = 5,
                      seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = rng.normal(size=(n, d))
    regime = np.zeros(n, dtype=int)
    sigma = np.ones(n)
    beta = np.linspace(0.5, 1.5, d)
    intercept = np.zeros(n)

    if name == "stationary":
        pass
    elif name == "covariate_shift":
        shift = np.clip((t - n*0.55)/(n*0.25), 0, 1)
        x[:, 0] += 2.0 * shift
        x[:, 1] -= 1.2 * shift
    elif name == "conditional_drift":
        drift = np.clip((t - n*0.5)/(n*0.35), 0, 1)
        intercept = 3.0 * drift
        x[:, 0] *= 1.0 + drift
    elif name == "abrupt_drift":
        after = t >= int(n*0.65)
        intercept[after] = 4.0
        sigma[after] = 1.8
    elif name == "gradual_drift":
        phase = t / max(n-1, 1)
        intercept = 2*np.sin(2*np.pi*phase)
        x[:, 0] *= 0.5 + 1.5*phase
        sigma = 0.7 + phase
    elif name == "regime_switching":
        transition = np.array([[0.97,0.02,0.01],[0.03,0.94,0.03],[0.02,0.04,0.94]])
        for i in range(1, n):
            regime[i] = rng.choice(3, p=transition[regime[i-1]])
        intercept = np.choose(regime, [-2.0, 0.5, 3.0])
        sigma = np.choose(regime, [0.6, 1.2, 2.0])
        x[:,0] += np.choose(regime, [-1.0,0.0,1.5])
    elif name == "combined_shift":
        transition = np.array([[0.96,0.03,0.01],[0.03,0.94,0.03],[0.02,0.05,0.93]])
        for i in range(1,n):
            regime[i] = rng.choice(3,p=transition[regime[i-1]])
        phase=t/max(n-1,1)
        x[:,0] += 2.2*phase + np.choose(regime,[-0.8,0,1.0])
        intercept = 2.5*phase + np.choose(regime,[-1.5,0.2,2.5])
        sigma = 0.7 + 0.9*phase + np.choose(regime,[0,0.25,0.7])
    else:
        raise ValueError(f"Unknown scenario: {name}")

    nonlinear = 0.5*np.sin(x[:,0]) + 0.25*x[:,1]**2
    y = intercept + x @ beta + nonlinear + rng.normal(scale=sigma)
    df = pd.DataFrame(x, columns=[f"x{j+1}" for j in range(d)])
    df["time"] = t
    df["regime_true"] = regime
    df["y"] = y
    return df
