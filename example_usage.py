"""End-to-end DRACP usage example (this is the code shown in the appendix)."""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from dracp import ConformalForecaster

# 1. A user's own time series (here: a synthetic monthly series with a break)
rng = np.random.default_rng(0)
n = 400
t = np.arange(n)
level = np.where(t < 300, 0.0, 4.0)                      # structural break
sigma = np.where(t < 300, 1.0, 2.0)                      # volatility change
y = level + 3*np.sin(2*np.pi*t/12) + rng.normal(0, sigma)

# 2. Build autoregressive features (chronological order matters)
df = pd.DataFrame({"y": y})
for lag in (1, 2, 3, 12):
    df[f"lag_{lag}"] = df["y"].shift(lag)
df["month_sin"] = np.sin(2*np.pi*(t % 12)/12)
df["month_cos"] = np.cos(2*np.pi*(t % 12)/12)
df = df.dropna().reset_index(drop=True)
X = df.drop(columns="y"); target = df["y"]

# 3. Chronological train/test split
split = int(0.7*len(df))
X_tr, y_tr = X.iloc[:split], target.iloc[:split]
X_te, y_te = X.iloc[split:], target.iloc[split:]

# 4. Wrap ANY sklearn-style model in DRACP
cf = ConformalForecaster(model=RandomForestRegressor(n_estimators=200, random_state=0),
                         alpha=0.10, calibration_window=150, n_regimes=3)
cf.fit(X_tr, y_tr)

# 5. Sequential prediction intervals (outcomes revealed one step at a time)
lower, upper = cf.predict_interval(X_te, y_te)
print("first 3 intervals:", [(round(l,2), round(u,2)) for l, u in zip(lower[:3], upper[:3])])

# 6. Evaluate
print("metrics:", {k: round(v, 3) for k, v in cf.score(X_te, y_te).items()})

# 7. Tidy output
out = cf.predict_frame(X_te, y_te)
print(out.head(3).round(3).to_string(index=False))
