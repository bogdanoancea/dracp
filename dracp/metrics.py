from __future__ import annotations
import numpy as np
import pandas as pd

def interval_score(y, lower, upper, alpha=0.1):
    y = np.asarray(y, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    width = upper - lower
    return width + (2/alpha)*(lower-y)*(y < lower) + (2/alpha)*(y-upper)*(y > upper)

def interval_metrics(predictions: pd.DataFrame, alpha: float = 0.1) -> dict:
    valid = predictions.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["y", "lower", "upper"]
    )
    if len(valid) == 0:
        return {
            "n": 0, "coverage": np.nan, "coverage_error": np.nan,
            "mean_width": np.nan, "median_width": np.nan,
            "mean_interval_score": np.nan, "abstention_rate": 1.0,
        }
    covered = (valid["y"] >= valid["lower"]) & (valid["y"] <= valid["upper"])
    width = valid["upper"] - valid["lower"]
    scores = interval_score(valid["y"], valid["lower"], valid["upper"], alpha)
    return {
        "n": int(len(valid)),
        "coverage": float(covered.mean()),
        "coverage_error": float(abs(covered.mean() - (1-alpha))),
        "mean_width": float(width.mean()),
        "median_width": float(width.median()),
        "mean_interval_score": float(np.mean(scores)),
        "abstention_rate": float(predictions.get("abstained", pd.Series(False, index=predictions.index)).mean()),
        "fallback_rate": float(predictions.get("fallback_used", pd.Series(False, index=predictions.index)).mean()),
    }

def rolling_metrics(predictions: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    out = predictions.copy()
    out["covered"] = ((out["y"] >= out["lower"]) & (out["y"] <= out["upper"])).astype(float)
    out["width"] = out["upper"] - out["lower"]
    out["rolling_coverage"] = out["covered"].rolling(window, min_periods=max(5, window//5)).mean()
    out["rolling_width"] = out["width"].rolling(window, min_periods=max(5, window//5)).mean()
    return out
