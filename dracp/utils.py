from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def chronological_split(
    df: pd.DataFrame,
    train_fraction: float,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train + calibration fractions must be below 1")
    n = len(df)
    i = int(n * train_fraction)
    j = i + int(n * calibration_fraction)
    return df.iloc[:i].copy(), df.iloc[i:j].copy(), df.iloc[j:].copy()
