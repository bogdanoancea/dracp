from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main(results_dir: str):
    path = Path(results_dir) / "metrics.csv"
    df = pd.read_csv(path)
    cols = [c for c in [
        "scenario", "method", "coverage", "coverage_error",
        "mean_width", "mean_interval_score", "abstention_rate",
        "fallback_rate"
    ] if c in df.columns]
    table = df[cols].sort_values(["scenario", "coverage_error", "mean_width"])
    table.to_csv(Path(results_dir) / "publication_table.csv", index=False)
    try:
        table.to_latex(
            Path(results_dir) / "publication_table.tex",
            index=False, float_format="%.4f"
        )
    except Exception:
        pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()
    main(args.results_dir)
