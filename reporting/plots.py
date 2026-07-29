from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def make_all_plots(result_dir, metrics_df: pd.DataFrame):
    result_dir = Path(result_dir)
    figdir = result_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    if not metrics_df.empty:
        pivot = metrics_df.pivot(index="scenario", columns="method", values="coverage")
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("Empirical coverage")
        ax.axhline(0.9, linestyle="--")
        ax.figure.tight_layout()
        ax.figure.savefig(figdir / "coverage_by_scenario.png", dpi=200)
        plt.close(ax.figure)

        pivot = metrics_df.pivot(index="scenario", columns="method", values="mean_width")
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_ylabel("Mean interval width")
        ax.figure.tight_layout()
        ax.figure.savefig(figdir / "width_by_scenario.png", dpi=200)
        plt.close(ax.figure)

    for path in result_dir.glob("*__rolling.csv"):
        df = pd.read_csv(path)
        if "rolling_coverage" not in df:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["rolling_coverage"])
        ax.axhline(0.9, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rolling coverage")
        ax.set_xlabel("Prediction index")
        fig.tight_layout()
        fig.savefig(figdir / f"{path.stem}_coverage.png", dpi=200)
        plt.close(fig)
