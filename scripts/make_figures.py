#!/usr/bin/env python3
"""Create simple paper figures from saved BayPIRE result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    df = pd.read_csv(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in ["prompt_mae", "test_log_loss", "test_brier"]:
        if metric not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, group in df.groupby("method"):
            means = group.groupby("budget")[metric].mean()
            ax.plot(means.index, means.values, marker="o", label=method)
        ax.set_xlabel("Observed fraction")
        ax.set_ylabel(metric)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
