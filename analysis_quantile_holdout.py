#!/usr/bin/env python3
"""Holdout validation of the interval widening factor (reviewer request).

The x1.2 widening about p50 was originally chosen on the same evaluation
set it was reported on. This script does it properly: split the eval
origin-DAYS chronologically in half, choose the smallest widening factor
that reaches nominal 80% coverage on the FIRST half, then report coverage
with that frozen factor on the SECOND half (overall, by regime, by bucket,
by ride-size bin).

Widening: p10' = p50 - f*(p50-p10), p90' = p50 + f*(p90-p10 ... p50), clip 0.
No retraining; reads data/quantile_gbm_fc.parquet + data/hourly.parquet.
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, HORIZON_BUCKETS
from harness import evaluate

FACTORS = np.round(np.arange(1.00, 1.61, 0.05), 2)
SIZE_BINS = [0, 10, 20, 30, 50, np.inf]
SIZE_LABELS = ["<10m", "10-20m", "20-30m", "30-50m", "50m+"]


def coverage(df: pd.DataFrame, f: float) -> pd.Series:
    lo = np.clip(df["p50"] - f * (df["p50"] - df["p10"]), 0, None)
    hi = df["p50"] + f * (df["p90"] - df["p50"])
    return ((df["y"] >= lo) & (df["y"] <= hi)).astype(float)


def main():
    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    fc = pd.read_parquet(DATA_DIR / "quantile_gbm_fc.parquet")

    sc = evaluate(fc.rename(columns={"p50": "y_hat"}), hourly)
    sc = sc.rename(columns={"y_hat": "p50"})
    sc["bucket"] = pd.Categorical(sc["bucket"], categories=list(HORIZON_BUCKETS),
                                  ordered=True)

    days = sorted(sc["origin"].dt.normalize().unique())
    cut = days[len(days) // 2]
    sc["half"] = np.where(sc["origin"].dt.normalize() < cut, "cal", "test")
    n_cal = (pd.Index(days) < cut).sum()
    print(f"{len(days)} origin-days: {n_cal} calibration "
          f"(< {pd.Timestamp(cut).date()}), {len(days) - n_cal} test")

    cal = sc[sc["half"] == "cal"]
    test = sc[sc["half"] == "test"]
    print(f"rows: cal {len(cal):,}, test {len(test):,}")

    print("\n=== calibration-half coverage by factor ===")
    chosen = None
    for f in FACTORS:
        c = coverage(cal, f).mean()
        star = ""
        if chosen is None and c >= 0.80:
            chosen, star = f, "  <-- chosen (smallest f with coverage >= 0.80)"
        print(f"  f={f:.2f}  coverage={c:.3f}{star}")
    assert chosen is not None, "no factor reached 0.80 on calibration half"

    test = test.copy()
    test["cover"] = coverage(test, chosen)
    print(f"\n=== test-half coverage with frozen f={chosen:.2f} ===")
    print(f"raw (f=1.00) test coverage: {coverage(test, 1.0).mean():.3f}")
    print(f"widened test coverage:      {test['cover'].mean():.3f}  (nominal 0.80)")

    print("\nby regime:")
    print(test.groupby("regime", observed=True)["cover"].agg(["size", "mean"])
              .round(3).to_string())
    print("\nby bucket:")
    print(test.groupby("bucket", observed=True)["cover"].agg(["size", "mean"])
              .round(3).to_string())
    lvl = hourly.groupby("series_id")["y"].mean().rename("mean_wait")
    test = test.join(lvl, on="series_id")
    test["size_bin"] = pd.cut(test["mean_wait"], SIZE_BINS, labels=SIZE_LABELS)
    print("\nby ride-size bin:")
    print(test.groupby("size_bin", observed=True)["cover"].agg(["size", "mean"])
              .round(3).to_string())

    # stability check: per-origin-day coverage spread on the test half
    per_day = test.groupby(test["origin"].dt.normalize())["cover"].mean()
    print(f"\nper-origin-day test coverage: min {per_day.min():.3f}, "
          f"median {per_day.median():.3f}, max {per_day.max():.3f}")


if __name__ == "__main__":
    main()
