#!/usr/bin/env python3
"""Ratio-stability analysis (reviewer request, Q3 mechanism).

For each eval origin and each k in {2,4,8,16,full}, recompute the per-series
own/coarse calibration ratio exactly as run_q3_curve.window_ratio does, then
measure how stable/accurate the k-window estimate is:

  1. per-series CV of the ratio across origins, summarized by k
  2. median |log(ratio_k / ratio_full)| at the SAME origin (estimation error
     relative to the full-window ratio)
  3. effective ratio-window length actually available at each origin/k

No model retraining; reads data/daily.parquet only.
"""

import numpy as np
import pandas as pd

from config import DATA_DIR
from run_daily import rolling_origins
from run_q3_curve import RATIO_WINDOW_DAYS, window_ratio

K_WEEKS = (2, 4, 8, 16, None)


def main():
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    fine = daily[daily["source"] == "own_5min"]
    coarse = daily[daily["source"] == "queue_times"]
    cend = coarse["date"].max()
    clast = coarse[coarse["date"] >= cend - pd.Timedelta(days=RATIO_WINDOW_DAYS)]
    coarse_level = clast.groupby("series_id", observed=True)["y"].mean()

    origins = rolling_origins(daily)
    print(f"{len(origins)} origins ({origins[0].date()} .. {origins[-1].date()})")

    rows = []
    for origin in origins:
        for k in K_WEEKS:
            if k is None:
                fine_win = fine
            else:
                fine_win = fine[fine["date"] >= origin - pd.Timedelta(days=7 * k)]
            own = fine_win[fine_win["date"] < origin]
            eff_days = (origin - own["date"].min()).days if len(own) else 0
            ratio = window_ratio(fine_win, coarse_level, origin)
            for sid, r in ratio.items():
                rows.append({
                    "origin": origin,
                    "k": "full" if k is None else f"k{k:02d}",
                    "series_id": sid,
                    "ratio": r,
                    "eff_window_days": min(eff_days, 7 * k) if k else eff_days,
                })
    df = pd.DataFrame(rows)

    # (3) effective window length by k
    print("\n=== effective fine-window days available (median across origins) ===")
    print(df.groupby("k")["eff_window_days"].median().to_string())

    # (1) per-series CV of ratio across origins, by k
    cv = (df.groupby(["k", "series_id"])["ratio"]
            .agg(["mean", "std", "count"]))
    cv = cv[cv["count"] >= 5]
    cv["cv"] = cv["std"] / cv["mean"]
    print("\n=== per-series CV of ratio across origins (series with >=5 origins) ===")
    print(cv.groupby("k")["cv"].describe(percentiles=[0.25, 0.5, 0.75])
            [["count", "25%", "50%", "75%"]].to_string())

    # (2) same-origin error vs full-window ratio
    full = df[df["k"] == "full"].set_index(["origin", "series_id"])["ratio"]
    sub = df[df["k"] != "full"].copy()
    sub["full_ratio"] = sub.set_index(["origin", "series_id"]).index.map(full)
    sub = sub.dropna(subset=["full_ratio"])
    sub["abs_log_err"] = (np.log(sub["ratio"]) - np.log(sub["full_ratio"])).abs()
    print("\n=== |log(ratio_k / ratio_full)| at the same origin ===")
    print(sub.groupby("k")["abs_log_err"]
             .describe(percentiles=[0.5, 0.75, 0.9])
             [["count", "50%", "75%", "90%"]].to_string())
    # percent of (origin,series) cells within 10% / 20% of full ratio
    for tol, lab in ((np.log(1.1), "10%"), (np.log(1.2), "20%")):
        frac = sub.groupby("k")["abs_log_err"].apply(lambda s: (s <= tol).mean())
        print(f"\nfraction within {lab} of full-window ratio:")
        print(frac.to_string())

    # overall ratio distribution by k (pooled)
    print("\n=== pooled ratio distribution by k ===")
    print(df.groupby("k")["ratio"].describe(percentiles=[0.25, 0.5, 0.75])
            [["count", "25%", "50%", "75%"]].to_string())

    df.to_parquet(DATA_DIR / "ratio_stability.parquet", index=False)
    print(f"\nsaved {len(df):,} rows -> data/ratio_stability.parquet")


if __name__ == "__main__":
    main()
