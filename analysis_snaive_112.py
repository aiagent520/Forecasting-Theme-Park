#!/usr/bin/env python3
"""Extended seasonal-naive baseline over the full 112d horizon (reviewer request).

run_daily's snaive7 only looks back 1-4 weeks, so it stops producing
forecasts beyond the 15-28d bucket. The standard seasonal-naive definition
extends to any lead: y_hat(t) = y(t - 7m) with the smallest m >= 1 such that
t - 7m < origin (i.e., repeat the last observed week), with up to 4 extra
weeks of fallback for missing source days.

Same protocol/origins/scales as run_daily; reports by bucket and
regime x bucket so the table rows are directly comparable.

Usage: python3 analysis_snaive_112.py | tee results_snaive112.txt
"""

import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, MAX_HORIZON_DAYS
from run_daily import evaluate, mase_scales, rolling_origins, summarize


def forecast_snaive7_ext(train, grid, origin):
    fc = grid.copy()
    lookup = train.set_index(["series_id", "date"])["y"]
    lead = (fc["date"] - origin).dt.days + 1          # 1-based lead
    m0 = np.ceil(lead / 7).astype(int)                # smallest m: t-7m < origin
    y_hat = pd.Series(index=fc.index, dtype=float)
    for extra in range(5):                            # fallback extra weeks
        need = y_hat.isna()
        if not need.any():
            break
        src = fc.loc[need, "date"] - pd.to_timedelta(7 * (m0[need] + extra), unit="D")
        keys = list(zip(fc.loc[need, "series_id"], src))
        y_hat.loc[need] = lookup.reindex(keys).to_numpy()
    fc["y_hat"] = y_hat
    return fc.dropna(subset=["y_hat"])


def main():
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    actuals = daily[daily["source"] == "own_5min"]
    origins = rolling_origins(daily)
    print(f"{len(origins)} origins ({origins[0].date()} .. {origins[-1].date()}), "
          f"horizon {MAX_HORIZON_DAYS}d")

    scored, scale_frames = [], []
    for i, origin in enumerate(origins):
        train = daily[daily["date"] < origin]
        horizon_end = origin + pd.Timedelta(days=MAX_HORIZON_DAYS)
        test = actuals[(actuals["date"] >= origin) & (actuals["date"] < horizon_end)]
        grid = test[["series_id", "date"]].copy()
        grid["origin"] = origin
        scales = mase_scales(train)
        scales.index.name = "series_id"
        scale_frames.append(scales.rename("scale").to_frame().assign(origin=origin))
        fc = forecast_snaive7_ext(train, grid, origin)
        scored.append(evaluate(fc, actuals))
        print(f"  origin {i + 1}/{len(origins)} {origin.date()} done", file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    sc = pd.concat(scored, ignore_index=True)
    print("\n=== snaive7_ext — by lead bucket ===")
    print(summarize(sc, scales, by=("bucket",)).to_string())
    print("\n=== snaive7_ext — by regime x lead bucket ===")
    print(summarize(sc, scales, by=("regime", "bucket")).to_string())

    sc[["series_id", "origin", "date", "y_hat"]].to_parquet(
        DATA_DIR / "snaive112_fc.parquet", index=False)
    print(f"\nsaved {len(sc):,} forecast rows -> data/snaive112_fc.parquet")


if __name__ == "__main__":
    main()
