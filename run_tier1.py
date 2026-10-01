#!/usr/bin/env python3
"""Tier 1 baselines on the hourly benchmark (smoke test of the harness).

Baselines implemented here (pure pandas; statsforecast ETS/Theta come later):
  snaive       — same hour, same day-of-week, most recent prior week
  hourly_avg   — historical mean by (series, day-of-week, hour) over train
                 window (the production fallback, ride_hourly_avg_v2 style)

Usage: python3 run_tier1.py [--origins N]   (N = limit origins, for quick runs)
"""

import argparse
import sys

import pandas as pd

from config import DATA_DIR, MAX_HORIZON_HOURS, SEASONAL_PERIOD_HOURS
from harness import (
    evaluate,
    mase_scales,
    rolling_origins,
    summarize,
    train_test_split,
)


def target_grid(test: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """The (series_id, ts) pairs to forecast = hours that actually occurred."""
    grid = test[["series_id", "ts"]].copy()
    grid["origin"] = origin
    return grid


def forecast_snaive(train, grid, origin):
    """Same hour one week earlier; fall back to 2/3/4 weeks if missing."""
    fc = grid.copy()
    lookup = train.set_index(["series_id", "ts"])["y"]
    y_hat = pd.Series(index=fc.index, dtype=float)
    for weeks_back in (1, 2, 3, 4):
        need = y_hat.isna()
        if not need.any():
            break
        src_ts = fc.loc[need, "ts"] - pd.Timedelta(hours=SEASONAL_PERIOD_HOURS * weeks_back)
        keys = list(zip(fc.loc[need, "series_id"], src_ts))
        vals = lookup.reindex(keys).to_numpy()
        y_hat.loc[need] = vals
    fc["y_hat"] = y_hat
    return fc.dropna(subset=["y_hat"])


def forecast_hourly_avg(train, grid, origin):
    """Mean by (series, dow, hour) over the training window."""
    tr = train.copy()
    tr["dow"] = tr["ts"].dt.dayofweek
    tr["hour"] = tr["ts"].dt.hour
    avg = tr.groupby(["series_id", "dow", "hour"])["y"].mean().rename("y_hat")
    fc = grid.copy()
    fc["dow"] = fc["ts"].dt.dayofweek
    fc["hour"] = fc["ts"].dt.hour
    fc = fc.join(avg, on=["series_id", "dow", "hour"])
    return fc.drop(columns=["dow", "hour"]).dropna(subset=["y_hat"])


BASELINES = {
    "snaive": forecast_snaive,
    "hourly_avg": forecast_hourly_avg,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=None)
    args = ap.parse_args()

    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    origins = rolling_origins(hourly)
    if args.origins:
        origins = origins[-args.origins :]
    print(
        f"Hourly data: {len(hourly):,} rows, {hourly['series_id'].nunique()} series | "
        f"{len(origins)} origins ({origins[0].date()} .. {origins[-1].date()}), "
        f"horizon {MAX_HORIZON_HOURS}h"
    )

    all_scored = {name: [] for name in BASELINES}
    scale_frames = []
    for i, origin in enumerate(origins):
        train, test = train_test_split(hourly, origin)
        grid = target_grid(test, origin)
        scales = mase_scales(train)
        scales.index.name = "series_id"
        scale_frames.append(
            scales.rename("scale").to_frame().assign(origin=origin)
        )
        for name, fn in BASELINES.items():
            fc = fn(train, grid, origin)
            scored = evaluate(fc, hourly)
            all_scored[name].append(scored)
        print(f"  origin {i + 1}/{len(origins)} {origin.date()} done", file=sys.stderr)

    # Use per-origin scales; for summary simplicity take each series' mean scale.
    scales = (
        pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    )

    for name in BASELINES:
        scored = pd.concat(all_scored[name], ignore_index=True)
        print(f"\n=== {name} — by horizon bucket ===")
        print(summarize(scored, scales, by=("bucket",)).to_string())
        print(f"\n=== {name} — by park ===")
        print(summarize(scored, scales, by=("park",)).to_string())


if __name__ == "__main__":
    main()
