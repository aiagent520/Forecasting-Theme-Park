#!/usr/bin/env python3
"""Statsforecast Tier-1 baselines on the hourly benchmark.

Models:
  mstl_ets — MSTL decomposition (daily 24h + weekly 168h seasonality) with
             AutoETS (no-seasonal, "ZZN") on the deseasonalized trend.
             Plain ETS cannot handle m=168; MSTL is the standard route.
  theta    — DynamicOptimizedTheta with weekly (168h) seasonality.

Irregular-series handling: parks close overnight, so observed hours have
gaps. For FITTING each series is placed on a regular hourly grid from its
first observation to the origin, unobserved hours filled with 0 (closed =>
no wait). Evaluation only touches observed hours (the target grid), so the
fill affects model fitting only. Documented in decisions doc.

Usage: python3 run_tier1_sf.py [--origins N] [--origin-hours 0,9,13]
"""

import argparse
import sys

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS, DynamicOptimizedTheta

from config import DATA_DIR, MAX_HORIZON_HOURS
from harness import (
    evaluate,
    mase_scales,
    rolling_origins,
    summarize,
    train_test_split,
)
from run_tier1 import target_grid


def make_models():
    return [
        MSTL(
            season_length=[24, 168],
            trend_forecaster=AutoETS(model="ZZN"),
            alias="mstl_ets",
        ),
        DynamicOptimizedTheta(season_length=168, alias="theta"),
    ]


MIN_FIT_HOURS = 2 * 168  # MSTL needs >= 2 full weekly periods to extract
                         # the 168h seasonal component


def to_regular_long(train: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Per-series regular hourly grid up to origin-1h, missing hours -> 0.

    Series with < MIN_FIT_HOURS of grid at this origin are skipped (e.g. a
    ride that opened recently); they simply have no forecast at this origin.
    """
    end = origin - pd.Timedelta(hours=1)
    frames = []
    for sid, g in train.groupby("series_id"):
        idx = pd.date_range(g["ts"].min(), end, freq="h")
        if len(idx) < MIN_FIT_HOURS:
            continue
        y = g.set_index("ts")["y"].reindex(idx, fill_value=0.0)
        frames.append(pd.DataFrame({"unique_id": sid, "ds": idx, "y": y.values}))
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=None)
    ap.add_argument("--origin-hours", type=str, default=None,
                    help="comma list, e.g. 0,9,13 (default: all)")
    args = ap.parse_args()

    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    origins = rolling_origins(hourly)
    if args.origin_hours:
        keep = {int(h) for h in args.origin_hours.split(",")}
        origins = [o for o in origins if o.hour in keep]
    if args.origins:
        origins = origins[-args.origins :]
    model_names = [m.alias for m in make_models()]
    print(
        f"Hourly data: {len(hourly):,} rows, {hourly['series_id'].nunique()} series | "
        f"{len(origins)} origins ({origins[0]} .. {origins[-1]}), "
        f"horizon {MAX_HORIZON_HOURS}h | models: {model_names}"
    )

    all_scored = {name: [] for name in model_names}
    scale_frames = []
    for i, origin in enumerate(origins):
        train, test = train_test_split(hourly, origin)
        grid = target_grid(test, origin)
        scales = mase_scales(train)
        scales.index.name = "series_id"
        scale_frames.append(scales.rename("scale").to_frame().assign(origin=origin))

        long = to_regular_long(train, origin)
        sf = StatsForecast(models=make_models(), freq="h", n_jobs=-1)
        fc = sf.forecast(df=long, h=MAX_HORIZON_HOURS)
        fc = fc.rename(columns={"unique_id": "series_id", "ds": "ts"})
        for name in model_names:
            m = grid.merge(
                fc[["series_id", "ts", name]].rename(columns={name: "y_hat"}),
                on=["series_id", "ts"],
                how="inner",
            )
            all_scored[name].append(evaluate(m, hourly))
        print(f"  origin {i + 1}/{len(origins)} {origin} done", file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()

    for name in model_names:
        scored = pd.concat(all_scored[name], ignore_index=True)
        print(f"\n=== {name} — by horizon bucket ===")
        print(summarize(scored, scales, by=("bucket",)).to_string())
        print(f"\n=== {name} — by park ===")
        print(summarize(scored, scales, by=("park",)).to_string())


if __name__ == "__main__":
    main()
