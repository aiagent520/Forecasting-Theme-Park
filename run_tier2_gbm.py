#!/usr/bin/env python3
"""Tier-2: global LightGBM on the hourly benchmark (M5-style GBM baseline).

One model over all 80 series, direct multi-horizon (lead is a feature).
Design:
  * Training samples are built from PSEUDO-ORIGINS on the same weekly x
    ORIGIN_HOURS cadence as the eval protocol, so the model is fit on the
    exact (lead, feature) distribution it predicts.
  * Features are origin-anchored (recency: last observed y, trailing means)
    or target-anchored seasonal lags at multiples of 168h — every referenced
    value is strictly before the sample's origin, so leakage-free by
    construction (max lead 168h == min seasonal lag).
  * For an eval origin O, the fit uses only samples whose pseudo-origin
    + 168h <= O (no training target overlaps the eval window). One fit per
    origin-DAY, shared by the three intraday origins (recency features still
    differ per origin; only the fitted parameters are shared).
  * objective=regression_l1 (matches MAE/MASE reporting), default-ish
    hyperparameters, no tuning — this is a baseline, not a tuned system.

Usage: python3 run_tier2_gbm.py [--origins N]
"""

import argparse
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import DATA_DIR, MAX_HORIZON_HOURS, ORIGIN_HOURS, ORIGIN_SPACING_DAYS
from harness import evaluate, mase_scales, rolling_origins, summarize

SEASONAL_LAGS = (168, 336, 504, 672)  # hours; all >= MAX_HORIZON_HOURS
PSEUDO_ORIGIN_MIN_DAYS = 28           # first pseudo-origin: 4 weeks of history

FEATURES = [
    "lead", "hour", "dow", "is_weekend", "month",
    "lag168", "lag336", "lag504", "lag672", "sdhr4",
    "last_y", "hrs_since_last", "mean_24h", "mean_7d", "mean_28d",
    "series_id", "park", "regime",
]
CATEGORICALS = ["series_id", "park", "regime"]


def pseudo_origins(hourly: pd.DataFrame) -> list[pd.Timestamp]:
    """Weekly x ORIGIN_HOURS origins, starting earlier than the eval origins
    (aligned: 28d vs 56d offset, both multiples of 7)."""
    start = hourly["ts"].min().normalize() + pd.Timedelta(days=PSEUDO_ORIGIN_MIN_DAYS)
    end = hourly["ts"].max().normalize() - pd.Timedelta(hours=MAX_HORIZON_HOURS)
    out = []
    t = start
    while t <= end:
        for h in ORIGIN_HOURS:
            out.append(t + pd.Timedelta(hours=h))
        t += pd.Timedelta(days=ORIGIN_SPACING_DAYS)
    return out


def build_samples(hourly: pd.DataFrame, lookup: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """Feature rows for all observed target hours in (origin, origin+168h),
    using only data strictly before `origin`."""
    train = hourly[hourly["ts"] < origin]
    horizon_end = origin + pd.Timedelta(hours=MAX_HORIZON_HOURS)
    fc = hourly.loc[
        (hourly["ts"] >= origin) & (hourly["ts"] < horizon_end),
        ["series_id", "ts", "y"],
    ].copy()
    if fc.empty:
        return fc.assign(origin=origin)
    fc["origin"] = origin
    fc["lead"] = ((fc["ts"] - origin).dt.total_seconds() // 3600).astype(int) + 1
    fc["hour"] = fc["ts"].dt.hour
    fc["dow"] = fc["ts"].dt.dayofweek
    fc["is_weekend"] = (fc["dow"] >= 5).astype(int)
    fc["month"] = fc["ts"].dt.month

    # target-anchored seasonal lags (ts - k*168h < origin always, since
    # lead <= 168h); the global lookup is therefore safe here
    lag_cols = []
    for lag_h in SEASONAL_LAGS:
        src = fc["ts"] - pd.Timedelta(hours=lag_h)
        col = f"lag{lag_h}"
        fc[col] = lookup.reindex(list(zip(fc["series_id"], src))).to_numpy()
        lag_cols.append(col)
    fc["sdhr4"] = fc[lag_cols].mean(axis=1)  # nan-aware mean of the 4 lags

    # origin-anchored recency / level features (from train only)
    last_rows = train.sort_values("ts").groupby("series_id").tail(1)
    rec = last_rows.set_index("series_id")[["ts", "y"]].rename(
        columns={"ts": "last_ts", "y": "last_y"}
    )
    rec["hrs_since_last"] = (origin - rec["last_ts"]).dt.total_seconds() / 3600
    for col, days in (("mean_24h", 1), ("mean_7d", 7), ("mean_28d", 28)):
        w = train[train["ts"] >= origin - pd.Timedelta(days=days)]
        rec[col] = w.groupby("series_id")["y"].mean()

    fc = fc.join(rec[["last_y", "hrs_since_last", "mean_24h", "mean_7d", "mean_28d"]],
                 on="series_id")
    return fc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=None)
    args = ap.parse_args()

    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    static = hourly[["series_id", "park", "regime"]].drop_duplicates("series_id")
    lookup = hourly.set_index(["series_id", "ts"])["y"]

    eval_origins = rolling_origins(hourly)
    if args.origins:
        eval_origins = eval_origins[-args.origins:]
    all_origins = sorted(set(pseudo_origins(hourly)) | set(eval_origins))
    print(
        f"Hourly data: {len(hourly):,} rows, {hourly['series_id'].nunique()} series | "
        f"{len(eval_origins)} eval origins ({eval_origins[0]} .. {eval_origins[-1]}) | "
        f"{len(all_origins)} sample origins"
    )

    # ---- precompute samples for every origin once -------------------------
    frames = []
    for i, o in enumerate(all_origins):
        frames.append(build_samples(hourly, lookup, o))
        if (i + 1) % 20 == 0:
            print(f"  samples {i + 1}/{len(all_origins)}", file=sys.stderr)
    samples = pd.concat(frames, ignore_index=True)
    samples = samples.merge(static, on="series_id", how="left")
    for c in CATEGORICALS:
        samples[c] = samples[c].astype("category")
    print(f"  sample matrix: {len(samples):,} rows", file=sys.stderr)

    horizon = pd.Timedelta(hours=MAX_HORIZON_HOURS)

    # ---- rolling-origin fit/predict (one fit per origin-day) --------------
    scored_frames = []
    scale_frames = []
    model, fit_day = None, None
    for i, origin in enumerate(eval_origins):
        day = origin.normalize()
        if day != fit_day:
            train_mask = samples["origin"] + horizon <= day  # tightest of the 3
            tr = samples[train_mask]
            model = LGBMRegressor(
                objective="regression_l1",
                n_estimators=400,
                learning_rate=0.06,
                num_leaves=63,
                min_child_samples=40,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                random_state=0,
                verbose=-1,
            )
            model.fit(tr[FEATURES], tr["y"])
            fit_day = day
            print(
                f"  fit {day.date()} on {len(tr):,} samples", file=sys.stderr
            )

        pred = samples[samples["origin"] == origin].copy()
        pred["y_hat"] = np.clip(model.predict(pred[FEATURES]), 0, None)
        scored_frames.append(
            evaluate(pred[["series_id", "origin", "ts", "y_hat"]], hourly)
        )

        train = hourly[hourly["ts"] < origin]
        scales = mase_scales(train)
        scales.index.name = "series_id"
        scale_frames.append(scales.rename("scale").to_frame().assign(origin=origin))
        print(f"  origin {i + 1}/{len(eval_origins)} {origin} done", file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    scored = pd.concat(scored_frames, ignore_index=True)

    print("\n=== gbm_global — by horizon bucket ===")
    print(summarize(scored, scales, by=("bucket",)).to_string())
    print("\n=== gbm_global — by park ===")
    print(summarize(scored, scales, by=("park",)).to_string())
    print("\n=== gbm_global — by regime x bucket ===")
    print(summarize(scored, scales, by=("regime", "bucket")).to_string())

    # feature importances (gain) for the last fitted model
    imp = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=FEATURES,
    ).sort_values(ascending=False)
    print("\n=== feature importance (gain, last fit) ===")
    print((imp / imp.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
