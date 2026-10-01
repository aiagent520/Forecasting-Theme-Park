#!/usr/bin/env python3
"""Quantile variant of the Tier-2 global GBM (hourly protocol).

Same sample matrix, features, hyperparameters, and rolling-origin protocol
as run_tier2_gbm.py — the ONLY change is the objective: three LightGBM
fits per origin-day with objective="quantile" at alpha in {0.1, 0.5, 0.9}
(the l1 point model is alpha=0.5 in the limit, so p50 doubles as a sanity
anchor against the saved rec_none point forecasts).

Motivation (per-ride heterogeneity): pooled w10 hides a strong ride-size
gradient — headliners (mean wait 50m+) are within 10 min only ~39% of the
time even though their RELATIVE error is the best in the fleet (~23% MAE/
mean). For those rides an honest [p10, p90] band is more useful than any
point forecast, so the paper needs calibration evidence.

Metrics:
  * pinball loss per quantile (scaled by the per-series MASE scale so it
    aggregates across series, like everything else in the benchmark)
  * 80% interval: empirical coverage of [p10, p90] (target 0.80) + mean
    width (raw minutes and scaled)
  * p50 vs saved rec_none point forecasts (MAE by bucket, sanity)
  * coverage by regime x bucket and by ride-size bin (the motivation cut)

Quantile crossing is repaired by sorting the three predictions per row
(standard post-hoc fix); predictions are clipped at 0.

Usage: python3 run_quantile_gbm.py [--origins N] | tee results_quantile_gbm.txt
"""

import argparse
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import DATA_DIR, HORIZON_BUCKETS, MAX_HORIZON_HOURS
from harness import evaluate, mase_scales, rolling_origins
from run_tier2_gbm import CATEGORICALS, FEATURES, build_samples, pseudo_origins

QUANTILES = (0.1, 0.5, 0.9)
QCOLS = [f"p{int(q * 100)}" for q in QUANTILES]
SIZE_BINS = [0, 10, 20, 30, 50, np.inf]
SIZE_LABELS = ["<10m", "10-20m", "20-30m", "30-50m", "50m+"]


def make_model(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
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


def pinball(y: pd.Series, yhat: pd.Series, q: float) -> pd.Series:
    d = y - yhat
    return np.maximum(q * d, (q - 1) * d)


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
        f"{len(all_origins)} sample origins | quantiles {QUANTILES}"
    )

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

    fc_frames, scale_frames = [], []
    models, fit_day = None, None
    for i, origin in enumerate(eval_origins):
        day = origin.normalize()
        if day != fit_day:
            tr = samples[samples["origin"] + horizon <= day]
            models = {}
            for q in QUANTILES:
                m = make_model(q)
                m.fit(tr[FEATURES], tr["y"])
                models[q] = m
            fit_day = day
            print(f"  fit {day.date()} on {len(tr):,} samples x {len(QUANTILES)} quantiles",
                  file=sys.stderr)

        pred = samples[samples["origin"] == origin].copy()
        raw = np.column_stack(
            [models[q].predict(pred[FEATURES]) for q in QUANTILES]
        )
        raw = np.clip(np.sort(raw, axis=1), 0, None)   # fix crossing, clip
        for j, col in enumerate(QCOLS):
            pred[col] = raw[:, j]
        fc_frames.append(pred[["series_id", "origin", "ts"] + QCOLS])

        scales = mase_scales(hourly[hourly["ts"] < origin])
        scales.index.name = "series_id"
        scale_frames.append(scales.rename("scale").to_frame().assign(origin=origin))
        print(f"  origin {i + 1}/{len(eval_origins)} {origin} done", file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    fc = pd.concat(fc_frames, ignore_index=True)
    fc.to_parquet(DATA_DIR / "quantile_gbm_fc.parquet", index=False)
    print(f"saved {len(fc):,} forecast rows -> data/quantile_gbm_fc.parquet")

    # ---- score: bucket/regime via evaluate() on p50, then join the band ----
    sc = evaluate(fc.rename(columns={"p50": "y_hat"}), hourly)  # keeps p10/p90
    sc = sc.rename(columns={"y_hat": "p50"}).join(scales.rename("scale"),
                                                  on="series_id")
    sc = sc[sc["scale"].notna()].copy()
    sc["bucket"] = pd.Categorical(sc["bucket"], categories=list(HORIZON_BUCKETS),
                                  ordered=True)
    for q, col in zip(QUANTILES, QCOLS):
        sc[f"pb{col[1:]}"] = pinball(sc["y"], sc[col], q) / sc["scale"]
    sc["cover"] = ((sc["y"] >= sc["p10"]) & (sc["y"] <= sc["p90"])).astype(float)
    sc["width"] = sc["p90"] - sc["p10"]
    sc["width_s"] = sc["width"] / sc["scale"]

    agg = dict(n=("cover", "size"), cover80=("cover", "mean"),
               width=("width", "mean"), width_s=("width_s", "mean"),
               pb10=("pb10", "mean"), pb50=("pb50", "mean"),
               pb90=("pb90", "mean"))

    print("\n=== quantile gbm — by horizon bucket ===")
    print(sc.groupby("bucket", observed=True).agg(**agg).round(3).to_string())
    print("\n=== quantile gbm — by regime x bucket ===")
    print(sc.groupby(["regime", "bucket"], observed=True).agg(**agg).round(3).to_string())

    # ride-size cut (the motivation): coverage on headliners vs kiddie rides
    lvl = hourly.groupby("series_id")["y"].mean().rename("mean_wait")
    sc = sc.join(lvl, on="series_id")
    sc["size_bin"] = pd.cut(sc["mean_wait"], SIZE_BINS, labels=SIZE_LABELS)
    print("\n=== quantile gbm — by ride-size bin (mean wait) ===")
    print(sc.groupby("size_bin", observed=True).agg(**agg).round(3).to_string())

    # ---- sanity: p50 vs the saved rec_none point model ---------------------
    s1 = pd.read_parquet(DATA_DIR / "q2_stage1_fc.parquet")
    pt = s1[["series_id", "origin", "ts", "rec_none"]].copy()
    pt["series_id"] = pt["series_id"].astype(str)
    both = sc.merge(pt, on=["series_id", "origin", "ts"], how="inner")
    both["ae_p50"] = (both["y"] - both["p50"]).abs()
    both["ae_pt"] = (both["y"] - both["rec_none"]).abs()
    cmp = both.groupby("bucket", observed=True)[["ae_p50", "ae_pt"]].mean()
    cmp.columns = ["mae_p50", "mae_rec_none"]
    print(f"\n=== sanity: p50 vs rec_none point model (n={len(both):,} joint rows) ===")
    print(cmp.round(2).to_string())


if __name__ == "__main__":
    main()
