#!/usr/bin/env python3
"""Q2 Stage 1: explicit calibration layers vs learned recency (hourly protocol).

Factorial {base model} x {calibration stack}, all LightGBM (no library
confound), fine benchmark data held fixed:

  bases:
    cal - calendar/identity features only (hour, dow, weekend, month,
          series/park/regime) -> a learned climatology; structurally the
          production Layer-1 (no recency information at all).
    rec - full Tier-2 feature set (lead, seasonal lags, origin-anchored
          recency) == gbm_global.
  stacks: none / +L2 / +L2+L3, production semantics on the benchmark scale:
    L2 = per (series, hour) clip(mean actual / max(mean base fc, 1), 0.3, 3)
         over the trailing 30 days; closest-hour fallback, else 1.0.
    L3 = same ratio over the trailing 5 days against base*L2; stacked on top.
    (Production's 5-min floors are artifacts of its inflated scale; here the
    denominator floor is 1 min and final outputs clip at 0.)

  Calibration denominators use GENUINE out-of-sample base forecasts: weekly
  midnight-origin forecasts from *previous* origin-day fits partition past
  weeks exactly (origin spacing 7d == horizon 168h). Pseudo-origin days before
  the first eval origin serve as warm-up to seed the pool; the first fittable
  warm-up day has no earlier samples, so the earliest eval origins see a
  partially covered 30d window (missing (series,hour) cells fall back to 1.0).

Hypothesis under test: calibration layers and recency features are
substitutes -- L2/L3 should rescue `cal` and add ~nothing to `rec`.

Usage: python3 run_q2_calibration.py [--days N]
"""

import argparse
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import DATA_DIR, MAX_HORIZON_HOURS
from harness import evaluate, mase_scales, rolling_origins, summarize
from run_tier2_gbm import CATEGORICALS, FEATURES, build_samples, pseudo_origins

CAL_FEATURES = ["hour", "dow", "is_weekend", "month", "series_id", "park", "regime"]
BASES = {"cal": CAL_FEATURES, "rec": FEATURES}
L2_WINDOW_DAYS = 30
L3_WINDOW_DAYS = 5
RATIO_CLAMP = (0.3, 3.0)
DEN_FLOOR = 1.0


def new_model() -> LGBMRegressor:
    return LGBMRegressor(
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


def ratio_table(win: pd.DataFrame, pred_col: str) -> pd.Series:
    """clip(mean actual / max(mean forecast, floor)) per (series_id, hour)."""
    g = win.assign(series_id=win["series_id"].astype(str)).groupby(
        ["series_id", "hour"]
    ).agg(num=("y", "mean"), den=(pred_col, "mean"))
    return np.clip(g["num"] / np.maximum(g["den"], DEN_FLOOR), *RATIO_CLAMP)


def apply_ratio(df: pd.DataFrame, r: pd.Series) -> np.ndarray:
    """Row-wise ratio with production's closest-hour fallback (else 1.0)."""
    sids = df["series_id"].astype(str).to_numpy()
    idx = pd.MultiIndex.from_arrays([sids, df["hour"].to_numpy()])
    out = r.reindex(idx).to_numpy(dtype=float)
    missing = np.flatnonzero(np.isnan(out))
    if len(missing):
        by_sid: dict = {}
        for (sid, h), v in r.items():
            by_sid.setdefault(sid, {})[h] = v
        hours = df["hour"].to_numpy()
        for i in missing:
            d = by_sid.get(sids[i])
            if d:
                out[i] = d[min(d, key=lambda x: abs(x - hours[i]))]
            else:
                out[i] = 1.0
    return out


def calibrated(pred: pd.DataFrame, pool: pd.DataFrame, origin: pd.Timestamp,
               pred_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (l2, l2*l3) multipliers for `pred` rows at `origin`."""
    past = pool[pool["ts"] < origin]
    w2 = past[past["ts"] >= origin - pd.Timedelta(days=L2_WINDOW_DAYS)]
    r2 = ratio_table(w2, pred_col)
    l2 = apply_ratio(pred, r2)

    w3 = past[past["ts"] >= origin - pd.Timedelta(days=L3_WINDOW_DAYS)].copy()
    if w3.empty or r2.empty:
        return l2, l2
    w3["expected"] = w3[pred_col] * apply_ratio(w3, r2)
    r3 = ratio_table(w3.assign(**{pred_col: w3["expected"]}), pred_col)
    l3 = apply_ratio(pred, r3)
    return l2, l2 * l3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None,
                    help="limit to the first N eval origin-days (smoke test)")
    args = ap.parse_args()

    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    static = hourly[["series_id", "park", "regime"]].drop_duplicates("series_id")
    lookup = hourly.set_index(["series_id", "ts"])["y"]

    eval_origins = rolling_origins(hourly)
    all_origins = sorted(set(pseudo_origins(hourly)) | set(eval_origins))
    eval_days = sorted({o.normalize() for o in eval_origins})
    if args.days:
        eval_days = eval_days[: args.days]
        eval_origins = [o for o in eval_origins if o.normalize() in set(eval_days)]
        all_origins = [o for o in all_origins if o.normalize() <= eval_days[-1]]
    origin_days = sorted({o.normalize() for o in all_origins})
    print(
        f"Hourly data: {len(hourly):,} rows, {hourly['series_id'].nunique()} series | "
        f"{len(eval_origins)} eval origins | {len(origin_days)} origin-days "
        f"({origin_days[0].date()} .. {origin_days[-1].date()}, "
        f"{len(origin_days) - len(eval_days)} warm-up)"
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
    eval_day_set = set(eval_days)
    pool_frames = []       # out-of-sample midnight forecasts, both bases
    scored = {}            # cell name -> list of scored frames
    fc_frames = []         # saved forecasts, all cells
    scale_frames = []

    for i, day in enumerate(origin_days):
        tr = samples[samples["origin"] + horizon <= day]
        if tr.empty:
            print(f"  day {day.date()}: no training samples, skipped",
                  file=sys.stderr)
            continue
        models = {}
        for base, feats in BASES.items():
            m = new_model()
            m.fit(tr[feats], tr["y"])
            models[base] = m

        # out-of-sample midnight forecast -> calibration pool for later origins
        mid = samples[samples["origin"] == day].copy()
        if not mid.empty:
            for base, feats in BASES.items():
                mid[f"p_{base}"] = np.clip(models[base].predict(mid[feats]), 0, None)
            pool_frames.append(
                mid[["series_id", "ts", "hour", "y", "p_cal", "p_rec"]]
            )

        if day not in eval_day_set:
            print(f"  warm-up {day.date()} done ({len(tr):,} samples)",
                  file=sys.stderr)
            continue

        pool = pd.concat(pool_frames, ignore_index=True)
        for origin in [o for o in eval_origins if o.normalize() == day]:
            pred = samples[samples["origin"] == origin].copy()
            if pred.empty:
                continue
            for base, feats in BASES.items():
                p = np.clip(models[base].predict(pred[feats]), 0, None)
                pred[f"p_{base}"] = p
                l2, l2l3 = calibrated(pred, pool, origin, f"p_{base}")
                pred[f"{base}_none"] = p
                pred[f"{base}_l2"] = p * l2
                pred[f"{base}_l2l3"] = p * l2l3
            for cell in CELLS:
                fc = pred[["series_id", "origin", "ts"]].assign(
                    y_hat=pred[cell].to_numpy()
                )
                scored.setdefault(cell, []).append(evaluate(fc, hourly))
            fc_frames.append(
                pred[["series_id", "origin", "ts", "y"] + CELLS]
            )
            train = hourly[hourly["ts"] < origin]
            scales = mase_scales(train)
            scales.index.name = "series_id"
            scale_frames.append(
                scales.rename("scale").to_frame().assign(origin=origin)
            )
        print(f"  eval day {day.date()} done ({i + 1}/{len(origin_days)})",
              file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    out = pd.concat(fc_frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "q2_stage1_fc.parquet", index=False)
    print(f"saved {len(out):,} forecast rows -> data/q2_stage1_fc.parquet")

    for cell in CELLS:
        sc = pd.concat(scored[cell], ignore_index=True)
        print(f"\n=== {cell} — by horizon bucket ===")
        print(summarize(sc, scales, by=("bucket",)).to_string())
    for cell in CELLS:
        sc = pd.concat(scored[cell], ignore_index=True)
        print(f"\n=== {cell} — by regime x bucket ===")
        print(summarize(sc, scales, by=("regime", "bucket")).to_string())


CELLS = [f"{b}_{s}" for b in ("cal", "rec") for s in ("none", "l2", "l2l3")]


if __name__ == "__main__":
    main()
