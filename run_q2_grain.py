#!/usr/bin/env python3
"""Q2 Stage 2: data-grain/source conditions under a FIXED model (daily 112d).

Complement to run_q2_calibration.py (which fixed the data and varied the
architecture). Here the model is fixed -- one global daily LightGBM, same
hyperparameters and feature set everywhere -- and only the training data
varies:

  fine        - own-era (5-min pipeline) daily means only; no 2025 data at
                all, so annual-lag features are empty (NaN).
  splice      - naive concatenation of coarse (queue_times) + own eras, the
                production choice; coarse levels are ~2.2-2.6x inflated vs
                own (no overlap window; see decisions doc).
  splice_cal  - coarse era rescaled per series by the hindsight-free
                boundary-window level ratio (run_daily._level_ratio: first
                56d of own era vs last 56d of coarse era, all strictly
                before the first eval origin), then concatenated.

The data condition acts through two channels at once: (a) training TARGETS
(splice trains on inflated levels), and (b) annual-lag FEATURE values at
prediction time (fine has none). Both are part of "what is this history
worth", so they are deliberately not separated here.

Protocol: identical to run_daily.py (weekly origins, 112d horizon, own-era
daily-mean targets, same MASE scales from the spliced train frame) so cells
are directly comparable with results_daily_tier1.txt.

Leakage: training uses row-level truncation (target date < eval origin)
rather than tier2's whole-window cutoff (origin + horizon <= eval day),
because with a 112d horizon the latter would erase nearly the entire
fine era. Each training row remains leakage-free: features are anchored at
its own origin (< target date < eval origin) and annual lags reach 364-350
days behind the target, always pre-origin at lead <= 112.

Usage: python3 run_q2_grain.py [--origins N]
"""

import argparse
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import DATA_DIR, MAX_HORIZON_DAYS
from run_daily import (
    _level_ratio,
    evaluate,
    mase_scales,
    rolling_origins,
    summarize,
)

FEATURES = [
    "lead", "dow", "is_weekend", "month",          # calendar
    "r7", "r28", "dow8w",                          # origin-anchored recency
    "lag364", "ann28",                             # target-anchored annual
    "series_id", "park", "regime",                 # identity
]
CATEGORICALS = ["series_id", "park", "regime"]
CONDITIONS = ("fine", "splice", "splice_cal")
MIN_ORIGIN_HISTORY_DAYS = 28


def new_model() -> LGBMRegressor:
    # identical to run_tier2_gbm / run_q2_calibration
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


def annual_lookups(df: pd.DataFrame) -> pd.DataFrame:
    """Per (series_id, date): lag364 and ann28 (29d centered mean at -364).

    Target-anchored and origin-independent; at lead <= 112 the newest value
    touched (date-350) is always before the row's origin.
    """
    parts = []
    for sid, g in df.groupby("series_id", observed=True):
        s = g.set_index("date")["y"].sort_index()
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"))
        roll = full.rolling(29, center=True, min_periods=7).mean()
        parts.append(pd.DataFrame({
            "series_id": sid,
            "date": s.index + pd.Timedelta(days=364),
            "lag364": s.to_numpy(),
            "ann28": roll.reindex(s.index).to_numpy(),
        }))
    return pd.concat(parts, ignore_index=True)


def recency_features(df: pd.DataFrame, origin: pd.Timestamp):
    """(per-series r7/r28, per series x dow trailing-8w dow mean), all < origin."""
    past = df[df["date"] < origin]
    w28 = past[past["date"] >= origin - pd.Timedelta(days=28)]
    w7 = w28[w28["date"] >= origin - pd.Timedelta(days=7)]
    r = pd.DataFrame({
        "r7": w7.groupby("series_id", observed=True)["y"].mean(),
        "r28": w28.groupby("series_id", observed=True)["y"].mean(),
    })
    w56 = past[past["date"] >= origin - pd.Timedelta(days=56)].copy()
    w56["dow"] = w56["date"].dt.dayofweek
    dow8w = w56.groupby(["series_id", "dow"], observed=True)["y"].mean().rename("dow8w")
    return r, dow8w


def build_samples(df: pd.DataFrame, origins: list[pd.Timestamp],
                  annual: pd.DataFrame, label: str) -> pd.DataFrame:
    frames = []
    horizon = pd.Timedelta(days=MAX_HORIZON_DAYS)
    for i, origin in enumerate(origins):
        rows = df[(df["date"] > origin) & (df["date"] <= origin + horizon)]
        if rows.empty:
            continue
        rows = rows[["series_id", "date", "y"]].copy()
        rows["origin"] = origin
        r, dow8w = recency_features(df, origin)
        rows = rows.join(r, on="series_id")
        rows["dow"] = rows["date"].dt.dayofweek
        rows = rows.join(dow8w, on=["series_id", "dow"])
        frames.append(rows)
        if (i + 1) % 25 == 0:
            print(f"  [{label}] samples {i + 1}/{len(origins)}", file=sys.stderr)
    out = pd.concat(frames, ignore_index=True)
    out["lead"] = (out["date"] - out["origin"]).dt.days
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = out["date"].dt.month
    out = out.merge(annual, on=["series_id", "date"], how="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=None,
                    help="limit to the first N eval origins (smoke test)")
    args = ap.parse_args()

    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    static = daily[["series_id", "park", "regime"]].drop_duplicates("series_id")
    actuals = daily[daily["source"] == "own_5min"]

    eval_origins = rolling_origins(daily)
    if args.origins:
        eval_origins = eval_origins[: args.origins]

    # hindsight-free level ratio: boundary windows end before the 1st origin
    ratio = _level_ratio(daily[daily["date"] < eval_origins[0]])
    print(f"level ratios: {len(ratio)} series, "
          f"median own/coarse = {ratio.median():.2f}")

    fine = daily[daily["source"] == "own_5min"].copy()
    cal = daily.copy()
    is_coarse = cal["source"] == "queue_times"
    cal_r = cal["series_id"].map(ratio)
    keep = ~is_coarse | cal_r.notna()
    cal = cal[keep].copy()
    cal.loc[cal["source"] == "queue_times", "y"] *= cal_r[keep][
        cal["source"] == "queue_times"
    ]
    frames_by_cond = {"fine": fine, "splice": daily, "splice_cal": cal}

    print(
        f"Daily data: {len(daily):,} rows | {len(eval_origins)} eval origins "
        f"({eval_origins[0].date()} .. {eval_origins[-1].date()}) | "
        f"conditions: " + ", ".join(
            f"{k}={len(v):,}" for k, v in frames_by_cond.items())
    )

    # weekly pseudo-origins on the eval cadence, back through each history
    def origin_grid(df):
        first = df["date"].min() + pd.Timedelta(days=MIN_ORIGIN_HISTORY_DAYS)
        o = eval_origins[0]
        pre = []
        while o - pd.Timedelta(days=7) >= first:
            o -= pd.Timedelta(days=7)
            pre.append(o)
        return sorted(pre) + eval_origins

    samples = {}
    for cond, df in frames_by_cond.items():
        annual = annual_lookups(df)
        samples[cond] = build_samples(df, origin_grid(df), annual, cond)
        print(f"  [{cond}] sample matrix: {len(samples[cond]):,} rows",
              file=sys.stderr)

    for cond in CONDITIONS:
        s = samples[cond].merge(static, on="series_id", how="left")
        for c in CATEGORICALS:
            s[c] = s[c].astype("category")
        samples[cond] = s

    scored = {cond: [] for cond in CONDITIONS}
    fc_frames = []
    scale_frames = []
    for i, origin in enumerate(eval_origins):
        train_full = daily[daily["date"] < origin]
        scales = mase_scales(train_full)
        scales.index.name = "series_id"
        scale_frames.append(
            scales.rename("scale").to_frame().assign(origin=origin))
        for cond in CONDITIONS:
            s = samples[cond]
            tr = s[s["date"] < origin]           # row-level truncation
            pred = s[s["origin"] == origin]
            if tr.empty or pred.empty:
                continue
            m = new_model()
            m.fit(tr[FEATURES], tr["y"])
            fc = pred[["series_id", "origin", "date"]].copy()
            fc["y_hat"] = np.clip(m.predict(pred[FEATURES]), 0, None)
            scored[cond].append(evaluate(fc, actuals))
            fc_frames.append(fc.assign(cond=cond))
        print(f"  origin {i + 1}/{len(eval_origins)} {origin.date()} done",
              file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    out = pd.concat(fc_frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "q2_stage2_fc.parquet", index=False)
    print(f"saved {len(out):,} forecast rows -> data/q2_stage2_fc.parquet")

    for cond in CONDITIONS:
        sc = pd.concat(scored[cond], ignore_index=True)
        print(f"\n=== gbm_{cond} — by lead bucket ===")
        print(summarize(sc, scales, by=("bucket",)).to_string())
    for cond in CONDITIONS:
        sc = pd.concat(scored[cond], ignore_index=True)
        print(f"\n=== gbm_{cond} — by regime x lead bucket ===")
        print(summarize(sc, scales, by=("regime", "bucket")).to_string())


if __name__ == "__main__":
    main()
