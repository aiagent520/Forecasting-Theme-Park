#!/usr/bin/env python3
"""Q3: learning curve / crossover — when does clean data retire dirty history?

At each eval origin we pretend the fine (own 5-min) pipeline started only k
weeks earlier: training sees fine data from [origin - 7k, origin) only.
Factorial {k weeks} x {aug}:

  k in K_WEEKS (None = full own era, the Stage-2 setting)
  aug=False - truncated fine window only
  aug=True  - + coarse (queue_times) history, level-calibrated with a ratio
              computable INSIDE the truncated window: per-series mean of the
              first <=56d of the window vs the last 56d of the coarse era
              (>=7 own days required, else that series' coarse rows drop).

Model fixed: the Stage-2 daily global LightGBM (same features/hyperparams).
Protocol: daily 112d, same origins/targets/MASE scales as run_q2_grain so
cells are comparable across k (scales are condition-independent).

Mechanics:
  - The coarse sample matrix is built ONCE, uncalibrated; every level-unit
    column (y, r7, r28, dow8w, lag364, ann28) is multiplied by the
    per-(origin,k) ratio at fit time (all are linear in y).
  - Fine samples are rebuilt per (origin, k): weekly pseudo-origins inside
    the window (>=7d of window history), targets < origin. Recency features
    are fine-only (never reach into the coarse era); missing -> NaN.
  - Annual-lag features for fine-window targets and prediction rows come
    from the coarse era (scaled by the ratio) under aug, else NaN. Fine-era
    lag-364 values never exist (own era < 1 year).
  - Eval on all regimes; regime A is the substantive cell (2yr coarse),
    C is the no-coarse control (aug adds only cross-series pooling rows).

Usage: python3 run_q3_curve.py [--origins N]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, MAX_HORIZON_DAYS
from run_daily import evaluate, mase_scales, rolling_origins, summarize
from run_q2_grain import (
    CATEGORICALS,
    FEATURES,
    annual_lookups,
    build_samples,
    new_model,
    recency_features,
)

K_WEEKS = (2, 4, 8, 16, None)
LEVEL_COLS = ["y", "r7", "r28", "dow8w", "lag364", "ann28"]
RATIO_WINDOW_DAYS = 56
MIN_RATIO_DAYS = 7


def cell_name(k, aug) -> str:
    base = "full" if k is None else f"k{k:02d}"
    return f"{base}_{'aug' if aug else 'fine'}"


def window_ratio(fine_win: pd.DataFrame, coarse_level: pd.Series,
                 origin: pd.Timestamp) -> pd.Series:
    """Per-series own/coarse ratio from the first <=56d of the k-week window."""
    own = fine_win[fine_win["date"] < origin]
    if own.empty:
        return pd.Series(dtype=float)
    start = own["date"].min()
    w = own[own["date"] < start + pd.Timedelta(days=RATIO_WINDOW_DAYS)]
    g = w.groupby("series_id", observed=True)["y"].agg(["mean", "size"])
    g = g[g["size"] >= MIN_RATIO_DAYS]
    ratio = g["mean"] / coarse_level.reindex(g.index)
    return ratio[np.isfinite(ratio)]


def fine_window_samples(fine_win: pd.DataFrame, origin: pd.Timestamp,
                        annual: pd.DataFrame) -> pd.DataFrame:
    """Training samples from weekly pseudo-origins inside the window."""
    start = fine_win["date"].min()
    pos = []
    po = origin - pd.Timedelta(days=7)
    while po >= start + pd.Timedelta(days=7):
        pos.append(po)
        po -= pd.Timedelta(days=7)
    if not pos:
        return pd.DataFrame()
    tr_rows = fine_win[fine_win["date"] < origin]
    out = build_samples(tr_rows, sorted(pos), annual, "fine_win")
    return out[out["date"] < origin]


def predict_rows(fine_win: pd.DataFrame, actuals: pd.DataFrame,
                 origin: pd.Timestamp, annual: pd.DataFrame) -> pd.DataFrame:
    horizon = pd.Timedelta(days=MAX_HORIZON_DAYS)
    rows = actuals[(actuals["date"] > origin) & (actuals["date"] <= origin + horizon)]
    rows = rows[["series_id", "date", "y"]].copy()
    rows["origin"] = origin
    r, dow8w = recency_features(fine_win, origin)
    rows = rows.join(r, on="series_id")
    rows["dow"] = rows["date"].dt.dayofweek
    rows = rows.join(dow8w, on=["series_id", "dow"])
    rows["lead"] = (rows["date"] - rows["origin"]).dt.days
    rows["is_weekend"] = (rows["dow"] >= 5).astype(int)
    rows["month"] = rows["date"].dt.month
    return rows.merge(annual, on=["series_id", "date"], how="left")


def with_annual(df: pd.DataFrame, coarse_annual: pd.DataFrame,
                ratio: pd.Series | None) -> pd.DataFrame:
    """Attach (ratio-scaled) coarse annual lags; ratio=None -> NaN columns."""
    df = df.drop(columns=["lag364", "ann28"], errors="ignore")
    if ratio is None:
        return df.assign(lag364=np.nan, ann28=np.nan)
    df = df.merge(coarse_annual, on=["series_id", "date"], how="left")
    r = df["series_id"].map(ratio)
    df["lag364"] *= r
    df["ann28"] *= r
    return df


def finalize(df: pd.DataFrame, static: pd.DataFrame,
             cats: dict) -> pd.DataFrame:
    df = df.merge(static, on="series_id", how="left")
    for c in CATEGORICALS:
        df[c] = pd.Categorical(df[c], categories=cats[c])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=None,
                    help="limit to the first N eval origins (smoke test)")
    args = ap.parse_args()

    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    static = daily[["series_id", "park", "regime"]].drop_duplicates("series_id")
    cats = {
        "series_id": sorted(daily["series_id"].unique()),
        "park": sorted(daily["park"].unique()),
        "regime": sorted(daily["regime"].unique()),
    }
    fine = daily[daily["source"] == "own_5min"]
    coarse = daily[daily["source"] == "queue_times"]
    actuals = fine

    eval_origins = rolling_origins(daily)
    if args.origins:
        eval_origins = eval_origins[: args.origins]
    print(
        f"Q3 curve: {len(eval_origins)} eval origins "
        f"({eval_origins[0].date()} .. {eval_origins[-1].date()}), "
        f"k in {[k or 'full' for k in K_WEEKS]}, fine rows {len(fine):,}, "
        f"coarse rows {len(coarse):,}"
    )

    # coarse machinery, built once (all coarse data predates all origins)
    coarse_annual = annual_lookups(coarse)
    first = coarse["date"].min() + pd.Timedelta(days=28)
    pos, po = [], eval_origins[0]
    while po - pd.Timedelta(days=7) >= first:
        po -= pd.Timedelta(days=7)
        pos.append(po)
    coarse_samples = build_samples(coarse, sorted(pos), coarse_annual, "coarse")
    coarse_samples = finalize(coarse_samples, static, cats)
    print(f"  coarse sample matrix: {len(coarse_samples):,} rows", file=sys.stderr)
    cend = coarse["date"].max()
    clast = coarse[coarse["date"] >= cend - pd.Timedelta(days=RATIO_WINDOW_DAYS)]
    coarse_level = clast.groupby("series_id", observed=True)["y"].mean()

    scored = {cell_name(k, a): [] for k in K_WEEKS for a in (False, True)}
    fc_frames = []
    scale_frames = []
    for i, origin in enumerate(eval_origins):
        scales = mase_scales(daily[daily["date"] < origin])
        scales.index.name = "series_id"
        scale_frames.append(
            scales.rename("scale").to_frame().assign(origin=origin))
        for k in K_WEEKS:
            if k is None:
                fine_win = fine
            else:
                fine_win = fine[fine["date"] >= origin - pd.Timedelta(days=7 * k)]
            ratio = window_ratio(fine_win, coarse_level, origin)
            fs_raw = fine_window_samples(fine_win, origin, coarse_annual.iloc[0:0])
            pr_raw = predict_rows(fine_win, actuals, origin, coarse_annual.iloc[0:0])
            for aug in (False, True):
                cell = cell_name(k, aug)
                r = ratio if aug else None
                fs = with_annual(fs_raw, coarse_annual, r) if len(fs_raw) else fs_raw
                pr = with_annual(pr_raw, coarse_annual, r)
                parts = []
                if len(fs):
                    parts.append(finalize(fs, static, cats))
                if aug and len(ratio):
                    cs = coarse_samples[
                        coarse_samples["series_id"].isin(ratio.index)
                    ].copy()
                    rr = cs["series_id"].map(ratio).astype(float)
                    for col in LEVEL_COLS:
                        cs[col] = cs[col] * rr
                    parts.append(cs)
                if not parts:
                    continue
                tr = pd.concat(parts, ignore_index=True)
                pred = finalize(pr, static, cats)
                m = new_model()
                m.fit(tr[FEATURES], tr["y"])
                fc = pred[["series_id", "origin", "date"]].copy()
                fc["y_hat"] = np.clip(m.predict(pred[FEATURES]), 0, None)
                scored[cell].append(evaluate(fc, actuals))
                fc_frames.append(fc.assign(cell=cell))
        print(f"  origin {i + 1}/{len(eval_origins)} {origin.date()} done",
              file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    out = pd.concat(fc_frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "q3_curve_fc.parquet", index=False)
    print(f"saved {len(out):,} forecast rows -> data/q3_curve_fc.parquet")

    cells = [cell_name(k, a) for k in K_WEEKS for a in (False, True)]
    for cell in cells:
        if not scored[cell]:
            continue
        sc = pd.concat(scored[cell], ignore_index=True)
        print(f"\n=== {cell} — by lead bucket ===")
        print(summarize(sc, scales, by=("bucket",)).to_string())
    for cell in cells:
        if not scored[cell]:
            continue
        sc = pd.concat(scored[cell], ignore_index=True)
        print(f"\n=== {cell} — by regime x lead bucket ===")
        print(summarize(sc, scales, by=("regime", "bucket")).to_string())


if __name__ == "__main__":
    main()
