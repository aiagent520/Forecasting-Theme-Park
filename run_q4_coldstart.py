#!/usr/bin/env python3
"""Q4: cold-start cross-series transfer (Regime B / Epic as testbed, hourly).

Simulated cold start: at each eval origin-day D, EPIC's fine history is
truncated to its last k weeks ([D - 7k, D)); all other parks keep full
history. Factorial {k} x {pooling scope}, evaluated on Epic series only:

  k in K_WEEKS (None = full own era; sanity check vs gbm_global regime B)
  local_series - one model per Epic series, its truncated samples only
  local_park   - one model pooling the Epic series' truncated samples
  global_all   - Tier-2 global model: all non-Epic samples (full history,
                 tier2 cutoff origin+168h <= D) + Epic truncated samples

Model/features fixed: run_tier2_gbm's LightGBM + FEATURES everywhere; only
the pooling scope and Epic's available history vary.

Mechanics:
  * Epic samples/predictions are REBUILT per (D, k) from the truncated
    frame + truncated lookup, so recency AND seasonal-lag features honestly
    lose access to pre-window data (NaN) -- no feature leakage from the
    erased history.
  * Epic pseudo-origins: weekly x ORIGIN_HOURS back from D while >= window
    start + 7d. Epic training rows use row-level truncation ts < D (the
    tier2 whole-window cutoff would leave zero Epic samples at small k);
    with weekly spacing this only admits targets in [D-7d+ohr, D).
  * One fit per (D, k, scope), shared by the day's three intraday origins
    (same as tier2). MASE scales: full-history harness scales (fixed across
    cells, comparable with results_tier2_gbm regime B).

Usage: python3 run_q4_coldstart.py [--days N]
"""

import argparse
import sys

import numpy as np
import pandas as pd

from config import DATA_DIR, MAX_HORIZON_HOURS, ORIGIN_HOURS
from harness import evaluate, mase_scales, rolling_origins, summarize
from run_tier2_gbm import CATEGORICALS, FEATURES, build_samples, pseudo_origins
from run_q2_calibration import new_model

K_WEEKS = (2, 4, 8, None)
SCOPES = ("local_series", "local_park", "global_all")


def k_name(k) -> str:
    return "full" if k is None else f"k{k:02d}"


def epic_origin_grid(day: pd.Timestamp, win_start: pd.Timestamp,
                     eval_today: list[pd.Timestamp]) -> list[pd.Timestamp]:
    """Weekly x ORIGIN_HOURS pseudo-origins in [win_start+7d, day) + today's
    eval origins (prediction rows)."""
    out = list(eval_today)
    d = day - pd.Timedelta(days=7)
    while d >= win_start + pd.Timedelta(days=7):
        for h in ORIGIN_HOURS:
            out.append(d + pd.Timedelta(hours=h))
        d -= pd.Timedelta(days=7)
    return sorted(out)


def finalize(df: pd.DataFrame, static: pd.DataFrame, cats: dict) -> pd.DataFrame:
    df = df.merge(static, on="series_id", how="left")
    for c in CATEGORICALS:
        df[c] = pd.Categorical(df[c], categories=cats[c])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None,
                    help="limit to the first N eval origin-days (smoke test)")
    args = ap.parse_args()

    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    static = hourly[["series_id", "park", "regime"]].drop_duplicates("series_id")
    cats = {c: sorted(hourly[c].unique() if c != "series_id"
                      else hourly["series_id"].unique())
            for c in CATEGORICALS}

    epic = hourly[hourly["regime"] == "B"]
    rest = hourly[hourly["regime"] != "B"]
    epic_sids = sorted(epic["series_id"].unique())

    eval_origins = rolling_origins(hourly)
    eval_days = sorted({o.normalize() for o in eval_origins})
    if args.days:
        eval_days = eval_days[: args.days]
        eval_origins = [o for o in eval_origins if o.normalize() in set(eval_days)]
    print(
        f"Q4 cold start: {len(epic_sids)} Epic series, "
        f"{len(epic):,} Epic rows / {len(rest):,} rest rows | "
        f"{len(eval_days)} eval days ({eval_days[0].date()} .. "
        f"{eval_days[-1].date()}) | k in {[k or 'full' for k in K_WEEKS]}"
    )

    # non-Epic sample matrix, built once (tier2 style)
    rest_lookup = rest.set_index(["series_id", "ts"])["y"]
    rest_origins = sorted(set(pseudo_origins(rest)))
    frames = []
    for i, o in enumerate(rest_origins):
        frames.append(build_samples(rest, rest_lookup, o))
        if (i + 1) % 20 == 0:
            print(f"  rest samples {i + 1}/{len(rest_origins)}", file=sys.stderr)
    rest_samples = finalize(pd.concat(frames, ignore_index=True), static, cats)
    print(f"  rest sample matrix: {len(rest_samples):,} rows", file=sys.stderr)

    horizon = pd.Timedelta(hours=MAX_HORIZON_HOURS)
    scored = {}
    fc_frames = []
    scale_frames = []
    for i, day in enumerate(eval_days):
        eval_today = [o for o in eval_origins if o.normalize() == day]
        rest_tr = rest_samples[rest_samples["origin"] + horizon <= day]
        for origin in eval_today:
            scales = mase_scales(hourly[hourly["ts"] < origin])
            scales.index.name = "series_id"
            scale_frames.append(
                scales.rename("scale").to_frame().assign(origin=origin))
        for k in K_WEEKS:
            if k is None:
                win = epic
            else:
                win_start = day - pd.Timedelta(days=7 * k)
                win = epic[epic["ts"] >= win_start]
            if win.empty:
                continue
            lookup = win.set_index(["series_id", "ts"])["y"]
            grid = epic_origin_grid(day, win["ts"].min().normalize(), eval_today)
            eframes = [build_samples(win, lookup, o) for o in grid]
            esamples = pd.concat(
                [f for f in eframes if len(f)], ignore_index=True)
            esamples = finalize(esamples, static, cats)
            etr = esamples[esamples["ts"] < day]
            epred = esamples[esamples["origin"].isin(eval_today)]
            if etr.empty or epred.empty:
                continue

            fits = {}
            fits["local_park"] = [(etr, epred)]
            fits["local_series"] = [
                (etr[etr["series_id"] == s], epred[epred["series_id"] == s])
                for s in epic_sids
            ]
            fits["global_all"] = [
                (pd.concat([rest_tr, etr], ignore_index=True), epred)]
            for scope in SCOPES:
                cell = f"{k_name(k)}_{scope}"
                preds = []
                for tr, pr in fits[scope]:
                    if tr.empty or pr.empty:
                        continue
                    m = new_model()
                    m.fit(tr[FEATURES], tr["y"])
                    p = pr[["series_id", "origin", "ts"]].copy()
                    p["y_hat"] = np.clip(m.predict(pr[FEATURES]), 0, None)
                    preds.append(p)
                if not preds:
                    continue
                fc = pd.concat(preds, ignore_index=True)
                scored.setdefault(cell, []).append(evaluate(fc, hourly))
                fc_frames.append(fc.assign(cell=cell))
        print(f"  eval day {day.date()} done ({i + 1}/{len(eval_days)})",
              file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    out = pd.concat(fc_frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "q4_coldstart_fc.parquet", index=False)
    print(f"saved {len(out):,} forecast rows -> data/q4_coldstart_fc.parquet")

    for k in K_WEEKS:
        for scope in SCOPES:
            cell = f"{k_name(k)}_{scope}"
            if cell not in scored:
                continue
            sc = pd.concat(scored[cell], ignore_index=True)
            print(f"\n=== {cell} — Epic, by horizon bucket ===")
            print(summarize(sc, scales, by=("bucket",)).to_string())


if __name__ == "__main__":
    main()
