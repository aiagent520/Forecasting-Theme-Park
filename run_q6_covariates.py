#!/usr/bin/env python3
"""Q6: which input signals matter at which lead times? (covariate ablation)

Everything is held fixed at the best-known configuration — the Stage-2 daily
global LightGBM trained on the CALIBRATED SPLICE (run_q2_grain splice_cal) —
and only the FEATURE SET varies:

  base        - the Stage-2 feature set (calendar + origin-anchored recency
                r7/r28/dow8w + target-anchored annual lag364/ann28 + identity)
  no_recency  - base minus r7/r28/dow8w -> how much recency is worth, by lead
  school      - base + Orlando school-break calendar (district-weighted break
                scores, districts-on-break count, holiday score)
  events      - base + park event / peak-week scores
  weather     - base + weather at the TARGET DATE. Actuals, not forecasts ->
                an ORACLE UPPER BOUND on what any weather signal could add
                (real deployments only have skillful weather ~10d out)
  all         - base + school + events + weather

Covariates are Orlando-only (crowd_calendar, weather_daily); for regime C
rows (USJ/USH) they are set to NaN and LightGBM routes them down the
missing-value branch. Regime C therefore acts as a built-in control: its
rows never see a covariate value, so regime-stratified tables show whether
adding Orlando signals to the global model helps Orlando without hurting
the covariate-less parks.

Leakage: school/holiday/event calendars are known far in advance ->
legitimately available at any lead. Weather is the one oracle cell and is
labelled as such. All base features follow run_q2_grain's leakage rules
(row-level truncation; recency anchored at each row's own origin).

Protocol, sample construction, model, and MASE scales are identical to
run_q2_grain (daily 112d, weekly origins, per-origin-averaged scales), so
`base` here reproduces gbm_splice_cal there up to the shared pipeline.

Runs AFTER run_q2_grain.py conceptually but is self-contained (rebuilds the
splice_cal frame). Usage:
    python3 run_q6_covariates.py [--origins N] | tee results_q6_covariates.txt
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd

from config import DAILY_HORIZON_BUCKETS, DATA_DIR, DB_PATH
from run_daily import _level_ratio, evaluate, mase_scales, rolling_origins, summarize
from run_q2_grain import (
    CATEGORICALS,
    FEATURES,
    MIN_ORIGIN_HISTORY_DAYS,
    annual_lookups,
    build_samples,
    new_model,
)
from run_significance import DAILY_LAG, report

SCHOOL = [
    "total_break_score", "spring_break_score", "summer_break_score",
    "winter_break_score", "thanksgiving_break_score",
    "num_districts_on_break", "holiday_score",
]
EVENTS = ["event_score", "peak_week_score"]
WEATHER = ["temp_max_f", "temp_min_f", "humidity", "precip_prob",
           "is_rainy", "is_hot"]
COVARIATES = SCHOOL + EVENTS + WEATHER

RECENCY = ["r7", "r28", "dow8w"]
FEATURE_SETS = {
    "base": FEATURES,
    "no_recency": [f for f in FEATURES if f not in RECENCY],
    "school": FEATURES + SCHOOL,
    "events": FEATURES + EVENTS,
    "weather": FEATURES + WEATHER,
    "all": FEATURES + COVARIATES,
}
CONDITIONS = list(FEATURE_SETS)


def load_covariates() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        cal = pd.read_sql(
            "SELECT date, " + ", ".join(SCHOOL + EVENTS) + " FROM crowd_calendar",
            con,
        )
        wx = pd.read_sql(
            "SELECT date, " + ", ".join(WEATHER) + " FROM weather_daily", con
        )
    finally:
        con.close()
    cov = cal.merge(wx, on="date", how="outer")
    cov["date"] = pd.to_datetime(cov["date"])
    return cov


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

    # splice_cal frame, exactly as in run_q2_grain
    ratio = _level_ratio(daily[daily["date"] < eval_origins[0]])
    cal = daily.copy()
    is_coarse = cal["source"] == "queue_times"
    cal_r = cal["series_id"].map(ratio)
    keep = ~is_coarse | cal_r.notna()
    cal = cal[keep].copy()
    cal.loc[cal["source"] == "queue_times", "y"] *= cal_r[keep][
        cal["source"] == "queue_times"
    ]

    # weekly pseudo-origins on the eval cadence, back through the history
    first = cal["date"].min() + pd.Timedelta(days=MIN_ORIGIN_HISTORY_DAYS)
    o, pre = eval_origins[0], []
    while o - pd.Timedelta(days=7) >= first:
        o -= pd.Timedelta(days=7)
        pre.append(o)
    origins = sorted(pre) + eval_origins

    annual = annual_lookups(cal)
    s = build_samples(cal, origins, annual, "splice_cal")
    s = s.merge(static, on="series_id", how="left")

    cov = load_covariates()
    s = s.merge(cov, on="date", how="left")
    s.loc[s["regime"] == "C", COVARIATES] = np.nan   # Orlando-only signals
    for c in CATEGORICALS:
        s[c] = s[c].astype("category")

    n_orl = (s["regime"] != "C").sum()
    n_cov = ((s["regime"] != "C") & s["total_break_score"].notna()).sum()
    print(
        f"sample matrix: {len(s):,} rows | {len(eval_origins)} eval origins "
        f"({eval_origins[0].date()} .. {eval_origins[-1].date()}) | "
        f"calendar coverage on Orlando rows: {n_cov / n_orl:.0%}"
    )

    scored = {cond: [] for cond in CONDITIONS}
    fc_frames = []
    scale_frames = []
    for i, origin in enumerate(eval_origins):
        train_full = daily[daily["date"] < origin]
        scales = mase_scales(train_full)
        scales.index.name = "series_id"
        scale_frames.append(
            scales.rename("scale").to_frame().assign(origin=origin))
        tr = s[s["date"] < origin]               # row-level truncation
        pred = s[s["origin"] == origin]
        if tr.empty or pred.empty:
            continue
        for cond, feats in FEATURE_SETS.items():
            m = new_model()
            m.fit(tr[feats], tr["y"])
            fc = pred[["series_id", "origin", "date"]].copy()
            fc["y_hat"] = np.clip(m.predict(pred[feats]), 0, None)
            scored[cond].append(evaluate(fc, actuals))
            fc_frames.append(fc.assign(cond=cond))
        print(f"  origin {i + 1}/{len(eval_origins)} {origin.date()} done",
              file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    out = pd.concat(fc_frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "q6_covariates_fc.parquet", index=False)
    print(f"saved {len(out):,} forecast rows -> data/q6_covariates_fc.parquet")

    for cond in CONDITIONS:
        sc = pd.concat(scored[cond], ignore_index=True)
        print(f"\n=== q6_{cond} — by lead bucket ===")
        print(summarize(sc, scales, by=("bucket",)).to_string())
    for cond in CONDITIONS:
        sc = pd.concat(scored[cond], ignore_index=True)
        print(f"\n=== q6_{cond} — by regime x lead bucket ===")
        print(summarize(sc, scales, by=("regime", "bucket")).to_string())

    # --- DM + MCB -----------------------------------------------------------
    sc = pd.concat(
        [pd.concat(scored[cond], ignore_index=True).assign(cond=cond)
         for cond in CONDITIONS],
        ignore_index=True,
    ).join(scales, on="series_id")
    sc = sc[sc["scale"].notna()].copy()
    sc["sae"] = sc["abs_err"] / sc["scale"]
    wide = sc.pivot_table(
        index=["series_id", "regime", "origin", "bucket", "date"],
        columns="cond", values="sae", aggfunc="first",
    ).reset_index()
    wide["day"] = wide["origin"]

    buckets = list(DAILY_HORIZON_BUCKETS)
    pairs = [
        ("school", "base"),
        ("events", "base"),
        ("weather", "base"),
        ("all", "base"),
        ("base", "no_recency"),
    ]
    report(
        "Q6 covariates (daily, all regimes)", wide, CONDITIONS,
        pairs=pairs, buckets=buckets, lag=DAILY_LAG,
        note="daily long buckets: overlapping origins -> anti-conservative p",
    )
    report(
        "Q6 covariates (daily, regime A only)",
        wide[wide["regime"] == "A"], CONDITIONS,
        pairs=pairs, buckets=buckets, lag=DAILY_LAG,
    )
    report(
        "Q6 covariates (daily, regime C only — covariate-less control)",
        wide[wide["regime"] == "C"], CONDITIONS,
        pairs=pairs, buckets=buckets, lag=DAILY_LAG,
    )


if __name__ == "__main__":
    main()
