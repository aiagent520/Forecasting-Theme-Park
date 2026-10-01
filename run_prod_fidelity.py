#!/usr/bin/env python3
"""Fidelity + deployed-accuracy check for the production trip-plan system.

Three questions:
  1. DEPLOYED ACCURACY — how good were the forecasts production actually
     served? Two stored assets in `forecast_trip_plan`:
       * 82 surviving next-day rows (run on day G at ~00:12 wrote targets
         G+1; data cutoff = midnight G) — true hindsight-free lead-1 set;
       * the frozen 2026-05-31 run (targets 2026-06-01 → 11-26) — a genuine
         long-lead snapshot, scoreable at leads 1-112d vs benchmark actuals.
     NOTE the deployed system's Layer 1 was STALE: forecast_trip_plan_base
     holds a single snapshot generated 2026-03-22; the daily job applied
     fresh L2/L3 to that base until 05-31.
  2. REPRODUCTION FIDELITY — does our retrained-at-origin backtest
     (run_prod_backtest.py) emit forecasts close to what production served?
     Compared on identical targets: (a) frozen snapshot vs an ad-hoc
     reproduction at origin-day 2026-05-31; (b) stored lead-1 rows vs the
     backtest's day-2 forecasts for the five May origin-days.
  3. SAME-TARGET ACCURACY — deployed vs reproduction on the identical
     target set (removes any grid-composition confound).

Run AFTER run_prod_backtest.py (needs data/prod_backtest_fc.parquet).
Usage: python3 run_prod_fidelity.py
"""

import sqlite3

import numpy as np
import pandas as pd

from config import DATA_DIR, DB_PATH, MAX_HORIZON_DAYS
import harness
import run_daily
import run_prod_backtest as rpb

FROZEN_DAY = pd.Timestamp("2026-05-31")


def load_stored():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT substr(generated_at, 1, 10) AS gen, target_date, target_hour,
                  ride_slug AS series_id, predicted_wait
           FROM forecast_trip_plan""",
        conn,
    )
    conn.close()
    df["gen"] = pd.to_datetime(df["gen"])
    df["date"] = pd.to_datetime(df["target_date"])
    df["ts"] = df["date"] + pd.to_timedelta(df["target_hour"], unit="h")
    lead1 = df[df["gen"] < FROZEN_DAY]
    frozen = df[df["gen"] == FROZEN_DAY]
    return lead1, frozen


def agreement(a: pd.Series, b: pd.Series) -> str:
    d = (a - b).abs()
    return (
        f"n={len(a):,}  MAE={d.mean():.2f}  corr={a.corr(b):.3f}  "
        f"mean {a.mean():.1f} vs {b.mean():.1f}"
    )


def main():
    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    hourly = hourly[hourly["resort"] == "Orlando"].copy()
    daily_bench = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily_bench["date"] = pd.to_datetime(daily_bench["date"])
    daily_actuals = daily_bench[
        (daily_bench["source"] == "own_5min")
        & (daily_bench["regime"].isin(["A", "B"]))
    ]
    lead1, frozen = load_stored()

    # ---- reproduce the pipeline at the frozen origin-day ------------------
    raw, v2, weather = rpb.load_inputs()
    D = FROZEN_DAY
    wx = rpb.monthly_weather(weather, D)
    h_avg, ride_avg = rpb.hourly_patterns(raw, D)
    model, ride_enc, park_enc = rpb.train_l1(v2, h_avg, ride_avg, wx, D)
    l1_30 = rpb._l1_hour_means(raw, model, h_avg, ride_avg, wx, ride_enc,
                               park_enc, D - pd.Timedelta(days=30), D)
    l1_5 = rpb._l1_hour_means(raw, model, h_avg, ride_avg, wx, ride_enc,
                              park_enc, D - pd.Timedelta(days=5), D)
    get_l2 = rpb._closest_hour_getter(rpb.compute_layer2(raw, D, l1_30))
    get_l3 = rpb._closest_hour_getter(
        rpb.compute_layer3(raw, D, get_l2, l1_5.to_dict())
    )
    span = hourly[
        (hourly["ts"] >= D)
        & (hourly["ts"] < D + pd.Timedelta(days=MAX_HORIZON_DAYS))
        & (hourly["series_id"].isin(ride_enc.classes_))
    ]
    grid = pd.DataFrame({
        "slug": span["series_id"].to_numpy(),
        "date": span["ts"].dt.normalize().to_numpy(),
        "hour": span["ts"].dt.hour.to_numpy(),
    })
    feats = rpb.build_l1_features(grid, h_avg, ride_avg, wx, ride_enc,
                                 park_enc, baseline_default=30)
    repro = pd.DataFrame({
        "series_id": span["series_id"].to_numpy(),
        "ts": span["ts"].to_numpy(),
        "date": grid["date"].to_numpy(),
        "l1": rpb.predict_l1(model, feats),
    })
    l2 = np.array([get_l2(s, h) for s, h in zip(grid["slug"], grid["hour"])])
    l3 = np.array([get_l3(s, h) for s, h in zip(grid["slug"], grid["hour"])])
    repro["full"] = np.clip(repro["l1"] * l2 * l3, 5, 300)

    d_scales = run_daily.mase_scales(daily_actuals[daily_actuals["date"] < D])
    d_scales.index.name = "series_id"

    # ---- 1a. frozen snapshot: deployed accuracy, daily protocol -----------
    # aggregate stored hourly predictions over each day's observed benchmark
    # hours so the daily mean is comparable to benchmark ground truth
    obs = hourly[["series_id", "ts"]]
    stored_obs = frozen.merge(obs, on=["series_id", "ts"], how="inner")
    stored_daily = (
        stored_obs.groupby(["series_id", "date"], as_index=False)["predicted_wait"]
        .mean()
        .rename(columns={"predicted_wait": "y_hat"})
        .assign(origin=D)
    )
    print("=== deployed frozen snapshot (gen 2026-05-31) — daily protocol ===")
    scored = run_daily.evaluate(stored_daily, daily_actuals)
    print(run_daily.summarize(scored, d_scales, by=("bucket",)).to_string())
    print("\n--- by regime x lead bucket ---")
    print(run_daily.summarize(scored, d_scales, by=("regime", "bucket")).to_string())

    # ---- 1b. reproduction at the same origin, same protocol ---------------
    repro_daily = (
        repro.groupby(["series_id", "date"], as_index=False)[["l1", "full"]]
        .mean()
        .assign(origin=D)
    )
    for col in ("l1", "full"):
        print(f"\n=== reproduction ({col}) at origin 2026-05-31 — daily protocol ===")
        scored = run_daily.evaluate(
            repro_daily[["series_id", "date", "origin"]].assign(
                y_hat=repro_daily[col].to_numpy()
            ),
            daily_actuals,
        )
        print(run_daily.summarize(scored, d_scales, by=("bucket",)).to_string())

    # ---- 2a. agreement: reproduction vs stored frozen (hourly rows) -------
    j = frozen.merge(
        repro[["series_id", "ts", "l1", "full"]], on=["series_id", "ts"],
        how="inner",
    )
    j["lead"] = (j["date"] - D).dt.days + 1
    j["bucket"] = run_daily._bucket(j["lead"])
    print("\n=== agreement: reproduced vs stored frozen forecasts ===")
    print("full pipeline:", agreement(j["full"], j["predicted_wait"]))
    for b, g in j.groupby("bucket", sort=False):
        print(f"  {b:>8}: {agreement(g['full'], g['predicted_wait'])}")

    # ---- 2b. agreement: backtest day-2 forecasts vs stored lead-1 rows ----
    fc = pd.read_parquet(DATA_DIR / "prod_backtest_fc.parquet")
    fc["ts"] = fc["date"] + pd.to_timedelta(fc["hour"], unit="h")
    day2 = fc[fc["date"] == fc["origin_day"] + pd.Timedelta(days=1)]
    j2 = day2.merge(
        lead1, left_on=["series_id", "ts", "origin_day"],
        right_on=["series_id", "ts", "gen"], how="inner",
    )
    print("\n=== agreement: backtest next-day vs stored next-day rows ===")
    print("full pipeline:", agreement(j2["full"], j2["predicted_wait"]))

    # ---- 3. same-target accuracy: deployed vs reproduction ----------------
    tgt = j.merge(hourly[["series_id", "ts", "y"]], on=["series_id", "ts"],
                  how="inner")
    print("\n=== same-target hourly accuracy (frozen grid ∩ benchmark) ===")
    for name, col in (("deployed", "predicted_wait"), ("repro_full", "full"),
                      ("repro_l1", "l1")):
        err = (tgt[col] - tgt["y"]).abs()
        print(f"  {name:>10}: n={len(tgt):,}  MAE={err.mean():.2f}  "
              f"w5={(err <= 5).mean():.2f}  w10={(err <= 10).mean():.2f}")

    tgt1 = lead1.merge(hourly[["series_id", "ts", "y"]], on=["series_id", "ts"],
                       how="inner")
    err = (tgt1["predicted_wait"] - tgt1["y"]).abs()
    print("\n=== deployed lead-1 rows (82 runs, Mar-May) vs actuals ===")
    print(f"  n={len(tgt1):,}  MAE={err.mean():.2f}  "
          f"w5={(err <= 5).mean():.2f}  w10={(err <= 10).mean():.2f}")


if __name__ == "__main__":
    main()
