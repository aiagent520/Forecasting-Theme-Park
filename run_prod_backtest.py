#!/usr/bin/env python3
"""Backtest reproduction of the deployed 3-layer XGBoost trip-plan system.

Faithful re-implementation of production (themeparkdata/src/train_trip_plan.py,
forecast_trip_plan_weekly.py, forecast_trip_plan_daily.py), retrained at every
weekly origin-day with data strictly before the origin:

  L1  XGBoost (200 trees, depth 8) trained on ride_daily_history_v2 daily
      avg_wait (>0, Orlando slugs) expanded to hours 8-22 via per-(slug,hour)
      ratios from raw wait_times (status OPEN, wait>0); features: cyclical
      calendar, monthly-climatology weather, hardcoded school-break scores,
      label-encoded slug/park, ride_hourly_baseline. Predictions clip [5,300].
  L2  per (slug,hour): clamp(all-history actual mean / max(mean L1 over the
      trailing 30d grid, 5), 0.3, 3.0); closest-hour fallback, else 1.0.
  L3  per (slug,hour): clamp(5-day actual mean / (5-day L1 mean x L2),
      0.3, 3.0) when L1>=5 and expected>=5; closest-hour fallback, else 1.0.
  Final = clip(L1 x L2 x L3, 5, 300); refreshed daily at midnight, so the
      three intraday eval origins of a day share one forecast (the deployed
      system has no intraday trip-plan update — part of what is measured).

Documented deviations from production (necessary for a rolling backtest):
  * all inputs restricted to dates < origin-day (production trains ad hoc on
    whatever exists at training time);
  * L1 retrained each origin-day (weekly cadence — matches the weekly base
    regeneration; production retrains the model file less often);
  * L2/L3 denominator grids use each ride's OBSERVED (date, hour) operating
    hours instead of the park_hours table;
  * weather monthly climatology restricted to dates < origin-day.

Evaluated on BOTH benchmark protocols against benchmark ground truth
(zeros included — the production wait>0 semantics are part of the system
under evaluation), Orlando series only (regimes A + B):
  * hourly protocol: 168h horizon from origins at 00/09/13;
  * daily protocol: 112d horizon from midnight origins (predictions averaged
    over each day's observed benchmark hours).
Two variants scored: prod_l1 (Layer 1 alone = forecast_trip_plan_base) and
prod_full (L1 x L2 x L3 = forecast_trip_plan).

Usage: python3 run_prod_backtest.py [--origins N]   (N = limit origin-days)
"""

import argparse
import sqlite3
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATA_DIR, DB_PATH, MAX_HORIZON_DAYS, MAX_HORIZON_HOURS, ORIGIN_HOURS
import harness
import run_daily

UOR_LIKE = ("uor.ioa.%", "uor.usf.%", "uor.ueu.%")
PARK_PREFIX_MAP = {"uor.ioa.": 64, "uor.usf.": 65, "uor.ueu.": 334}
TRAIN_HOURS = list(range(8, 23))  # production trains on 8am-10pm rows

FEATURES = [
    "day_of_year", "day_of_year_sin", "day_of_year_cos",
    "day_of_week", "day_of_week_sin", "day_of_week_cos",
    "month", "month_sin", "month_cos",
    "is_weekend",
    "hour", "hour_sin", "hour_cos",
    "is_peak_hour", "is_evening", "is_morning",
    "temp_max_f", "temp_min_f", "humidity", "precip_prob",
    "is_hot", "is_rainy",
    "spring_break_score", "summer_break_score", "winter_break_score",
    "ride_slug_encoded", "park_id_encoded",
    "ride_hourly_baseline",
]

XGB_PARAMS = dict(
    n_estimators=200, max_depth=8, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_inputs():
    conn = sqlite3.connect(DB_PATH)
    uor = " OR ".join(f"attraction_id LIKE '{p}'" for p in UOR_LIKE)
    raw = pd.read_sql(
        f"""SELECT attraction_id AS slug, date,
                   CAST(SUBSTR(time, 1, 2) AS INTEGER) AS hour,
                   COUNT(*) AS cnt, SUM(wait_time) AS s
            FROM wait_times
            WHERE status = 'OPEN' AND wait_time > 0 AND ({uor})
            GROUP BY attraction_id, date, hour""",
        conn,
    )
    v2 = pd.read_sql(
        """SELECT slug, ride_name, park_id, date, avg_wait
           FROM ride_daily_history_v2
           WHERE slug LIKE 'uor.ioa.%' OR slug LIKE 'uor.usf.%'
              OR slug LIKE 'uor.ueu.%'""",
        conn,
    )
    weather = pd.read_sql(
        "SELECT date, temp_max_f, temp_min_f, humidity, precip_prob FROM weather_daily",
        conn,
    )
    conn.close()
    raw["date"] = pd.to_datetime(raw["date"])
    v2["date"] = pd.to_datetime(v2["date"])
    v2 = v2[v2["avg_wait"] > 0].copy()
    for prefix, pid in PARK_PREFIX_MAP.items():
        v2.loc[v2["slug"].str.startswith(prefix), "park_id"] = pid
    weather["date"] = pd.to_datetime(weather["date"])
    return raw, v2, weather

# ---------------------------------------------------------------------------
# Feature pieces (formulas identical to production)
# ---------------------------------------------------------------------------

def calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = df[date_col]
    out = pd.DataFrame(index=df.index)
    out["day_of_year"] = d.dt.dayofyear
    out["day_of_week"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365)
    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    day = d.dt.day
    out["spring_break_score"] = np.where(
        ((out["month"] == 3) & (day >= 10)) | ((out["month"] == 4) & (day <= 15)),
        0.9, 0.2,
    )
    out["summer_break_score"] = np.where(
        (out["month"].isin([6, 7])) | ((out["month"] == 8) & (day <= 15)),
        0.9, 0.1,
    )
    out["winter_break_score"] = np.where(
        ((out["month"] == 12) & (day >= 20)) | ((out["month"] == 1) & (day <= 5)),
        0.9, 0.1,
    )
    return out


def hour_features(hours: pd.Series) -> pd.DataFrame:
    """Callers already carry the 'hour' column; only the derived ones here."""
    out = pd.DataFrame(index=hours.index)
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    out["is_peak_hour"] = ((hours >= 10) & (hours <= 14)).astype(int)
    out["is_evening"] = (hours >= 17).astype(int)
    out["is_morning"] = (hours <= 10).astype(int)
    return out


def monthly_weather(weather: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    w = weather[weather["date"] < before]
    m = w.groupby(w["date"].dt.month).agg(
        temp_max_f=("temp_max_f", "mean"), temp_min_f=("temp_min_f", "mean"),
        humidity=("humidity", "mean"), precip_prob=("precip_prob", "mean"),
    )
    m = m.reindex(range(1, 13)).fillna(
        {"temp_max_f": 80, "temp_min_f": 65, "humidity": 70, "precip_prob": 20}
    )
    m["is_hot"] = (m["temp_max_f"] > 85).astype(int)
    m["is_rainy"] = (m["precip_prob"] > 30).astype(int)
    m.index.name = "month"
    return m


def hourly_patterns(raw: pd.DataFrame, before: pd.Timestamp):
    """(h_avg per slug-hour with >=3 samples, all-history mean per slug)."""
    r = raw[raw["date"] < before]
    by_sh = r.groupby(["slug", "hour"]).agg(cnt=("cnt", "sum"), s=("s", "sum"))
    by_sh = by_sh[by_sh["cnt"] >= 3]
    h_avg = (by_sh["s"] / by_sh["cnt"]).rename("h_avg")
    by_s = r.groupby("slug").agg(cnt=("cnt", "sum"), s=("s", "sum"))
    ride_avg = (by_s["s"] / by_s["cnt"]).rename("ride_avg")
    return h_avg, ride_avg

# ---------------------------------------------------------------------------
# Layer 1
# ---------------------------------------------------------------------------

def build_l1_features(grid, h_avg, ride_avg, wx, ride_enc, park_enc,
                      baseline_default):
    """grid: [slug, date, hour(, park_id)] -> production feature frame.
    baseline_default: per-row fallback when the (slug,hour) pattern is
    missing — training uses the slug's ride_avg (or the row's avg_wait when
    the slug has no pattern at all); inference uses the constant 30."""
    fc = grid.copy()
    if "park_id" not in fc.columns:
        fc["park_id"] = 64
        for prefix, pid in PARK_PREFIX_MAP.items():
            fc.loc[fc["slug"].str.startswith(prefix), "park_id"] = pid
    fc = pd.concat(
        [fc, calendar_features(fc), hour_features(fc["hour"])], axis=1
    )
    fc = fc.join(wx, on="month")
    fc = fc.join(h_avg, on=["slug", "hour"])
    fc["ride_hourly_baseline"] = fc["h_avg"].fillna(baseline_default)
    known = dict(zip(ride_enc.classes_, ride_enc.transform(ride_enc.classes_)))
    fc["ride_slug_encoded"] = fc["slug"].map(known).fillna(0).astype(int)
    kp = dict(zip(park_enc.classes_, park_enc.transform(park_enc.classes_)))
    fc["park_id_encoded"] = fc["park_id"].map(kp).fillna(0).astype(int)
    return fc


def train_l1(v2, h_avg, ride_avg, wx, origin_day):
    daily = v2[v2["date"] < origin_day]
    # cross-join daily rows x hours 8..22
    tr = daily.loc[daily.index.repeat(len(TRAIN_HOURS))].reset_index(drop=True)
    tr["hour"] = np.tile(TRAIN_HOURS, len(daily))
    tr = tr.join(h_avg, on=["slug", "hour"])
    tr = tr.join(ride_avg, on="slug")
    # slugs with at least one qualifying pattern hour (production dict keys)
    in_patterns = tr["slug"].isin(h_avg.index.get_level_values("slug"))
    ra = np.where(in_patterns, tr["ride_avg"].fillna(30), tr["avg_wait"])
    ratio = np.where(tr["h_avg"].notna() & (ra > 0), tr["h_avg"] / ra, 1.0)
    tr["wait_time"] = tr["avg_wait"] * ratio
    tr["ride_hourly_baseline"] = np.where(tr["h_avg"].notna(), tr["h_avg"], ra)
    tr = pd.concat([tr, calendar_features(tr), hour_features(tr["hour"])], axis=1)
    tr = tr.join(wx, on="month")
    ride_enc = LabelEncoder().fit(tr["slug"])
    park_enc = LabelEncoder().fit(tr["park_id"])
    tr["ride_slug_encoded"] = ride_enc.transform(tr["slug"])
    tr["park_id_encoded"] = park_enc.transform(tr["park_id"])
    X_train, _, y_train, _ = train_test_split(
        tr[FEATURES], tr["wait_time"], test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model, ride_enc, park_enc


def predict_l1(model, feats):
    return np.clip(model.predict(feats[FEATURES]), 5, 300)

# ---------------------------------------------------------------------------
# Layers 2 & 3 (dict + closest-hour fallback, like production)
# ---------------------------------------------------------------------------

def _closest_hour_getter(d: dict):
    hours_by_slug = {}
    for (slug, hour) in d:
        hours_by_slug.setdefault(slug, set()).add(hour)

    def get(slug, hour):
        if (slug, hour) in d:
            return d[(slug, hour)]
        avail = hours_by_slug.get(slug)
        if not avail:
            return 1.0
        closest = min(avail, key=lambda h: abs(h - hour))
        return d.get((slug, closest), 1.0)

    return get


def _l1_hour_means(raw, model, h_avg, ride_avg, wx, ride_enc, park_enc,
                   start, end):
    """Mean L1 per (slug,hour) over each ride's observed operating grid in
    [start, end) — stands in for the forecast_trip_plan_base rows."""
    grid = raw.loc[
        (raw["date"] >= start) & (raw["date"] < end), ["slug", "date", "hour"]
    ].drop_duplicates()
    if grid.empty:
        return pd.Series(dtype=float)
    feats = build_l1_features(grid.reset_index(drop=True), h_avg, ride_avg,
                              wx, ride_enc, park_enc, baseline_default=30)
    feats["l1"] = predict_l1(model, feats)
    return feats.groupby(["slug", "hour"])["l1"].mean()


def compute_layer2(raw, origin_day, l1_mean_30d):
    r = raw[raw["date"] < origin_day]
    g = r.groupby(["slug", "hour"]).agg(cnt=("cnt", "sum"), s=("s", "sum"))
    actual = g["s"] / g["cnt"]
    df = pd.concat([actual.rename("actual"), l1_mean_30d.rename("l1")], axis=1)
    df = df.dropna()
    ratio = (df["actual"] / df["l1"].clip(lower=5)).clip(0.3, 3.0)
    return dict(zip(df.index, ratio))


def compute_layer3(raw, origin_day, get_l2, l1_mean_5d):
    r = raw[
        (raw["date"] >= origin_day - pd.Timedelta(days=5))
        & (raw["date"] < origin_day)
    ]
    g = r.groupby(["slug", "hour"]).agg(cnt=("cnt", "sum"), s=("s", "sum"))
    actual = g["s"] / g["cnt"]
    adjustments = {}
    for (slug, hour), act in actual.items():
        l1 = l1_mean_5d.get((slug, hour))
        if l1 is None or l1 < 5:
            continue
        expected = l1 * get_l2(slug, hour)
        if expected >= 5:
            adjustments[(slug, hour)] = max(0.3, min(3.0, act / expected))
    return adjustments

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=None, help="limit origin-DAYS")
    args = ap.parse_args()

    raw, v2, weather = load_inputs()
    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    hourly = hourly[hourly["resort"] == "Orlando"].copy()
    daily_bench = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily_bench["date"] = pd.to_datetime(daily_bench["date"])
    daily_actuals = daily_bench[
        (daily_bench["source"] == "own_5min")
        & (daily_bench["regime"].isin(["A", "B"]))
    ]

    origin_days = sorted({o.normalize() for o in harness.rolling_origins(hourly)})
    if args.origins:
        origin_days = origin_days[-args.origins:]
    bench_slugs = set(hourly["series_id"].unique())
    print(
        f"Orlando benchmark: {len(hourly):,} ride-hours, {len(bench_slugs)} series | "
        f"{len(origin_days)} origin-days ({origin_days[0].date()} .. "
        f"{origin_days[-1].date()}) | v2 slugs in bench: "
        f"{len(bench_slugs & set(v2['slug'].unique()))}"
    )

    hourly_scored = {"prod_l1": [], "prod_full": []}
    daily_scored = {"prod_l1": [], "prod_full": []}
    h_scale_frames, d_scale_frames, fc_frames = [], [], []

    for i, D in enumerate(origin_days):
        wx = monthly_weather(weather, D)
        h_avg, ride_avg = hourly_patterns(raw, D)
        model, ride_enc, park_enc = train_l1(v2, h_avg, ride_avg, wx, D)

        l1_30 = _l1_hour_means(raw, model, h_avg, ride_avg, wx, ride_enc,
                               park_enc, D - pd.Timedelta(days=30), D)
        l1_5 = _l1_hour_means(raw, model, h_avg, ride_avg, wx, ride_enc,
                              park_enc, D - pd.Timedelta(days=5), D)
        get_l2 = _closest_hour_getter(compute_layer2(raw, D, l1_30))
        get_l3 = _closest_hour_getter(
            compute_layer3(raw, D, get_l2, l1_5.to_dict())
        )

        # forecast grid: observed benchmark hours, restricted to slugs the
        # model knows (production only forecasts rides in the encoder)
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
        feats = build_l1_features(grid, h_avg, ride_avg, wx, ride_enc,
                                  park_enc, baseline_default=30)
        fc = pd.DataFrame({
            "series_id": span["series_id"].to_numpy(),
            "ts": span["ts"].to_numpy(),
            "date": grid["date"].to_numpy(),
            "hour": grid["hour"].to_numpy(),
            "l1": predict_l1(model, feats),
        })
        l2 = np.array([get_l2(s, h) for s, h in zip(fc["series_id"], fc["hour"])])
        l3 = np.array([get_l3(s, h) for s, h in zip(fc["series_id"], fc["hour"])])
        fc["full"] = np.clip(fc["l1"] * l2 * l3, 5, 300)
        fc_frames.append(fc.assign(origin_day=D))

        # hourly protocol: 3 intraday origins share the midnight forecast
        for h in ORIGIN_HOURS:
            origin = D + pd.Timedelta(hours=h)
            sub = fc[(fc["ts"] >= origin)
                     & (fc["ts"] < origin + pd.Timedelta(hours=MAX_HORIZON_HOURS))]
            for name, col in (("prod_l1", "l1"), ("prod_full", "full")):
                out = sub[["series_id", "ts"]].assign(origin=origin,
                                                      y_hat=sub[col].to_numpy())
                hourly_scored[name].append(harness.evaluate(out, hourly))
        h_scales = harness.mase_scales(hourly[hourly["ts"] < D])
        h_scales.index.name = "series_id"
        h_scale_frames.append(h_scales.rename("scale").to_frame().assign(origin=D))

        # daily protocol: average predictions over each day's observed hours
        dfc = fc.groupby(["series_id", "date"], as_index=False)[["l1", "full"]].mean()
        for name, col in (("prod_l1", "l1"), ("prod_full", "full")):
            out = dfc[["series_id", "date"]].assign(origin=D,
                                                    y_hat=dfc[col].to_numpy())
            daily_scored[name].append(run_daily.evaluate(out, daily_actuals))
        d_scales = run_daily.mase_scales(
            daily_actuals[daily_actuals["date"] < D]
        )
        d_scales.index.name = "series_id"
        d_scale_frames.append(d_scales.rename("scale").to_frame().assign(origin=D))

        print(f"  origin-day {i + 1}/{len(origin_days)} {D.date()} done",
              file=sys.stderr)

    pd.concat(fc_frames, ignore_index=True).to_parquet(
        DATA_DIR / "prod_backtest_fc.parquet"
    )

    h_scales = pd.concat(h_scale_frames).groupby("series_id")["scale"].mean()
    d_scales = pd.concat(d_scale_frames).groupby("series_id")["scale"].mean()

    for name in ("prod_l1", "prod_full"):
        scored = pd.concat(hourly_scored[name], ignore_index=True)
        print(f"\n=== {name} — hourly protocol, by horizon bucket ===")
        print(harness.summarize(scored, h_scales, by=("bucket",)).to_string())
        print(f"\n=== {name} — hourly protocol, by regime x bucket ===")
        print(harness.summarize(scored, h_scales, by=("regime", "bucket")).to_string())

    for name in ("prod_l1", "prod_full"):
        scored = pd.concat(daily_scored[name], ignore_index=True)
        print(f"\n=== {name} — daily protocol, by lead bucket ===")
        print(run_daily.summarize(scored, d_scales, by=("bucket",)).to_string())
        print(f"\n=== {name} — daily protocol, by regime x lead bucket ===")
        print(run_daily.summarize(scored, d_scales, by=("regime", "bucket")).to_string())


if __name__ == "__main__":
    main()
