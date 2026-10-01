#!/usr/bin/env python3
"""Daily-grain long-horizon benchmark (Tier-1, pandas methods).

The heterogeneity question at its cheapest level: does long COARSE history
(queue-times era) improve long-lead daily forecasts over fine-era data alone?

Methods (pure pandas):
  snaive7       — same weekday last week (fallback 2/3/4 weeks back)
  dow_mean_8w   — day-of-week mean over the trailing 8 weeks
  dow_mean_full — day-of-week mean over FULL history (spliced coarse+fine
                  where it exists: regime A ~2.5yr, Epic ~10mo, C fine-only)
                  -> direct A/B against dow_mean_8w on the value of history
  snaive364     — same date last year (lag 364 keeps the weekday); only
                  possible where long history exists; NO fallback, so its
                  n per group shows coverage

Protocol: weekly origin dates (same cadence as hourly benchmark), horizon
MAX_HORIZON_DAYS, buckets in DAILY_HORIZON_BUCKETS. Targets = own-era daily
means only. MASE scale: per-series in-sample lag-7 MAE over the training
window.

Usage: python3 run_daily.py
"""

import sys

import numpy as np
import pandas as pd

from config import (
    DAILY_HORIZON_BUCKETS,
    DATA_DIR,
    MAX_HORIZON_DAYS,
    MIN_TRAIN_DAYS,
    ORIGIN_SPACING_DAYS,
    SEASONAL_PERIOD_DAYS,
)

# ---------------------------------------------------------------------------
# Protocol pieces (daily-grain analogues of harness.py)
# ---------------------------------------------------------------------------

def rolling_origins(daily: pd.DataFrame) -> list[pd.Timestamp]:
    own = daily[daily["source"] == "own_5min"]
    start = own["date"].min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    end = own["date"].max() - pd.Timedelta(days=7)  # >= first bucket evaluable
    origins = []
    t = start
    while t <= end:
        origins.append(t)
        t += pd.Timedelta(days=ORIGIN_SPACING_DAYS)
    return origins


def _bucket(days_ahead: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=days_ahead.index, dtype="object")
    for label, (lo, hi) in DAILY_HORIZON_BUCKETS.items():
        out[(days_ahead >= lo) & (days_ahead <= hi)] = label
    return out


def mase_scales(train: pd.DataFrame) -> pd.Series:
    scales = {}
    for sid, g in train.groupby("series_id"):
        s = g.set_index("date")["y"].sort_index()
        lagged = s.copy()
        lagged.index = lagged.index + pd.Timedelta(days=SEASONAL_PERIOD_DAYS)
        diff = (s - lagged.reindex(s.index)).abs().dropna()
        if len(diff) >= 28:
            scale = diff.mean()
            if scale > 0:
                scales[sid] = scale
    return pd.Series(scales, name="scale")


def evaluate(fc: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    df = fc.merge(
        actuals[["series_id", "park", "regime", "date", "y"]],
        on=["series_id", "date"],
        how="inner",
    )
    df["days_ahead"] = (df["date"] - df["origin"]).dt.days + 1
    df["bucket"] = _bucket(df["days_ahead"])
    df = df.dropna(subset=["bucket"])
    df["abs_err"] = (df["y"] - df["y_hat"]).abs()
    df["sq_err"] = (df["y"] - df["y_hat"]) ** 2
    return df


def summarize(scored: pd.DataFrame, scales: pd.Series, by=("bucket",)) -> pd.DataFrame:
    df = scored.join(scales, on="series_id")
    df = df[df["scale"].notna()].copy()
    df["ase"] = df["abs_err"] / df["scale"]
    out = df.groupby(list(by)).agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        rmse=("sq_err", lambda s: float(np.sqrt(s.mean()))),
        mase=("ase", "mean"),
        w5=("abs_err", lambda s: (s <= 5).mean()),
        w10=("abs_err", lambda s: (s <= 10).mean()),
    )
    if "bucket" in out.index.names:
        order = [
            b
            for b in DAILY_HORIZON_BUCKETS
            if b in out.index.get_level_values("bucket")
        ]
        out = out.reindex(order, level="bucket")
    return out.round(2)

# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def forecast_snaive7(train, grid, origin):
    fc = grid.copy()
    lookup = train.set_index(["series_id", "date"])["y"]
    y_hat = pd.Series(index=fc.index, dtype=float)
    for weeks_back in (1, 2, 3, 4):
        need = y_hat.isna()
        if not need.any():
            break
        src = fc.loc[need, "date"] - pd.Timedelta(days=7 * weeks_back)
        keys = list(zip(fc.loc[need, "series_id"], src))
        y_hat.loc[need] = lookup.reindex(keys).to_numpy()
    fc["y_hat"] = y_hat
    return fc.dropna(subset=["y_hat"])


def _dow_mean(train, grid):
    tr = train.copy()
    tr["dow"] = tr["date"].dt.dayofweek
    avg = tr.groupby(["series_id", "dow"])["y"].mean().rename("y_hat")
    fc = grid.copy()
    fc["dow"] = fc["date"].dt.dayofweek
    fc = fc.join(avg, on=["series_id", "dow"])
    return fc.drop(columns=["dow"]).dropna(subset=["y_hat"])


def forecast_dow_mean_8w(train, grid, origin):
    recent = train[train["date"] >= origin - pd.Timedelta(days=56)]
    return _dow_mean(recent, grid)


def forecast_dow_mean_full(train, grid, origin):
    return _dow_mean(train, grid)


def forecast_snaive364(train, grid, origin):
    fc = grid.copy()
    lookup = train.set_index(["series_id", "date"])["y"]
    src = fc["date"] - pd.Timedelta(days=364)
    keys = list(zip(fc["series_id"], src))
    fc["y_hat"] = lookup.reindex(keys).to_numpy()
    return fc.dropna(subset=["y_hat"])


# --- calibrated coarse-history methods -------------------------------------
# The coarse (queue-times) era is level-incompatible with the own era
# (~2.2-2.6x per ride; see decisions doc) and there is NO overlap window.
# These methods use coarse history for SHAPE and fine data for LEVEL.

def _level_ratio(train, window_days=56):
    """Per-series own/coarse level ratio from boundary windows (all < origin):
    first `window_days` of the own era vs last `window_days` of the coarse
    era. Seasonally confounded (Feb vs Mar) but strictly hindsight-free."""
    qt = train[train["source"] == "queue_times"]
    own = train[train["source"] == "own_5min"]
    ratios = {}
    for sid, g in own.groupby("series_id"):
        q = qt[qt["series_id"] == sid]
        if q.empty:
            continue
        own_w = g[g["date"] < g["date"].min() + pd.Timedelta(days=window_days)]
        qt_w = q[q["date"] >= q["date"].max() - pd.Timedelta(days=window_days)]
        if len(own_w) >= 14 and len(qt_w) >= 14 and qt_w["y"].mean() > 0:
            ratios[sid] = own_w["y"].mean() / qt_w["y"].mean()
    return pd.Series(ratios, name="ratio")


def forecast_snaive364_cal(train, grid, origin):
    fc = forecast_snaive364(train, grid, origin)
    ratio = _level_ratio(train)
    fc = fc.join(ratio, on="series_id")
    fc = fc.dropna(subset=["ratio"])
    fc["y_hat"] = fc["y_hat"] * fc["ratio"]
    return fc.drop(columns=["ratio"])


def forecast_dow_shape_annual(train, grid, origin):
    """dow_mean_8w level x annual month-shape factor from the coarse era.

    factor(sid, month) = coarse mean in that calendar month / coarse mean in
    the months spanned by the trailing 8-week level window, so the factor is
    1 when the target month 'looks like' the recent window.
    """
    qt = train[train["source"] == "queue_times"]
    if qt.empty:
        return grid.iloc[0:0].assign(y_hat=pd.Series(dtype=float))
    qt = qt.copy()
    qt["month"] = qt["date"].dt.month
    month_mean = qt.groupby(["series_id", "month"])["y"].mean().rename("m_target")

    # coarse mean over the months covered by the trailing 8w window,
    # weighted by how many of the window's days fall in each month
    window = pd.date_range(
        origin - pd.Timedelta(days=56), origin - pd.Timedelta(days=1), freq="D"
    )
    month_weights = window.month.value_counts(normalize=True)
    base = (
        month_mean.reset_index()
        .assign(w=lambda d: d["month"].map(month_weights).fillna(0.0))
        .groupby("series_id")
        .apply(
            lambda d: (d["m_target"] * d["w"]).sum() / d["w"].sum()
            if d["w"].sum() > 0
            else np.nan,
            include_groups=False,
        )
        .rename("m_base")
    )

    fc = _dow_mean(train[train["date"] >= origin - pd.Timedelta(days=56)], grid)
    fc["month"] = fc["date"].dt.month
    fc = fc.join(month_mean, on=["series_id", "month"])
    fc = fc.join(base, on="series_id")
    fc = fc.dropna(subset=["m_target", "m_base"])
    fc = fc[fc["m_base"] > 0]
    fc["y_hat"] = fc["y_hat"] * fc["m_target"] / fc["m_base"]
    return fc.drop(columns=["month", "m_target", "m_base"])


METHODS = {
    "snaive7": forecast_snaive7,
    "dow_mean_8w": forecast_dow_mean_8w,
    "dow_mean_full": forecast_dow_mean_full,
    "snaive364": forecast_snaive364,
    "snaive364_cal": forecast_snaive364_cal,
    "dow_shape_annual": forecast_dow_shape_annual,
}

# ---------------------------------------------------------------------------

def main():
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    actuals = daily[daily["source"] == "own_5min"]
    origins = rolling_origins(daily)
    print(
        f"Daily data: {len(daily):,} rows ({len(actuals):,} own-era), "
        f"{daily['series_id'].nunique()} series | {len(origins)} origins "
        f"({origins[0].date()} .. {origins[-1].date()}), "
        f"horizon {MAX_HORIZON_DAYS}d"
    )

    all_scored = {name: [] for name in METHODS}
    scale_frames = []
    for i, origin in enumerate(origins):
        train = daily[daily["date"] < origin]
        horizon_end = origin + pd.Timedelta(days=MAX_HORIZON_DAYS)
        test = actuals[(actuals["date"] >= origin) & (actuals["date"] < horizon_end)]
        grid = test[["series_id", "date"]].copy()
        grid["origin"] = origin
        scales = mase_scales(train)
        scales.index.name = "series_id"
        scale_frames.append(scales.rename("scale").to_frame().assign(origin=origin))
        for name, fn in METHODS.items():
            fc = fn(train, grid, origin)
            all_scored[name].append(evaluate(fc, actuals))
        print(f"  origin {i + 1}/{len(origins)} {origin.date()} done", file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()

    for name in METHODS:
        scored = pd.concat(all_scored[name], ignore_index=True)
        print(f"\n=== {name} — by lead bucket ===")
        print(summarize(scored, scales, by=("bucket",)).to_string())
        print(f"\n=== {name} — by regime x lead bucket ===")
        print(summarize(scored, scales, by=("regime", "bucket")).to_string())


if __name__ == "__main__":
    main()
