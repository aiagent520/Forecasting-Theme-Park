"""Evaluation harness: rolling-origin splits, metrics, horizon buckets.

Forecast interface (all methods must produce this):
    DataFrame with columns [series_id, origin, ts, y_hat]
      origin = the forecast-origin timestamp (last observation <= origin is
               visible to the model); ts = target hour being forecast.

Evaluation joins forecasts to actuals on (series_id, ts), assigns each row a
horizon bucket from (ts - origin), and reports MAE / RMSE / MASE.

MASE scaling: per (series_id, origin), the in-sample MAE of the seasonal
naive (lag = SEASONAL_PERIOD_HOURS) over the training window. Scale-free, so
metrics aggregate across rides with different wait magnitudes.
"""

import numpy as np
import pandas as pd

from config import (
    HORIZON_BUCKETS,
    MAX_HORIZON_HOURS,
    MIN_TRAIN_DAYS,
    ORIGIN_HOURS,
    ORIGIN_SPACING_DAYS,
    SEASONAL_PERIOD_HOURS,
)

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def rolling_origins(hourly: pd.DataFrame) -> list[pd.Timestamp]:
    """Weekly origin-days x ORIGIN_HOURS intraday origins over the window."""
    start = hourly["ts"].min().normalize() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    end = hourly["ts"].max().normalize() - pd.Timedelta(hours=MAX_HORIZON_HOURS)
    origins = []
    t = start
    while t <= end:
        for h in ORIGIN_HOURS:
            origins.append(t + pd.Timedelta(hours=h))
        t += pd.Timedelta(days=ORIGIN_SPACING_DAYS)
    return origins


def train_test_split(hourly: pd.DataFrame, origin: pd.Timestamp):
    """Data visible at the origin, and the target window after it."""
    train = hourly[hourly["ts"] < origin]
    horizon_end = origin + pd.Timedelta(hours=MAX_HORIZON_HOURS)
    test = hourly[(hourly["ts"] >= origin) & (hourly["ts"] < horizon_end)]
    return train, test

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _bucket(hours_ahead: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=hours_ahead.index, dtype="object")
    for label, (lo, hi) in HORIZON_BUCKETS.items():
        out[(hours_ahead >= lo) & (hours_ahead <= hi)] = label
    return out


def mase_scales(train: pd.DataFrame) -> pd.Series:
    """Per-series in-sample seasonal-naive MAE (the MASE denominator)."""
    scales = {}
    for sid, g in train.groupby("series_id"):
        s = g.set_index("ts")["y"].sort_index()
        lagged = s.copy()
        lagged.index = lagged.index + pd.Timedelta(hours=SEASONAL_PERIOD_HOURS)
        diff = (s - lagged.reindex(s.index)).abs().dropna()
        if len(diff) >= 24:  # need a meaningful sample
            scale = diff.mean()
            if scale > 0:
                scales[sid] = scale
    return pd.Series(scales, name="scale")


def evaluate(forecasts: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Score forecasts against actuals. Returns per-row scored frame."""
    actuals = hourly[["series_id", "park", "regime", "ts", "y"]]
    df = forecasts.merge(actuals, on=["series_id", "ts"], how="inner")
    df["hours_ahead"] = (
        (df["ts"] - df["origin"]).dt.total_seconds() / 3600
    ).astype(int) + 1
    df["origin_hour"] = df["origin"].dt.hour
    df["bucket"] = _bucket(df["hours_ahead"])
    df = df.dropna(subset=["bucket"])
    df["abs_err"] = (df["y"] - df["y_hat"]).abs()
    df["sq_err"] = (df["y"] - df["y_hat"]) ** 2
    return df


def summarize(scored: pd.DataFrame, scales: pd.Series, by=("bucket",)) -> pd.DataFrame:
    """Aggregate scored rows -> MAE / RMSE / MASE table."""
    df = scored.join(scales, on="series_id")
    df = df[df["scale"].notna()].copy()
    df["ase"] = df["abs_err"] / df["scale"]  # abs scaled error
    out = df.groupby(list(by)).agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        rmse=("sq_err", lambda s: float(np.sqrt(s.mean()))),
        mase=("ase", "mean"),
        # decision-relevant tolerance hit rates (share of forecasts within
        # +/- 5 / 10 actual minutes; absolute, not %, because error cost is
        # guest-minutes and MAPE breaks on walk-on rides near 0)
        w5=("abs_err", lambda s: (s <= 5).mean()),
        w10=("abs_err", lambda s: (s <= 10).mean()),
    )
    # order horizon buckets naturally if present
    if "bucket" in out.index.names:
        order = [b for b in HORIZON_BUCKETS if b in out.index.get_level_values("bucket")]
        out = out.reindex(order, level="bucket")
    return out.round(2)
