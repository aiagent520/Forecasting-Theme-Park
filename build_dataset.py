#!/usr/bin/env python3
"""Build uniform benchmark datasets from raw wait_times (+ queue-times daily).

Outputs (parquet, in benchmark/data/):
  hourly.parquet: series_id, park, resort, regime, ts (hourly local), y, n_obs
      — own 5-min data only (>= OWN_DATA_START), uniform aggregation rule.
  daily.parquet:  series_id, park, resort, regime, date, y, source
      — source='own_5min' for >= OWN_DATA_START (recomputed from hourly, NOT
        the spliced production table); source='queue_times' for earlier dates
        (segment 1 of ride_daily_history_v2: integer daily averages).

Usage: python3 build_dataset.py
"""

import sqlite3

import pandas as pd

from config import (
    DATA_DIR,
    DB_PATH,
    MIN_HOURS_PER_DAY,
    MIN_OBS_PER_HOUR,
    MIN_SERIES_DAYS,
    OWN_DATA_START,
    park_info,
)


def build_hourly(conn) -> pd.DataFrame:
    print("Building hourly series from wait_times (5-min STANDBY readings)...")
    df = pd.read_sql(
        """
        SELECT attraction_id AS series_id,
               strftime('%Y-%m-%d %H:00:00', local_timestamp) AS ts,
               AVG(wait_time) AS y,
               COUNT(wait_time) AS n_obs
        FROM wait_times
        WHERE queue_type = 'STANDBY'
          AND wait_time IS NOT NULL
          AND date(local_timestamp) >= ?
          AND date(local_timestamp) < date('now', 'localtime')
        GROUP BY series_id, ts
        """,
        conn,
        params=(OWN_DATA_START,),
    )
    print(f"  {len(df):,} raw ride-hours")

    df = df[df["n_obs"] >= MIN_OBS_PER_HOUR].copy()
    info = df["series_id"].map(lambda s: park_info(s))
    df = df[info.notna()].copy()
    meta = pd.DataFrame(list(df["series_id"].map(park_info)), index=df.index)
    df = pd.concat([df, meta], axis=1)
    df["ts"] = pd.to_datetime(df["ts"])

    # Curation: series need >= MIN_SERIES_DAYS distinct days of hourly data.
    days_per_series = df.groupby("series_id")["ts"].apply(
        lambda s: s.dt.normalize().nunique()
    )
    dropped = days_per_series[days_per_series < MIN_SERIES_DAYS]
    if len(dropped):
        print(f"  dropped {len(dropped)} series with < {MIN_SERIES_DAYS} days:")
        for sid, d in dropped.sort_values().items():
            print(f"    {sid} ({d}d)")
    df = df[~df["series_id"].isin(dropped.index)].copy()

    print(
        f"  kept {len(df):,} ride-hours "
        f"({df['series_id'].nunique()} series, "
        f"{df['ts'].dt.date.nunique()} days) after filters"
    )
    return df[["series_id", "park", "resort", "regime", "ts", "y", "n_obs"]]


def build_daily(conn, hourly: pd.DataFrame) -> pd.DataFrame:
    print("Building daily series...")

    # Own-pipeline era: recompute from the hourly table (mean of hourly means,
    # so long closed stretches don't need special-casing).
    own = hourly.copy()
    own["date"] = own["ts"].dt.strftime("%Y-%m-%d")
    grp = own.groupby(["series_id", "park", "resort", "regime", "date"])
    daily_own = grp.agg(y=("y", "mean"), n_hours=("y", "size")).reset_index()
    daily_own = daily_own[daily_own["n_hours"] >= MIN_HOURS_PER_DAY].drop(
        columns="n_hours"
    )
    daily_own["source"] = "own_5min"
    print(f"  own_5min: {len(daily_own):,} ride-days")

    # Queue-times era (segment 1 only — dates strictly before OWN_DATA_START).
    qt = pd.read_sql(
        """
        SELECT slug AS series_id, date, avg_wait AS y
        FROM ride_daily_history_v2
        WHERE date < ? AND avg_wait > 0
        """,
        conn,
        params=(OWN_DATA_START,),
    )
    info = qt["series_id"].map(lambda s: park_info(s))
    qt = qt[info.notna()].copy()
    meta = pd.DataFrame(list(qt["series_id"].map(park_info)), index=qt.index)
    qt = pd.concat([qt, meta], axis=1)
    qt["source"] = "queue_times"
    print(f"  queue_times: {len(qt):,} ride-days")

    daily = pd.concat(
        [
            qt[["series_id", "park", "resort", "regime", "date", "y", "source"]],
            daily_own[
                ["series_id", "park", "resort", "regime", "date", "y", "source"]
            ],
        ],
        ignore_index=True,
    ).sort_values(["series_id", "date"])
    return daily


def main():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    hourly = build_hourly(conn)
    hourly.to_parquet(DATA_DIR / "hourly.parquet", index=False)

    daily = build_daily(conn, hourly)
    daily.to_parquet(DATA_DIR / "daily.parquet", index=False)
    conn.close()

    print("\nSummary by park:")
    hsum = hourly.groupby("park").agg(
        series=("series_id", "nunique"),
        hours=("y", "size"),
        first=("ts", "min"),
        last=("ts", "max"),
    )
    print(hsum.to_string())
    print("\nDaily by park and source:")
    dsum = daily.groupby(["park", "source"]).agg(
        series=("series_id", "nunique"),
        days=("y", "size"),
        first=("date", "min"),
        last=("date", "max"),
    )
    print(dsum.to_string())
    print(f"\nWrote {DATA_DIR / 'hourly.parquet'} and {DATA_DIR / 'daily.parquet'}")


if __name__ == "__main__":
    main()
