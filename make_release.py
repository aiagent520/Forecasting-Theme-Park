"""Build the public dataset release bundle (Hugging Face-ready).

Exports the Tier-1 assets (everything we own outright) into ../release/:

    release/
      README.md                 Hugging Face dataset card (YAML front matter)
      DATASHEET.md              extracted from PAPER_DRAFT.md Appendix A
      LICENSE                   CC BY 4.0 (data) + attributions
      CITATION.cff
      data/
        qt_ride_mapping.csv     queue-times ride id -> series_id crosswalk
                                (provenance; no third-party values)
        raw_5min.parquet        all raw readings, all queue types, all parks
        hourly.parquet/.csv.gz  curated benchmark ground truth (80 series)
        daily.parquet/.csv.gz   daily means (coarse rows stripped by default)
        series_metadata.csv     park/resort/regime/coverage per series
        level_ratios.csv        per-series own/coarse boundary-window ratios
        covariates/
          school_calendar.csv   district-weighted break/holiday/event scores
          weather_daily.csv     Open-Meteo daily weather (Orlando)
        protocol/
          origins_hourly.csv    the 60 rolling-origin timestamps
          origins_daily.csv     the 20 daily-protocol origin dates
        forecasts/*.parquet     saved per-forecast outputs of every experiment
        production/
          stored_forecasts.parquet       82 daily runs + frozen 05-31 snapshot
          stored_forecasts_l1base.parquet  Layer-1-only snapshot (2026-03-22)

Tier 2 (the third-party coarse archive values) is EXCLUDED: Queue-Times'
terms require attribution but don't grant redistribution, and as of
2026-09 their historical calendar pages sit behind bot protection, so a
user-side re-fetch script is not viable either. The bundle ships our
derived per-series level ratios, coarse coverage counts, and the id
crosswalk (provenance) instead. (--include-coarse still exists should
redistribution permission ever be granted.)

Usage:
    python3 make_release.py                # Tier-1 bundle (coarse stripped)
    python3 make_release.py --include-coarse
"""

import argparse
import gzip
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from config import BENCH_DIR, DATA_DIR, DB_PATH, OWN_DATA_START, park_info
from harness import rolling_origins as hourly_origins
from run_daily import rolling_origins as daily_origins, _level_ratio

RELEASE_DIR = BENCH_DIR.parent / "release"
PAPER = BENCH_DIR.parent / "PAPER_DRAFT.md"

FC_PARQUETS = [
    "q2_stage1_fc.parquet",
    "q2_stage2_fc.parquet",
    "q2_shape_layer_fc.parquet",
    "q3_curve_fc.parquet",
    "q4_coldstart_fc.parquet",
    "q4_chronos_fc.parquet",
    "q5_models_fc.parquet",
    "q6_covariates_fc.parquet",
    "quantile_gbm_fc.parquet",
    "prod_backtest_fc.parquet",
]


def csv_gz(df: pd.DataFrame, path: Path) -> None:
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)


def export_raw(out: Path) -> int:
    con = sqlite3.connect(DB_PATH)
    raw = pd.read_sql(
        """SELECT local_timestamp, attraction_id AS series_id, queue_type,
                  status, wait_time
           FROM wait_times ORDER BY local_timestamp""",
        con,
    )
    con.close()
    raw.to_parquet(out / "raw_5min.parquet", index=False)
    return len(raw)


def export_ground_truth(out: Path, include_coarse: bool) -> pd.DataFrame:
    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    hourly.to_parquet(out / "hourly.parquet", index=False)
    csv_gz(hourly, out / "hourly.csv.gz")

    daily_full = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily = daily_full if include_coarse else daily_full[daily_full["source"] == "own_5min"]
    daily.to_parquet(out / "daily.parquet", index=False)
    csv_gz(daily, out / "daily.csv.gz")

    # series metadata (coverage facts are always reported, even when the
    # coarse VALUES are stripped)
    meta = (
        hourly.groupby(["series_id", "park", "resort", "regime"])
        .agg(
            n_hours=("y", "size"),
            n_days=("ts", lambda s: s.dt.normalize().nunique()),
            mean_wait=("y", "mean"),
            first_ts=("ts", "min"),
            last_ts=("ts", "max"),
        )
        .round({"mean_wait": 2})
        .reset_index()
    )
    coarse_days = (
        daily_full[daily_full["source"] == "queue_times"]
        .groupby("series_id")["date"]
        .nunique()
        .rename("coarse_days")
    )
    meta = meta.merge(coarse_days, on="series_id", how="left")
    meta["coarse_days"] = meta["coarse_days"].fillna(0).astype(int)
    meta.to_csv(out / "series_metadata.csv", index=False)

    # per-series own/coarse level ratios (derived, ours). Static full-frame
    # version of the boundary-window rule the runners recompute per origin.
    ratio_frame = daily_full.assign(date=pd.to_datetime(daily_full["date"]))
    ratios = pd.Series(_level_ratio(ratio_frame), name="own_over_coarse_ratio")
    ratios.round(4).rename_axis("series_id").reset_index().to_csv(
        out / "level_ratios.csv", index=False
    )
    return daily_full


def export_covariates(out: Path) -> None:
    cov = out / "covariates"
    cov.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    pd.read_sql("SELECT * FROM crowd_calendar ORDER BY date", con).to_csv(
        cov / "school_calendar.csv", index=False
    )
    pd.read_sql("SELECT * FROM weather_daily ORDER BY date", con).to_csv(
        cov / "weather_daily.csv", index=False
    )
    con.close()


def export_protocol(out: Path, daily_full: pd.DataFrame) -> None:
    proto = out / "protocol"
    proto.mkdir(exist_ok=True)
    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    pd.DataFrame({"origin": hourly_origins(hourly)}).to_csv(
        proto / "origins_hourly.csv", index=False
    )
    daily_parsed = daily_full.assign(date=pd.to_datetime(daily_full["date"]))
    pd.DataFrame({"origin": daily_origins(daily_parsed)}).to_csv(
        proto / "origins_daily.csv", index=False
    )


def export_forecasts(out: Path) -> None:
    fdir = out / "forecasts"
    fdir.mkdir(exist_ok=True)
    for name in FC_PARQUETS:
        shutil.copy2(DATA_DIR / name, fdir / name)


def export_production(out: Path) -> None:
    pdir = out / "production"
    pdir.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    pd.read_sql(
        """SELECT generated_at, target_date, target_hour, ride_slug,
                  predicted_wait, model_version
           FROM forecast_trip_plan""",
        con,
    ).to_parquet(pdir / "stored_forecasts.parquet", index=False)
    pd.read_sql(
        """SELECT generated_at, target_date, target_hour, ride_slug,
                  layer1_prediction
           FROM forecast_trip_plan_base""",
        con,
    ).to_parquet(pdir / "stored_forecasts_l1base.parquet", index=False)
    con.close()


def export_qt_mapping(out: Path) -> int:
    """Crosswalk queue-times numeric ride ids -> series_ids (provenance).

    Built by exact (park_id, ride_name) join between the original scrape
    table (numeric ids) and the migrated slug table, plus 5 manual pairs
    where the display name changed (tm-mark placement, renames). Shipped
    so the archive's provenance is auditable and so anyone who obtains
    the archive independently can map it without repeating the
    fuzzy-name-matching mistake documented in the datasheet. Contains
    ids only — no Queue-Times data values.
    """
    con = sqlite3.connect(DB_PATH)
    pairs = pd.read_sql(
        """SELECT DISTINCT v1.park_id AS qt_park_id, v1.ride_id AS qt_ride_id,
                  v2.slug AS series_id
           FROM (SELECT DISTINCT park_id, ride_id, ride_name
                 FROM ride_daily_history) v1
           JOIN (SELECT DISTINCT park_id, ride_name, slug
                 FROM ride_daily_history_v2 WHERE date < ?) v2
             ON v1.park_id = v2.park_id AND v1.ride_name = v2.ride_name""",
        con,
        params=(OWN_DATA_START,),
    )
    con.close()
    manual = pd.DataFrame(
        [
            (64, "5994", "uor.ioa.rides.jurassic_park_river_adventure"),
            (64, "5999", "uor.ioa.rides.pteranodon_flyers"),
            (64, "6015", "uor.ioa.rides.hogwarts_express_-_hogsmeade_station"),
            (65, "6016", "uor.usf.rides.hogwarts_express_-_kings_cross_station"),
            (334, "14695", "uor.ueu.rides.hiccups_wing_glider"),
        ],
        columns=["qt_park_id", "qt_ride_id", "series_id"],
    )
    pairs = pd.concat([pairs, manual], ignore_index=True)
    info = pairs["series_id"].map(park_info)
    pairs = pairs[info.notna()].copy()  # same curation filter as build_dataset
    meta = pd.DataFrame(list(pairs["series_id"].map(park_info)), index=pairs.index)
    pairs = pd.concat([pairs, meta], axis=1).sort_values(
        ["qt_park_id", "series_id"]
    )
    assert pairs["qt_ride_id"].is_unique and pairs["series_id"].is_unique
    pairs.to_csv(out / "qt_ride_mapping.csv", index=False)
    return len(pairs)


def export_datasheet() -> None:
    text = PAPER.read_text()
    start = text.index("# Appendix A — Datasheet for the dataset")
    end = text.index("# Appendix B")
    body = text[start:end].rstrip().rstrip("-").rstrip()
    body = body.replace(
        "# Appendix A — Datasheet for the dataset", "# Datasheet", 1
    )
    (RELEASE_DIR / "DATASHEET.md").write_text(body + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-coarse", action="store_true")
    args = ap.parse_args()

    out = RELEASE_DIR / "data"
    out.mkdir(parents=True, exist_ok=True)

    n_raw = export_raw(out)
    daily_full = export_ground_truth(out, args.include_coarse)
    export_covariates(out)
    export_protocol(out, daily_full)
    export_forecasts(out)
    export_production(out)
    n_map = export_qt_mapping(out)
    export_datasheet()

    coarse_state = (
        "INCLUDED"
        if args.include_coarse
        else f"stripped (ratios + {n_map}-ride id crosswalk shipped instead)"
    )
    print(f"raw readings exported: {n_raw:,}")
    print(f"coarse archive rows:   {coarse_state}")
    total = sum(f.stat().st_size for f in RELEASE_DIR.rglob("*") if f.is_file())
    print(f"bundle size:           {total / 1e6:.1f} MB at {RELEASE_DIR}")
    print("note: README.md / LICENSE / CITATION.cff are maintained by hand.")


if __name__ == "__main__":
    main()
