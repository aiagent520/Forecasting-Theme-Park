"""Shared configuration for the forecasting benchmark.

See ../DATA_STATUS_AND_DECISIONS.md for the rationale behind every rule here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).parent
DATA_DIR = BENCH_DIR / "data"
DB_PATH = Path(
    "/Users/shuaizhang/themepark/themeparkdata/data/db/universal_wait_times.db"
)

# ---------------------------------------------------------------------------
# Series universe: park mapping + data-availability regimes
#   Regime A (rich):       long coarse history + short fine stream (IOA, USF)
#   Regime B (cold-start): park opened 2025-05 (Epic)
#   Regime C (fine-only):  no coarse history at all (USJ, USH)
# Volcano Bay excluded (water park). Slugs must contain ".rides.".
# ---------------------------------------------------------------------------
PARKS = {
    "uor.ioa.": {"park": "IOA", "resort": "Orlando", "regime": "A"},
    "uor.usf.": {"park": "USF", "resort": "Orlando", "regime": "A"},
    "uor.ueu.": {"park": "Epic", "resort": "Orlando", "regime": "B"},
    "usj.": {"park": "USJ", "resort": "Japan", "regime": "C"},
    "ush.": {"park": "USH", "resort": "Hollywood", "regime": "C"},
}

def park_info(slug: str):
    """Return {park, resort, regime} for a slug, or None if out of scope."""
    if ".rides." not in slug:
        return None
    if is_excluded_series(slug):
        return None
    for prefix, info in PARKS.items():
        if slug.startswith(prefix):
            return info
    return None

# ---------------------------------------------------------------------------
# Series curation (rule-based, no hand-picking):
#   1. Pattern exclusions — entries under ".rides." that are not rides:
#      Halloween Horror Nights mazes/terror tram, HHN express-pass queues
#      (USH slugs ending "_express"; legit Hogwarts Express rides end in
#      "_station"), and first/last train/tram notification pseudo-series.
#   2. MIN_SERIES_DAYS — a series must have >= this many distinct days of
#      hourly data to be a forecast target (mirrors MIN_TRAIN_DAYS). Kills
#      short seasonal overlays / just-opened rides; long-running overlays
#      operating under a renamed slug (e.g. usj space_fantasy zedd remix,
#      165 days) survive because they are real sustained ride operations.
# Shows with genuine standby queues (ollivanders, conan_4d_live_show) are
# retained: they have real waits and production forecasts them.
# ---------------------------------------------------------------------------
EXCLUDE_SUBSTRINGS = (
    "hhn_",             # HHN mazes / terror tram (uor + ush 2026 naming)
    "haunted_house",
    "first_train",      # notification pseudo-series, not queues
    "last_train",
    "last_tram",
)
MIN_SERIES_DAYS = 56    # distinct days of hourly data required per series

def is_excluded_series(slug: str) -> bool:
    return any(p in slug for p in EXCLUDE_SUBSTRINGS) or slug.endswith("_express")

# ---------------------------------------------------------------------------
# Ground-truth aggregation rules (uniform for ALL methods)
#
# Raw readings: 5-min STANDBY rows. Non-operating statuses carry NULL
# wait_time (verified 2026-09-24), so "wait_time IS NOT NULL" == operating.
# Zeros (walk-ons) are INCLUDED — unlike the production ride_daily_history_v2
# segment-3 rule (AVG of wait>0), because ground truth should reflect actual
# experienced waits. The production table is NOT used as fine-grain ground
# truth (3-segment splice; see decisions doc §1b).
# ---------------------------------------------------------------------------
OWN_DATA_START = "2026-03-06"   # first day of own 5-min pipeline
MIN_OBS_PER_HOUR = 3            # of ~12 possible 5-min readings
MIN_HOURS_PER_DAY = 4           # daily aggregate needs at least this many hours

# ---------------------------------------------------------------------------
# Rolling-origin evaluation protocol (hourly benchmark)
# ---------------------------------------------------------------------------
ORIGIN_SPACING_DAYS = 7         # one forecast origin-day per week
ORIGIN_HOURS = (0, 9, 13)       # origins per origin-day (local time):
                                #   00:00 = overnight/long-lead view
                                #   09:00 = just after park opening
                                #   13:00 = midday (recency methods see the
                                #           morning) — fixes the thin 1h
                                #           bucket from midnight-only origins
MAX_HORIZON_HOURS = 7 * 24      # forecast 7 days ahead from each origin
MIN_TRAIN_DAYS = 56             # first origin needs >= 8 weeks of history
HORIZON_BUCKETS = {             # hours-ahead -> bucket label
    "1h": (1, 1),
    "same_day": (2, 24),
    "2-3d": (25, 72),
    "4-7d": (73, 168),
}
SEASONAL_PERIOD_HOURS = 168     # weekly seasonality for MASE scaling / snaive

# ---------------------------------------------------------------------------
# Daily-grain long-horizon benchmark protocol
# Same weekly origin-days as the hourly benchmark; targets are own-era daily
# means only (queue-times era is training input, never ground truth).
# ---------------------------------------------------------------------------
MAX_HORIZON_DAYS = 112          # forecast up to 16 weeks ahead
DAILY_HORIZON_BUCKETS = {       # days-ahead -> bucket label
    "1-7d": (1, 7),
    "8-14d": (8, 14),
    "15-28d": (15, 28),
    "29-56d": (29, 56),
    "57-112d": (57, 112),
}
SEASONAL_PERIOD_DAYS = 7        # weekly seasonality for daily MASE / snaive
