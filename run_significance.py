#!/usr/bin/env python3
"""Significance testing (DM + MCB) over the saved forecast parquets.

Runs AFTER the experiment scripts; pure computation over their saved
forecasts (q2_stage1_fc, q2_stage2_fc, q3_curve_fc, q4_coldstart_fc).

Loss = scaled absolute error (abs error / per-series seasonal-naive MAE,
i.e. MASE units) so losses pool across series. Scales are computed once per
series on full history — any fixed positive per-series weighting is valid
for the tests; this one matches the headline MASE tables.

Diebold-Mariano: per comparison pair and horizon bucket, the loss
differential is averaged per ORIGIN-DAY (intraday origins pooled), giving a
short time series d_1..d_T (T ~ 20-25 weekly origin-days). DM stat uses a
Newey-West (Bartlett) long-run variance and the Harvey-Leybourne-Newbold
small-sample correction; two-sided p from t(T-1).
  - hourly protocol: NW lag 1 (weekly origins x 168h horizon -> target
    windows do not overlap; lag 1 is a safety margin).
  - daily protocol: NW lag 4 (~1.5*T^(1/3)). CAVEAT: the 112d horizon
    overlaps ~15 weekly origins, so long-bucket differentials are serially
    correlated beyond the truncation lag -> p-values there are
    anti-conservative; treat marginal p's sceptically and lean on MCB.

MCB (multiple comparisons with the best, Koning et al. 2005): per bucket,
methods are ranked within each (series_id, origin) block on mean scaled AE;
mean ranks reported with the Nemenyi critical distance
CD = q_{0.95}(K, inf)/sqrt(2) * sqrt(K(K+1)/(6N)). Methods whose mean rank
is within CD of the best are "in the best set" (marked *).

Usage: python3 run_significance.py | tee results_significance.txt
"""

import numpy as np
import pandas as pd
from scipy import stats

import harness
import run_daily
from config import DATA_DIR

HOURLY_LAG = 1
DAILY_LAG = 4
ALPHA = 0.95

# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def dm_test(d: np.ndarray, lag: int):
    """DM test on a series of per-origin-day loss differentials.

    Returns (mean_d, stat, p, T). Negative mean_d favours the FIRST method.
    """
    d = np.asarray(d, dtype=float)
    T = len(d)
    if T < 5:
        return np.nan, np.nan, np.nan, T
    dbar = d.mean()
    e = d - dbar
    lag = min(lag, T - 2)
    lrv = e @ e / T
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * (e[k:] @ e[:-k]) / T
    if lrv <= 0:
        return dbar, np.nan, np.nan, T
    stat = dbar / np.sqrt(lrv / T)
    h = lag + 1  # HLN small-sample correction with horizon proxy h
    stat *= np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2.0 * stats.t.sf(abs(stat), df=T - 1)
    return dbar, stat, p, T


def mcb_table(block_loss: pd.DataFrame):
    """block_loss: (series_id, origin) blocks x method columns, no NaN.

    Returns (mean_ranks sorted ascending, critical distance, N blocks)."""
    ranks = block_loss.rank(axis=1)
    mean_ranks = ranks.mean().sort_values()
    K, N = block_loss.shape[1], len(block_loss)
    q = stats.studentized_range.ppf(ALPHA, K, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(K * (K + 1) / (6.0 * N))
    return mean_ranks, cd, N

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(name, wide, methods, pairs, buckets, lag, note=""):
    """wide: columns [series_id, origin, day, bucket] + one sae col/method."""
    print(f"\n=== {name} — DM tests (per-origin-day differentials, NW lag {lag}) ===")
    if note:
        print(f"    {note}")
    header = f"{'pair':<38}" + "".join(f"{b:>22}" for b in buckets)
    print(header)
    for a, b in pairs:
        cells = []
        for bucket in buckets:
            sub = wide[wide["bucket"] == bucket][["day", a, b]].dropna()
            if sub.empty:
                cells.append(f"{'—':>22}")
                continue
            d = sub.groupby("day").apply(
                lambda g: (g[a] - g[b]).mean(), include_groups=False
            )
            dbar, _, p, T = dm_test(d.to_numpy(), lag)
            star = "*" if (p == p and p < 0.05) else " "
            cells.append(f"{dbar:+.3f} p={p:.3f}{star}".rjust(22))
        print(f"{a + ' vs ' + b:<38}" + "".join(cells))

    print(f"\n=== {name} — MCB mean ranks (blocks = series x origin; * = in best set) ===")
    for bucket in buckets:
        sub = wide[wide["bucket"] == bucket]
        block = (
            sub.groupby(["series_id", "origin"])[list(methods)].mean().dropna()
        )
        if block.empty:
            continue
        mean_ranks, cd, N = mcb_table(block)
        best = mean_ranks.iloc[0]
        parts = [
            f"{m} {r:.2f}{'*' if r <= best + cd else ''}"
            for m, r in mean_ranks.items()
        ]
        print(f"  {bucket:>8} (N={N}, CD={cd:.2f}): " + " | ".join(parts))

# ---------------------------------------------------------------------------
# Loaders — produce wide frames of scaled AE per method
# ---------------------------------------------------------------------------

def hourly_scaffold():
    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    scales = harness.mase_scales(hourly)
    scales.index.name = "series_id"
    regime = hourly[["series_id", "regime"]].drop_duplicates()
    return hourly, scales, regime


def load_q2_stage1(scales):
    fc = pd.read_parquet(DATA_DIR / "q2_stage1_fc.parquet")
    methods = ["cal_none", "cal_l2", "cal_l2l3", "rec_none", "rec_l2", "rec_l2l3"]
    ha = ((fc["ts"] - fc["origin"]).dt.total_seconds() / 3600).astype(int) + 1
    fc["bucket"] = harness._bucket(ha)
    fc = fc.dropna(subset=["bucket"]).join(scales, on="series_id")
    fc = fc[fc["scale"].notna()].copy()
    for m in methods:
        fc[m] = (fc["y"] - fc[m]).abs() / fc["scale"]
    fc["day"] = fc["origin"].dt.normalize()
    return fc[["series_id", "origin", "day", "bucket"] + methods], methods


def load_q4(hourly, scales):
    fc = pd.read_parquet(DATA_DIR / "q4_coldstart_fc.parquet")
    fc = fc.merge(hourly[["series_id", "ts", "y"]], on=["series_id", "ts"])
    fc = fc.join(scales, on="series_id")
    fc = fc[fc["scale"].notna()].copy()
    fc["sae"] = (fc["y"] - fc["y_hat"]).abs() / fc["scale"]
    ha = ((fc["ts"] - fc["origin"]).dt.total_seconds() / 3600).astype(int) + 1
    fc["bucket"] = harness._bucket(ha)
    fc = fc.dropna(subset=["bucket"])
    wide = fc.pivot_table(
        index=["series_id", "origin", "bucket", "ts"],
        columns="cell", values="sae", aggfunc="first",
    ).reset_index()
    wide["day"] = wide["origin"].dt.normalize()
    methods = sorted(fc["cell"].unique())
    return wide, methods


def load_daily_long(fname, col):
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    scales = run_daily.mase_scales(daily)
    scales.index.name = "series_id"
    own = daily[daily["source"] == "own_5min"]
    fc = pd.read_parquet(DATA_DIR / fname)
    fc["date"] = pd.to_datetime(fc["date"])
    fc = fc.merge(
        own[["series_id", "date", "regime", "y"]], on=["series_id", "date"]
    )
    fc = fc.join(scales, on="series_id")
    fc = fc[fc["scale"].notna()].copy()
    fc["sae"] = (fc["y"] - fc["y_hat"]).abs() / fc["scale"]
    da = (fc["date"] - fc["origin"]).dt.days + 1
    fc["bucket"] = run_daily._bucket(da)
    fc = fc.dropna(subset=["bucket"])
    wide = fc.pivot_table(
        index=["series_id", "regime", "origin", "bucket", "date"],
        columns=col, values="sae", aggfunc="first",
    ).reset_index()
    wide["day"] = wide["origin"]
    methods = sorted(fc[col].unique())
    return wide, methods

# ---------------------------------------------------------------------------

def main():
    hourly, scales, _ = hourly_scaffold()
    hourly_buckets = ["1h", "same_day", "2-3d", "4-7d"]
    daily_buckets = ["1-7d", "8-14d", "15-28d", "29-56d", "57-112d"]

    # Q2 Stage 1 — calibration x base (hourly)
    wide, methods = load_q2_stage1(scales)
    report(
        "Q2 Stage 1 (hourly, all regimes)", wide, methods,
        pairs=[
            ("rec_none", "rec_l2"), ("rec_none", "rec_l2l3"),
            ("cal_none", "cal_l2"), ("cal_none", "cal_l2l3"),
            ("rec_none", "cal_none"),
        ],
        buckets=hourly_buckets, lag=HOURLY_LAG,
    )

    # Q2 Stage 2 — data grain (daily)
    wide, methods = load_daily_long("q2_stage2_fc.parquet", "cond")
    s2_pairs = [
        ("splice_cal", "fine"), ("splice_cal", "splice"), ("fine", "splice"),
    ]
    report(
        "Q2 Stage 2 (daily, all regimes)", wide, methods,
        pairs=s2_pairs, buckets=daily_buckets, lag=DAILY_LAG,
        note="daily long buckets: overlapping origins -> anti-conservative p",
    )
    report(
        "Q2 Stage 2 (daily, regime A only)", wide[wide["regime"] == "A"],
        methods, pairs=s2_pairs, buckets=daily_buckets, lag=DAILY_LAG,
    )

    # Q3 — learning curve (daily)
    wide, methods = load_daily_long("q3_curve_fc.parquet", "cell")
    q3_pairs = [
        ("k02_aug", "k02_fine"), ("k04_aug", "k04_fine"),
        ("k08_aug", "k08_fine"), ("k16_aug", "k16_fine"),
        ("full_aug", "full_fine"), ("k04_fine", "full_fine"),
    ]
    report(
        "Q3 learning curve (daily, all regimes)", wide, methods,
        pairs=q3_pairs, buckets=daily_buckets, lag=DAILY_LAG,
        note="daily long buckets: overlapping origins -> anti-conservative p",
    )
    report(
        "Q3 learning curve (daily, regime A only)",
        wide[wide["regime"] == "A"], methods,
        pairs=q3_pairs, buckets=daily_buckets, lag=DAILY_LAG,
    )

    # Q4 — cold start pooling (hourly, Epic only)
    wide, methods = load_q4(hourly, scales)
    report(
        "Q4 cold start (hourly, Epic)", wide, methods,
        pairs=[
            ("k02_global_all", "k02_local_series"),
            ("k04_global_all", "k04_local_series"),
            ("k08_global_all", "k08_local_series"),
            ("full_global_all", "full_local_series"),
            ("k02_global_all", "full_global_all"),
        ],
        buckets=hourly_buckets, lag=HOURLY_LAG,
    )


if __name__ == "__main__":
    main()
