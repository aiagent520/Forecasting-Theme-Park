#!/usr/bin/env python3
"""Origin-day block bootstrap CIs for the headline comparisons (reviewer request).

Complements run_significance.py's DM tests: instead of a t-approximation on
per-origin-day loss differentials, resample ORIGIN-DAYS with replacement
(the natural dependence block: rides within a day share weather/crowds) and
report 95% percentile CIs on (a) each method's MASE and (b) the pairwise
MASE difference. MASE here = mean over origin-days of the per-day mean
scaled AE — the same statistic the DM tests average.

No retraining; reads the saved forecast parquets via run_significance loaders.

Usage: python3 analysis_bootstrap_ci.py | tee results_bootstrap_ci.txt
"""

import numpy as np
import pandas as pd

from run_significance import (
    hourly_scaffold,
    load_daily_long,
    load_q2_stage1,
    load_q4,
)

B = 10_000
RNG = np.random.default_rng(0)


def boot_pair(wide: pd.DataFrame, a: str, b: str, bucket: str):
    """Per-origin-day means for methods a,b in a bucket -> bootstrap CIs."""
    sub = wide[wide["bucket"] == bucket][["day", a, b]].dropna()
    if sub.empty:
        return None
    day = sub.groupby("day")[[a, b]].mean()
    T = len(day)
    idx = RNG.integers(0, T, size=(B, T))
    va, vb = day[a].to_numpy(), day[b].to_numpy()
    ma, mb = va[idx].mean(axis=1), vb[idx].mean(axis=1)
    d = ma - mb
    ci = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))
    return {
        "T": T,
        "a_mean": va.mean(), "a_ci": ci(ma),
        "b_mean": vb.mean(), "b_ci": ci(mb),
        "d_mean": va.mean() - vb.mean(), "d_ci": ci(d),
        "p_sign": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())),
    }


def report(name, wide, pairs, buckets):
    print(f"\n=== {name} — origin-day block bootstrap (B={B:,}, 95% CI) ===")
    for a, b in pairs:
        print(f"\n{a} vs {b}:")
        for bucket in buckets:
            r = boot_pair(wide, a, b, bucket)
            if r is None:
                continue
            sig = "*" if r["d_ci"][0] > 0 or r["d_ci"][1] < 0 else " "
            print(
                f"  {bucket:>8} (T={r['T']:>2}): "
                f"{a}={r['a_mean']:.3f} [{r['a_ci'][0]:.3f},{r['a_ci'][1]:.3f}]  "
                f"{b}={r['b_mean']:.3f} [{r['b_ci'][0]:.3f},{r['b_ci'][1]:.3f}]  "
                f"diff={r['d_mean']:+.3f} [{r['d_ci'][0]:+.3f},{r['d_ci'][1]:+.3f}]{sig}"
            )


def main():
    hourly, scales, _ = hourly_scaffold()
    hourly_buckets = ["1h", "same_day", "2-3d", "4-7d"]
    daily_buckets = ["1-7d", "8-14d", "15-28d", "29-56d", "57-112d"]

    wide, _ = load_q2_stage1(scales)
    report("Q2 Stage 1 (hourly, all regimes)", wide,
           [("rec_none", "rec_l2"), ("rec_none", "rec_l2l3")], hourly_buckets)

    wide, _ = load_daily_long("q2_stage2_fc.parquet", "cond")
    report("Q2 Stage 2 (daily, all regimes)", wide,
           [("splice_cal", "fine"), ("splice_cal", "splice"),
            ("fine", "splice")], daily_buckets)
    report("Q2 Stage 2 (daily, regime A)", wide[wide["regime"] == "A"],
           [("splice_cal", "fine"), ("splice_cal", "splice")], daily_buckets)

    wide, _ = load_daily_long("q3_curve_fc.parquet", "cell")
    report("Q3 learning curve (daily, regime A)", wide[wide["regime"] == "A"],
           [("k02_aug", "k02_fine"), ("full_aug", "full_fine"),
            ("k04_fine", "full_fine")], daily_buckets)

    wide, _ = load_q4(hourly, scales)
    report("Q4 cold start (hourly, Epic)", wide,
           [("k02_global_all", "k02_local_series"),
            ("k02_global_all", "full_global_all")], hourly_buckets)


if __name__ == "__main__":
    main()
