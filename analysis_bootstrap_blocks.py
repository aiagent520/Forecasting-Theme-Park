#!/usr/bin/env python3
"""Moving-block bootstrap over consecutive weekly origins (reviewer request).

analysis_bootstrap_ci.py resamples origin days independently, which preserves
within-day dependence (rides sharing weather/crowd shocks) but not serial
dependence between consecutive origins whose long-horizon target windows
overlap. Here we repeat the daily headline comparisons with a moving-block
bootstrap over the ORDERED origin days, block lengths L = 1..4 weekly origins
(L=1 reproduces the i.i.d. origin-day bootstrap). Same statistic as before:
mean over origin-days of the per-day mean scaled AE; 95% percentile CI on the
pairwise difference.

No retraining; reads the saved forecast parquets via run_significance loaders.

Usage: python3 analysis_bootstrap_blocks.py | tee results_bootstrap_blocks.txt
"""

import numpy as np
import pandas as pd

from run_significance import load_daily_long

B = 10_000
LENGTHS = (1, 2, 3, 4)
RNG = np.random.default_rng(0)

DAILY_BUCKETS = ["1-7d", "8-14d", "15-28d", "29-56d", "57-112d"]


def block_diff_ci(va, vb, L):
    """Moving-block bootstrap CI for mean(va) - mean(vb), days in time order."""
    T = len(va)
    L = min(L, T)
    n_blocks = int(np.ceil(T / L))
    starts = RNG.integers(0, T - L + 1, size=(B, n_blocks))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :T]
    d = va[idx].mean(axis=1) - vb[idx].mean(axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def report(name, wide, pairs):
    print(f"\n=== {name} — moving-block bootstrap over ordered origins "
          f"(B={B:,}, 95% CI on diff) ===")
    for a, b in pairs:
        print(f"\n{a} vs {b}:")
        for bucket in DAILY_BUCKETS:
            sub = wide[wide["bucket"] == bucket][["day", a, b]].dropna()
            if sub.empty:
                continue
            day = sub.groupby("day")[[a, b]].mean().sort_index()
            va, vb = day[a].to_numpy(), day[b].to_numpy()
            parts = []
            for L in LENGTHS:
                lo, hi = block_diff_ci(va, vb, L)
                sig = "*" if lo > 0 or hi < 0 else " "
                parts.append(f"L={L}: [{lo:+.3f},{hi:+.3f}]{sig}")
            print(f"  {bucket:>8} (T={len(day):>2}) diff={va.mean() - vb.mean():+.3f}  "
                  + "  ".join(parts))


def main():
    wide, _ = load_daily_long("q2_stage2_fc.parquet", "cond")
    report("Q2 Stage 2 (daily, all regimes)", wide,
           [("splice_cal", "fine"), ("splice_cal", "splice")])
    report("Q2 Stage 2 (daily, regime A)", wide[wide["regime"] == "A"],
           [("splice_cal", "fine"), ("splice_cal", "splice")])

    wide, _ = load_daily_long("q3_curve_fc.parquet", "cell")
    report("Q3 learning curve (daily, regime A)", wide[wide["regime"] == "A"],
           [("k02_aug", "k02_fine"), ("full_aug", "full_fine")])


if __name__ == "__main__":
    main()
