#!/usr/bin/env python3
"""Q4/Q5 cross-cell: how much OWN data does the zero-shot foundation model
need before its cold-start win kicks in?

Q4 answered the cold-start question for the fixed GBM (cross-park pooling
is a substitute, wash by 4-8 weeks of own data). Q5 showed zero-shot
Chronos-Bolt wins regime B outright with full context. This addendum
crosses the axes: Epic's inference CONTEXT is truncated to k weeks of own
history at each origin — k in {2, 4, 8, full} (full = the model's native
2048h cap, ~12.2 weeks) — and scored against the saved Q4 GBM pooling
cells (k*_global_all) on identical targets.

Zero-shot everywhere: no training; truncation applies to the context
window only, so this measures pure data-need at inference. Same
regular-grid / closed-hours=0 convention as every hourly experiment;
evaluation touches observed hours only; per-origin full-history MASE
scales (identical denominators across conditions).

Usage: python3 run_q4_chronos_coldstart.py [--origins N] | tee results_q4_chronos.txt
"""

import argparse
import sys

import pandas as pd

from config import DATA_DIR, HORIZON_BUCKETS
from harness import evaluate, mase_scales, rolling_origins, summarize, train_test_split
from run_q5_models import chronos_forecast, load_chronos
from run_significance import HOURLY_LAG, report
from run_tier1 import target_grid
from run_tier1_sf import to_regular_long

KS = (2, 4, 8, None)  # weeks of own context; None = full (native 2048h cap)


def cell_name(k):
    return f"chronos_k{k:02d}" if k else "chronos_full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", type=int, default=None)
    args = ap.parse_args()

    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    epic = hourly[hourly["regime"] == "B"]
    epic_ids = set(epic["series_id"].unique())
    origins = rolling_origins(hourly)
    if args.origins:
        origins = origins[-args.origins:]
    print(
        f"Epic: {len(epic):,} rows, {len(epic_ids)} series | "
        f"{len(origins)} origins ({origins[0]} .. {origins[-1]}) | "
        f"context k = 2/4/8/full weeks"
    )

    pipe = load_chronos()

    fc_frames, scale_frames = [], []
    for i, origin in enumerate(origins):
        train, test = train_test_split(hourly, origin)
        grid = target_grid(test, origin)
        grid = grid[grid["series_id"].isin(epic_ids)]
        long_pred = to_regular_long(train, origin)
        long_pred = long_pred[long_pred["unique_id"].isin(epic_ids)]

        for k in KS:
            ctx = long_pred
            if k:
                cutoff = origin - pd.Timedelta(hours=k * 168)
                ctx = long_pred[long_pred["ds"] >= cutoff]
            fc = chronos_forecast(pipe, ctx, origin)
            fc = grid.merge(fc.drop(columns="origin"),
                            on=["series_id", "ts"], how="inner")
            fc_frames.append(fc.assign(cell=cell_name(k)))

        scales = mase_scales(train[train["series_id"].isin(epic_ids)])
        scales.index.name = "series_id"
        scale_frames.append(scales.rename("scale").to_frame().assign(origin=origin))
        if (i + 1) % 6 == 0:
            print(f"  origin {i + 1}/{len(origins)} done", file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    out = pd.concat(fc_frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "q4_chronos_fc.parquet", index=False)
    print(f"saved {len(out):,} forecast rows -> data/q4_chronos_fc.parquet")

    # saved Q4 GBM pooling cells on the identical targets
    q4 = pd.read_parquet(DATA_DIR / "q4_coldstart_fc.parquet")
    q4 = q4[q4["cell"].str.endswith("_global_all")].copy()
    q4["cell"] = "gbm_" + q4["cell"].str.replace("_global_all", "", regex=False)
    q4 = q4[q4["origin"].isin(origins)]
    out = pd.concat([out, q4], ignore_index=True)

    cells = [cell_name(k) for k in KS] + ["gbm_k02", "gbm_k04", "gbm_k08", "gbm_full"]
    scored = {}
    for cell in cells:
        sc = evaluate(out[out["cell"] == cell], hourly)
        scored[cell] = sc
        print(f"\n=== {cell} — by horizon bucket (Epic only) ===")
        print(summarize(sc, scales, by=("bucket",)).to_string())

    sc = pd.concat(
        [scored[c].assign(cell=c) for c in cells], ignore_index=True
    ).join(scales, on="series_id")
    sc = sc[sc["scale"].notna()].copy()
    sc["sae"] = sc["abs_err"] / sc["scale"]
    wide = sc.pivot_table(
        index=["series_id", "origin", "bucket", "ts"],
        columns="cell", values="sae", aggfunc="first",
    ).reset_index()
    wide["day"] = wide["origin"].dt.normalize()

    pairs = [
        ("chronos_k02", "gbm_k02"),
        ("chronos_k04", "gbm_k04"),
        ("chronos_k08", "gbm_k08"),
        ("chronos_full", "gbm_full"),
        ("chronos_k02", "chronos_full"),
    ]
    report("Q4 chronos cold-start (Epic, hourly)", wide, cells,
           pairs=pairs, buckets=list(HORIZON_BUCKETS), lag=HOURLY_LAG)


if __name__ == "__main__":
    main()
