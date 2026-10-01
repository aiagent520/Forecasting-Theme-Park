#!/usr/bin/env python3
"""Q5: model class x regime (hourly protocol, fine data held fixed).

The Q2-Q4 program varied the DATA under a fixed model; here the data is
fixed (own-era fine hourly, the same frame every hourly experiment uses)
and the MODEL CLASS varies:

  gbm          - the Tier-2 global LightGBM comparator, loaded from the
                 saved Q2 Stage-1 forecasts (rec_none column of
                 q2_stage1_fc.parquet == run_tier2_gbm's model)
  snaive       - Tier-1 anchor (same hour one week earlier), recomputed
  nhits        - N-HiTS (neuralforecast), global model over all series
  patchtst     - PatchTST (neuralforecast), global transformer baseline.
                 (TFT was the original plan; at library defaults one TFT
                 fit needs >1.5h on this machine — LSTM + attention over
                 504 raw hourly steps — infeasible for rolling-origin
                 refits. PatchTST is the standard transformer baseline
                 at feasible cost: attention over ~40 patches.)
  chronos_bolt - amazon/chronos-bolt-base, ZERO-SHOT per series (no
                 training at all): context = trailing grid hours at the
                 origin, median quantile; horizons beyond the model's
                 native 64 steps are autoregressively unrolled

Tuning-parity policy (documented, deliberately minimal — mirrors the
untuned ~default LightGBM): neural models get h=168, input_size=336
(2 full weekly cycles, = the MIN_FIT_HOURS rule), scaler_type="robust"
(standard choice for series with heterogeneous scales), and library
defaults for everything else (max_steps=1000, random_seed=1). No
per-model hyperparameter search for ANY entrant.

Refit cadence (documented deviation from the GBM comparator): neural
models are refit every REFIT_EVERY=4th origin-day (every 28 days) on
data strictly before that day's 00:00; intervening origins reuse the
latest weights but condition on context up to each origin (weights
never see post-refit-day data; inputs never see post-origin data).
The GBM comparator refits per origin-day — cheap for trees, standard
monthly cadence for neural nets; recency reaches the nets through the
input window either way. Chronos is zero-shot everywhere.

Irregular series: same regular-hourly-grid, closed-hours=0 convention as
run_tier1_sf (fitting/context only; evaluation touches observed hours).

Outputs: per-model bucket + regime x bucket tables, DM + MCB vs gbm and
snaive (per-origin-day differentials, NW lag 1), forecasts saved to
data/q5_models_fc.parquet.

Usage: python3 run_q5_models.py [--days N] | tee results_q5_models.txt
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch

from config import DATA_DIR, HORIZON_BUCKETS, MAX_HORIZON_HOURS
from harness import (
    evaluate,
    mase_scales,
    rolling_origins,
    summarize,
    train_test_split,
)
from run_significance import HOURLY_LAG, report
from run_tier1 import forecast_snaive, target_grid
from run_tier1_sf import to_regular_long

INPUT_SIZE = 336            # 2 full weekly cycles (= MIN_FIT_HOURS)
CHRONOS_CONTEXT = 2048      # chronos-bolt native max context
CHRONOS_MODEL = "amazon/chronos-bolt-base"
REFIT_EVERY = 4             # refit neural weights every 4th origin-day (28d)
NEURAL = ("nhits", "patchtst")
MODELS = ["snaive", "gbm", "nhits", "patchtst", "chronos_bolt"]


def make_nf(max_steps: int | None):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS, PatchTST

    common = dict(
        h=MAX_HORIZON_HOURS,
        input_size=INPUT_SIZE,
        scaler_type="robust",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    if max_steps is not None:
        common["max_steps"] = max_steps
    return NeuralForecast(
        models=[NHITS(**common, alias="nhits"),
                PatchTST(**common, alias="patchtst")],
        freq="h",
    )


def load_chronos():
    from chronos import BaseChronosPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return BaseChronosPipeline.from_pretrained(
        CHRONOS_MODEL, device_map=device, torch_dtype=torch.float32
    )


def chronos_forecast(pipe, long_pred: pd.DataFrame, origin: pd.Timestamp):
    ctxs, sids = [], []
    for sid, g in long_pred.groupby("unique_id", observed=True):
        ctxs.append(
            torch.tensor(g["y"].to_numpy()[-CHRONOS_CONTEXT:], dtype=torch.float32)
        )
        sids.append(sid)
    qs, _ = pipe.predict_quantiles(
        ctxs,
        prediction_length=MAX_HORIZON_HOURS,
        quantile_levels=[0.5],
    )
    med = qs[..., 0].cpu().numpy()          # [n_series, h]
    ts = pd.date_range(origin, periods=MAX_HORIZON_HOURS, freq="h")
    return pd.concat(
        [
            pd.DataFrame({
                "series_id": sid, "origin": origin, "ts": ts,
                "y_hat": np.clip(m, 0, None),
            })
            for sid, m in zip(sids, med)
        ],
        ignore_index=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None,
                    help="limit to the first N origin-days (smoke test)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override neural max_steps (plumbing smoke test)")
    args = ap.parse_args()

    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    origins = rolling_origins(hourly)
    days = sorted({o.normalize() for o in origins})
    if args.days:
        days = days[: args.days]
        origins = [o for o in origins if o.normalize() in set(days)]

    # GBM comparator from the saved Q2 Stage-1 run (rec_none == Tier-2 GBM)
    s1 = pd.read_parquet(DATA_DIR / "q2_stage1_fc.parquet")
    gbm = s1[["series_id", "origin", "ts", "rec_none"]].rename(
        columns={"rec_none": "y_hat"})
    gbm = gbm[gbm["origin"].isin(origins)].assign(model="gbm")
    gbm["series_id"] = gbm["series_id"].astype(str)

    print(
        f"Hourly data: {len(hourly):,} rows, {hourly['series_id'].nunique()} "
        f"series | {len(days)} origin-days x 3 origins "
        f"({origins[0]} .. {origins[-1]}) | gbm rows reused: {len(gbm):,}"
    )

    pipe = load_chronos()

    fc_frames = [gbm]
    scale_frames = []
    nf, fitted_ids = None, None
    for i, day in enumerate(days):
        day_origins = [o for o in origins if o.normalize() == day]
        midnight = day_origins[0]                      # 00:00 origin
        if i % REFIT_EVERY == 0:                       # monthly refit cadence
            train0, _ = train_test_split(hourly, midnight)
            long_fit = to_regular_long(train0, midnight)
            fitted_ids = set(long_fit["unique_id"].unique())
            nf = make_nf(args.max_steps)
            nf.fit(df=long_fit)
            print(f"  refit at {day.date()} ({len(fitted_ids)} series)",
                  file=sys.stderr)
        for origin in day_origins:
            train, test = train_test_split(hourly, origin)
            grid = target_grid(test, origin)
            long_pred = to_regular_long(train, origin)
            long_pred = long_pred[long_pred["unique_id"].isin(fitted_ids)]

            preds = nf.predict(df=long_pred).rename(
                columns={"unique_id": "series_id", "ds": "ts"})
            for name in NEURAL:
                fc = preds[["series_id", "ts", name]].rename(
                    columns={name: "y_hat"})
                fc["y_hat"] = np.clip(fc["y_hat"], 0, None)
                fc = grid.merge(fc, on=["series_id", "ts"], how="inner")
                fc_frames.append(fc.assign(model=name))

            fc = chronos_forecast(pipe, long_pred, origin)
            fc = grid.merge(fc.drop(columns="origin"),
                            on=["series_id", "ts"], how="inner")
            fc_frames.append(fc.assign(model="chronos_bolt"))

            fc_frames.append(
                forecast_snaive(train, grid, origin).assign(model="snaive"))

            scales = mase_scales(train)
            scales.index.name = "series_id"
            scale_frames.append(
                scales.rename("scale").to_frame().assign(origin=origin))
        print(f"  day {i + 1}/{len(days)} {day.date()} done", file=sys.stderr)

    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()
    out = pd.concat(fc_frames, ignore_index=True)
    out.to_parquet(DATA_DIR / "q5_models_fc.parquet", index=False)
    print(f"saved {len(out):,} forecast rows -> data/q5_models_fc.parquet")

    scored = {}
    for name in MODELS:
        sc = evaluate(out[out["model"] == name], hourly)
        scored[name] = sc
        print(f"\n=== q5_{name} — by horizon bucket ===")
        print(summarize(sc, scales, by=("bucket",)).to_string())
    for name in MODELS:
        print(f"\n=== q5_{name} — by regime x bucket ===")
        print(summarize(scored[name], scales, by=("regime", "bucket")).to_string())

    # --- DM + MCB -----------------------------------------------------------
    sc = pd.concat(
        [scored[name].assign(model=name) for name in MODELS], ignore_index=True
    ).join(scales, on="series_id")
    sc = sc[sc["scale"].notna()].copy()
    sc["sae"] = sc["abs_err"] / sc["scale"]
    wide = sc.pivot_table(
        index=["series_id", "regime", "origin", "bucket", "ts"],
        columns="model", values="sae", aggfunc="first",
    ).reset_index()
    wide["day"] = wide["origin"].dt.normalize()

    buckets = list(HORIZON_BUCKETS)
    pairs = [
        ("nhits", "gbm"),
        ("patchtst", "gbm"),
        ("chronos_bolt", "gbm"),
        ("nhits", "snaive"),
        ("chronos_bolt", "snaive"),
    ]
    report("Q5 model class (hourly, all regimes)", wide, MODELS,
           pairs=pairs, buckets=buckets, lag=HOURLY_LAG)
    for reg in ("A", "B", "C"):
        report(f"Q5 model class (hourly, regime {reg} only)",
               wide[wide["regime"] == reg], MODELS,
               pairs=pairs, buckets=buckets, lag=HOURLY_LAG)


if __name__ == "__main__":
    main()
