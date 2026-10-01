#!/usr/bin/env python3
"""Q2 addendum: shape as a LAYER vs shape in the DATA (same information).

Completes the factorial around "where do you inject the coarse archive's
seasonal knowledge":

  splice_cal   (run_q2_grain)  — inject into the TRAINING DATA (rescale
                coarse rows, let the GBM learn shape + interactions)
  fine_shape   (this script)   — inject into the OUTPUT: take the saved
                fine-only GBM forecasts (q2_stage2_fc.parquet, cond=fine)
                and multiply by the coarse-era month-shape factor used by
                Tier-1 dow_shape_annual:
                    factor(sid, month) = coarse mean in target month
                                       / coarse mean over the months spanned
                                         by the trailing 8-week window
                Level cancels inside the ratio, so NO level calibration is
                needed — this is the "can't retrain" deployment option.

Rows without a factor (regime C: no coarse era; or missing coarse months)
pass through unchanged, so regime C cells are identical to fine by
construction; regime-stratified tables are the honest read.

Scoring identical to run_q2_grain (same protocol, same per-origin-averaged
MASE scales), plus DM + MCB against fine / splice / splice_cal via
run_significance (per-origin-day differentials, NW lag 4; daily long
buckets anti-conservative — overlapping 112d windows).

Runs AFTER run_q2_grain.py. Usage:
    python3 run_q2_shape_layer.py | tee results_q2_shape_layer.txt
"""

import numpy as np
import pandas as pd

from config import DAILY_HORIZON_BUCKETS, DATA_DIR
from run_daily import evaluate, mase_scales, summarize
from run_significance import DAILY_LAG, report

METHODS = ["fine", "splice", "splice_cal", "fine_shape"]


def month_shape(train: pd.DataFrame, origin: pd.Timestamp):
    """Coarse-era month factors, exactly as in forecast_dow_shape_annual."""
    qt = train[train["source"] == "queue_times"].copy()
    if qt.empty:
        return None
    qt["month"] = qt["date"].dt.month
    month_mean = qt.groupby(["series_id", "month"])["y"].mean().rename("m_target")
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
    return month_mean, base


def main():
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    actuals = daily[daily["source"] == "own_5min"]

    fc_all = pd.read_parquet(DATA_DIR / "q2_stage2_fc.parquet")
    fc_all["date"] = pd.to_datetime(fc_all["date"])
    fine = fc_all[fc_all["cond"] == "fine"]
    origins = sorted(fine["origin"].unique())

    shaped, applied = [], 0
    scale_frames = []
    for origin in origins:
        train = daily[daily["date"] < origin]
        scales = mase_scales(train)
        scales.index.name = "series_id"
        scale_frames.append(
            scales.rename("scale").to_frame().assign(origin=origin))
        f = fine[fine["origin"] == origin].copy()
        res = month_shape(train, origin)
        if res is not None:
            month_mean, base = res
            f["month"] = f["date"].dt.month
            f = f.join(month_mean, on=["series_id", "month"])
            f = f.join(base, on="series_id")
            ok = f["m_target"].notna() & (f["m_base"] > 0)
            f.loc[ok, "y_hat"] *= f.loc[ok, "m_target"] / f.loc[ok, "m_base"]
            applied += int(ok.sum())
            f = f.drop(columns=["month", "m_target", "m_base"])
        shaped.append(f)
    shaped = pd.concat(shaped, ignore_index=True).assign(cond="fine_shape")
    scales = pd.concat(scale_frames).groupby("series_id")["scale"].mean()

    shaped.to_parquet(DATA_DIR / "q2_shape_layer_fc.parquet", index=False)
    print(
        f"fine_shape: {len(shaped):,} rows, factor applied to {applied:,} "
        f"({applied / len(shaped):.0%}); saved -> data/q2_shape_layer_fc.parquet"
    )

    combined = pd.concat([fc_all, shaped], ignore_index=True)
    for cond in METHODS:
        sc = evaluate(combined[combined["cond"] == cond], actuals)
        print(f"\n=== gbm_{cond} — by lead bucket ===")
        print(summarize(sc, scales, by=("bucket",)).to_string())
        print(f"\n=== gbm_{cond} — by regime x lead bucket ===")
        print(summarize(sc, scales, by=("regime", "bucket")).to_string())

    # --- DM + MCB -----------------------------------------------------------
    sc = evaluate(combined, actuals).join(scales, on="series_id")
    sc = sc[sc["scale"].notna()].copy()
    sc["sae"] = sc["abs_err"] / sc["scale"]
    wide = sc.pivot_table(
        index=["series_id", "regime", "origin", "bucket", "date"],
        columns="cond", values="sae", aggfunc="first",
    ).reset_index()
    wide["day"] = wide["origin"]

    buckets = list(DAILY_HORIZON_BUCKETS)
    pairs = [
        ("fine_shape", "fine"),
        ("fine_shape", "splice"),
        ("fine_shape", "splice_cal"),
    ]
    report(
        "Q2 shape layer (daily, all regimes)", wide, METHODS,
        pairs=pairs, buckets=buckets, lag=DAILY_LAG,
        note="daily long buckets: overlapping origins -> anti-conservative p",
    )
    report(
        "Q2 shape layer (daily, regime A only)",
        wide[wide["regime"] == "A"], METHODS,
        pairs=pairs, buckets=buckets, lag=DAILY_LAG,
    )
    report(
        "Q2 shape layer (daily, regime B only)",
        wide[wide["regime"] == "B"], METHODS,
        pairs=pairs, buckets=buckets, lag=DAILY_LAG,
    )


if __name__ == "__main__":
    main()
