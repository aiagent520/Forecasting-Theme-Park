#!/usr/bin/env python3
"""Generate all paper figures from the saved forecast parquets.

Every number is recomputed from data/*.parquet with the exact scoring
conventions of the runner that produced it (hourly: harness.evaluate +
per-origin lag-168 scales; daily: run_daily.evaluate vs own-era actuals +
per-origin lag-7 scales) — no hardcoded results.

Outputs PDF + PNG into ../figures/.

Usage: python3 make_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import harness
import run_daily
from config import DATA_DIR, DAILY_HORIZON_BUCKETS, HORIZON_BUCKETS

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

HOURLY_BUCKETS = list(HORIZON_BUCKETS)          # 1h, same_day, 2-3d, 4-7d
DAILY_BUCKETS = list(DAILY_HORIZON_BUCKETS)     # 1-7d .. 57-112d
BUCKET_LABELS = {"1h": "1 h", "same_day": "same day", "2-3d": "2-3 d",
                 "4-7d": "4-7 d"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})

COLORS = {  # Okabe-Ito, colorblind-safe
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "grey": "#7f7f7f",
    "sky": "#56B4E9", "yellow": "#F0E442",
}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf|png")


# ---------------------------------------------------------------------------
# scoring helpers (replicate each runner's conventions exactly)
# ---------------------------------------------------------------------------

def hourly_scales(hourly, origins):
    frames = []
    for o in origins:
        s = harness.mase_scales(hourly[hourly["ts"] < o])
        s.index.name = "series_id"
        frames.append(s.rename("scale").to_frame().assign(origin=o))
    return pd.concat(frames).groupby("series_id")["scale"].mean()


def daily_scales(daily, origins):
    frames = []
    for o in origins:
        s = run_daily.mase_scales(daily[daily["date"] < o])
        s.index.name = "series_id"
        frames.append(s.rename("scale").to_frame().assign(origin=o))
    return pd.concat(frames).groupby("series_id")["scale"].mean()


def mase_table(scored, scales, by):
    df = scored.join(scales, on="series_id")
    df = df[df["scale"].notna()].copy()
    df["ase"] = df["abs_err"] / df["scale"]
    return df.groupby(list(by), observed=True)["ase"].mean()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_q2_grain(daily, dscales):
    """Central thesis figure: same model, three data conditions."""
    fc = pd.read_parquet(DATA_DIR / "q2_stage2_fc.parquet")
    actuals = daily[daily["source"] == "own_5min"]
    styles = {"fine": ("fine only", COLORS["grey"], "--"),
              "splice": ("+ coarse (naive splice)", COLORS["orange"], "-."),
              "splice_cal": ("+ coarse (calibrated)", COLORS["blue"], "-")}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, reg, title in zip(axes, ["A", "B"],
                              ["Regime A (2 yr coarse archive)",
                               "Regime B (new park, unrepresentative history)"]):
        for cond, (label, color, ls) in styles.items():
            sc = run_daily.evaluate(fc[fc["cond"] == cond], actuals)
            m = mase_table(sc[sc["regime"] == reg], dscales, ["bucket"])
            ax.plot(range(len(DAILY_BUCKETS)), m.reindex(DAILY_BUCKETS),
                    ls, color=color, marker="o", ms=3, label=label)
        ax.set_xticks(range(len(DAILY_BUCKETS)), DAILY_BUCKETS)
        ax.set_xlabel("lead time")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("MASE")
    axes[0].legend(fontsize=8, loc="upper left")
    save(fig, "fig_q2_grain")


def fig_q3_crossover(daily, dscales):
    """Learning curve: value of the coarse archive vs weeks of fine data."""
    fc = pd.read_parquet(DATA_DIR / "q3_curve_fc.parquet")
    actuals = daily[daily["source"] == "own_5min"]
    ks = ["k02", "k04", "k08", "k16", "full"]
    xt = ["2", "4", "8", "16", "full (~29)"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    for ax, bucket in zip(axes, ["1-7d", "57-112d"]):
        for arm, label, color in (("fine", "fine only", COLORS["grey"]),
                                  ("aug", "+ calibrated coarse", COLORS["blue"])):
            ys = []
            for k in ks:
                sc = run_daily.evaluate(fc[fc["cell"] == f"{k}_{arm}"], actuals)
                m = mase_table(sc[sc["regime"] == "A"], dscales, ["bucket"])
                ys.append(m.get(bucket, np.nan))
            ax.plot(range(len(ks)), ys, marker="o", ms=3, color=color,
                    ls="--" if arm == "fine" else "-", label=label)
        ax.set_xticks(range(len(ks)), xt)
        ax.set_xlabel("weeks of fine data (k)")
        ax.set_title(f"lead {bucket} (regime A)", fontsize=9)
    axes[0].set_ylabel("MASE")
    axes[1].legend(fontsize=8)
    save(fig, "fig_q3_crossover")


def fig_q5_models(hourly, hscales):
    """Model class x regime interaction on fixed data."""
    fc = pd.read_parquet(DATA_DIR / "q5_models_fc.parquet")
    styles = {"snaive": ("seasonal naive", COLORS["grey"], ":"),
              "gbm": ("LightGBM", COLORS["blue"], "-"),
              "nhits": ("N-HiTS", COLORS["green"], "--"),
              "patchtst": ("PatchTST", COLORS["purple"], "--"),
              "chronos_bolt": ("Chronos-Bolt (zero-shot)", COLORS["red"], "-")}
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6), sharey=True)
    titles = {"A": "Regime A (rich history)", "B": "Regime B (cold start)",
              "C": "Regime C (fine only)"}
    for ax, reg in zip(axes, ["A", "B", "C"]):
        for model, (label, color, ls) in styles.items():
            sc = harness.evaluate(fc[fc["model"] == model], hourly)
            m = mase_table(sc[sc["regime"] == reg], hscales, ["bucket"])
            ax.plot(range(len(HOURLY_BUCKETS)), m.reindex(HOURLY_BUCKETS),
                    ls, color=color, marker="o", ms=3, label=label)
        ax.set_xticks(range(len(HOURLY_BUCKETS)),
                      [BUCKET_LABELS[b] for b in HOURLY_BUCKETS])
        ax.set_xlabel("lead time")
        ax.set_title(titles[reg], fontsize=9)
    axes[0].set_ylabel("MASE")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "fig_q5_models")


def fig_q4_chronos(hourly, hscales):
    """Cold start: zero-shot foundation model vs pooled GBM vs own data."""
    ch = pd.read_parquet(DATA_DIR / "q4_chronos_fc.parquet")
    q4 = pd.read_parquet(DATA_DIR / "q4_coldstart_fc.parquet")
    q4 = q4[q4["cell"].str.endswith("_global_all")].copy()
    q4["cell"] = "gbm_" + q4["cell"].str.replace("_global_all", "", regex=False)
    ks = ["k02", "k04", "k08", "full"]
    xt = ["2", "4", "8", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    for ax, bucket in zip(axes, ["same_day", "4-7d"]):
        for src, prefix, label, color in (
                (ch, "chronos_", "Chronos-Bolt (zero-shot)", COLORS["red"]),
                (q4, "gbm_", "LightGBM (pooled, trained)", COLORS["blue"])):
            ys = []
            for k in ks:
                sc = harness.evaluate(src[src["cell"] == f"{prefix}{k}"], hourly)
                ys.append(mase_table(sc, hscales, ["bucket"]).get(bucket, np.nan))
            ax.plot(range(len(ks)), ys, marker="o", ms=3, color=color, label=label)
        ax.set_xticks(range(len(ks)), xt)
        ax.set_xlabel("weeks of own data (k)")
        ax.set_title(f"lead {BUCKET_LABELS[bucket]} (Epic, regime B)", fontsize=9)
    axes[0].set_ylabel("MASE")
    axes[1].legend(fontsize=8)
    save(fig, "fig_q4_chronos")


def fig_q6_covariates(daily, dscales):
    """Which signals matter at which leads (regime A, delta vs base)."""
    fc = pd.read_parquet(DATA_DIR / "q6_covariates_fc.parquet")
    actuals = daily[daily["source"] == "own_5min"]
    base_m = None
    conds = {"no_recency": ("- recency", COLORS["grey"]),
             "school": ("+ school calendar", COLORS["blue"]),
             "events": ("+ events", COLORS["purple"]),
             "weather": ("+ weather (oracle)", COLORS["orange"]),
             "all": ("+ all covariates", COLORS["green"])}
    m = {}
    for cond in ["base"] + list(conds):
        sc = run_daily.evaluate(fc[fc["cond"] == cond], actuals)
        m[cond] = mase_table(sc[sc["regime"] == "A"], dscales,
                             ["bucket"]).reindex(DAILY_BUCKETS)
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    x = np.arange(len(DAILY_BUCKETS))
    w = 0.16
    for i, (cond, (label, color)) in enumerate(conds.items()):
        ax.bar(x + (i - 2) * w, m[cond] - m["base"], w, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, DAILY_BUCKETS)
    ax.set_xlabel("lead time")
    ax.set_ylabel("$\\Delta$MASE vs base (lower = better)")
    ax.set_title("Feature-set ablation, regime A", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    save(fig, "fig_q6_covariates")


def fig_coverage(hourly):
    """Interval calibration: raw quantile band vs x1.2 widened."""
    fc = pd.read_parquet(DATA_DIR / "quantile_gbm_fc.parquet")
    sc = harness.evaluate(fc.rename(columns={"p50": "y_hat"}), hourly)
    sc = sc.rename(columns={"y_hat": "p50"})
    lvl = hourly.groupby("series_id")["y"].mean().rename("mean_wait")
    sc = sc.join(lvl, on="series_id")
    sc["size_bin"] = pd.cut(sc["mean_wait"], [0, 10, 20, 30, 50, np.inf],
                            labels=["<10m", "10-20m", "20-30m", "30-50m", "50m+"])
    sc["cov_raw"] = (sc["y"] >= sc["p10"]) & (sc["y"] <= sc["p90"])
    lo = np.clip(sc["p50"] + 1.2 * (sc["p10"] - sc["p50"]), 0, None)
    hi = sc["p50"] + 1.2 * (sc["p90"] - sc["p50"])
    sc["cov_adj"] = (sc["y"] >= lo) & (sc["y"] <= hi)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, by, xt, title in (
            (axes[0], "bucket", [BUCKET_LABELS[b] for b in HOURLY_BUCKETS],
             "by lead time"),
            (axes[1], "size_bin", None, "by ride size (mean wait)")):
        g = sc.groupby(by, observed=True)[["cov_raw", "cov_adj"]].mean()
        if by == "bucket":
            g = g.reindex(HOURLY_BUCKETS)
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["cov_raw"], 0.36, color=COLORS["grey"],
               label="raw [p10, p90]")
        ax.bar(x + 0.18, g["cov_adj"], 0.36, color=COLORS["blue"],
               label="widened $\\times$1.2")
        ax.axhline(0.8, color=COLORS["red"], lw=1, ls="--", label="nominal 80%")
        ax.set_xticks(x, xt if xt else g.index)
        ax.set_ylim(0.5, 1.0)
        ax.set_title(title, fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend(handles[:3], labels[:3], fontsize=7, loc="lower left")
    save(fig, "fig_coverage")


def main():
    hourly = pd.read_parquet(DATA_DIR / "hourly.parquet")
    daily = pd.read_parquet(DATA_DIR / "daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    print("computing scales ...", file=sys.stderr)
    hscales = hourly_scales(hourly, harness.rolling_origins(hourly))
    dscales = daily_scales(daily, run_daily.rolling_origins(daily))

    fig_q2_grain(daily, dscales)
    fig_q3_crossover(daily, dscales)
    fig_q5_models(hourly, hscales)
    fig_q4_chronos(hourly, hscales)
    fig_q6_covariates(daily, dscales)
    fig_coverage(hourly)


if __name__ == "__main__":
    main()
