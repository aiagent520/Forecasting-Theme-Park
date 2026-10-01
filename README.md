# Forecasting Benchmark Workspace

Phase 1 infrastructure for the paper (see `../FORECASTING_PAPER_PLAN.md`).

## Files
- `config.py` — paths, park/regime mapping, series curation rules,
  aggregation rules, protocol constants
- `build_dataset.py` — builds `data/hourly.parquet` + `data/daily.parquet` from
  raw `wait_times` (uniform rules; does NOT trust the spliced production
  `ride_daily_history_v2` at fine grain). Curation: pattern exclusions
  (HHN mazes, `_express` queues, first/last train pseudo-series) +
  `MIN_SERIES_DAYS` (56 distinct days) → 80 series.
- `harness.py` — rolling-origin splits (weekly origin-days × intraday
  `ORIGIN_HOURS` 00/09/13), MAE/RMSE/MASE, horizon buckets
- `run_tier1.py` — pandas Tier-1 baselines (snaive, hourly_avg)
- `run_tier1_sf.py` — statsforecast Tier-1 (MSTL[24,168]+AutoETS, Theta);
  fits on a regular hourly grid with closed hours filled as 0
- `run_tier2_gbm.py` — Tier-2 global LightGBM (M5-style): one model over all
  series, direct multi-horizon with lead as a feature; training samples from
  pseudo-origins on the eval cadence (leakage-free: recency features anchored
  at origin, seasonal lags >= 168h = max horizon). MASE 0.57/0.73/0.80/0.71 —
  beats all statistical baselines except mstl_ets at 1h.
- `run_daily.py` — daily-grain long-horizon Tier-1 (112d, regime-stratified):
  snaive7, dow_mean_8w, dow_mean_full (naive splice), snaive364,
  snaive364_cal (level-calibrated), dow_shape_annual (coarse-for-shape).
  Core finding: coarse source is ~2.2-2.6x level-inflated vs own data;
  splice < borrow < calibrate < shape-transfer on regime A; all coarse
  methods hurt on regime B (cold start).
- `run_prod_backtest.py` — faithful backtest of the deployed 3-layer XGBoost
  trip-plan system (L1 retrained at each origin-day; exact production
  features/clamps; deviations documented in the docstring). Orlando-only.
  Reports prod_l1 vs prod_full on both hourly and daily protocols; saves
  forecasts to `data/prod_backtest_fc.parquet`. Hourly MASE: l1 ~2.2,
  full ~1.1-1.3 (vs gbm_global 0.57-0.80).
- `run_q2_calibration.py` — Q2 Stage 1: factorial {cal-gbm, rec-gbm} ×
  {none, +L2, +L2+L3}, all LightGBM, fine data held fixed (hourly protocol).
  Calibration denominators use genuine out-of-sample weekly midnight-origin
  forecasts (origin spacing 7d == horizon 168h partitions past weeks; 4
  warm-up origin-days seed the pool). Finding: L2/L3 hurt BOTH bases in
  every bucket on clean data (rec 0.71 → +L2 0.78 → +L2L3 0.83 at 4-7d) —
  calibration is bias correction for level-inflated training data, not an
  accuracy booster. Saves `data/q2_stage1_fc.parquet`.
- `run_q2_grain.py` — Q2 Stage 2: FIXED daily global LightGBM × training data
  {fine-only, naive splice, calibrated splice} on the daily 112d protocol.
  splice_cal rescales coarse rows by the hindsight-free boundary-window level
  ratio before training. Finding (mirror of Stage 1): naive splice is worse
  than dropping the history; calibrated splice is best everywhere, with the
  advantage growing at long leads (regime A 57-112d: 0.41 vs fine 0.49 vs
  splice 0.71). "Fix the data, not the architecture." Saves
  `data/q2_stage2_fc.parquet`.
- `run_q3_curve.py` — Q3 learning curve: at each origin, pretend fine
  collection started k weeks ago (k ∈ {2,4,8,16,full}) × {± calibrated
  coarse}, fixed Stage-2 daily GBM. Findings: fine learning curve ~flat
  after 4 weeks; calibrated coarse history is NOT retired within 6.5 months
  — its long-lead value GROWS with k (needs a stable ratio window, k≥8);
  at k≤4 coarse rows swamp the tiny fine sample (global-pooling caveat,
  worst in regime C). Saves `data/q3_curve_fc.parquet`.
- `run_q4_coldstart.py` — Q4 cold-start transfer (hourly, Epic only): Epic
  history truncated to k ∈ {2,4,8,full} weeks per origin-day while other
  parks keep full history × {local-per-series, local-park, global-all}
  pooling, fixed Tier-2 GBM. Findings: global pooling helps most at k=2
  (4-7d MASE 0.81 vs 0.88-0.92 local) and decays to a wash by 4-8 weeks —
  cross-series transfer is a cold-start substitute (mirror of Q3's
  cross-era complement); 2wk+pooling is within ~7% of full-history
  accuracy. Saves `data/q4_coldstart_fc.parquet`.
- `run_q2_shape_layer.py` — Q2 addendum (run after run_q2_grain.py): same
  coarse-archive shape information injected at the OUTPUT instead of the
  data — fine-GBM forecasts × dow_shape_annual month factor. Regime A:
  statistically tied with splice_cal (viable no-retraining second-best);
  regime B: worst method (1.29-1.42) — a multiplicative layer cannot
  ignore unrepresentative history, the data-fix degrades gracefully →
  "fix the data" wins on robustness. Saves `data/q2_shape_layer_fc.parquet`.
- `run_q6_covariates.py` — Q6 covariate ablation: fixed splice_cal daily GBM,
  only the feature set varies {base, no_recency, +school, +events,
  +weather(oracle), +all}; Orlando-only covariates NaN'd for regime C
  (built-in control). Findings: recency gain small but persistent to 112d
  (not a short-lead feature); school calendar carries long leads; oracle
  weather adds nothing ≤14d (dead weight for daily means); +all best cell.
  Saves `data/q6_covariates_fc.parquet`.
- `run_q5_models.py` — Q5 model class × regime: data fixed (own-era fine
  hourly), model class varies {snaive, gbm (reused Q2 Stage-1 rec_none),
  N-HiTS, PatchTST (trained, 28d refit cadence), Chronos-Bolt zero-shot}.
  Tuning parity: no hyperparameter search for anyone; TFT dropped
  (infeasible at defaults on this hardware, swap documented). Findings:
  every neural model beats GBM at 1h (chronos 0.47 vs 0.57 MASE); GBM wins
  multi-day leads overall; zero-shot Chronos-Bolt is MCB rank 1 in ALL
  buckets in cold-start regime B — the foundation prior substitutes for
  missing history. Saves `data/q5_models_fc.parquet`.
- `run_q4_chronos_coldstart.py` — Q4/Q5 cross-cell (run after both): Epic's
  inference context truncated to k∈{2,4,8,full} weeks, zero-shot Chronos-Bolt
  vs the saved Q4 GBM pooling cells on identical targets. Findings: chronos
  saturates by ~2 weeks of own data; at k=2 it beats the pooled GBM at
  1h..2-3d (p≤0.012) and beats gbm_full point-wise except 4-7d — day-1 new
  park = zero-shot foundation model. Saves `data/q4_chronos_fc.parquet`.
- `run_quantile_gbm.py` — quantile variant of the Tier-2 GBM: identical
  samples/features/hyperparameters, objective→quantile at α∈{0.1,0.5,0.9}.
  Findings: p50 ≈ l1 point model (no accuracy tax); raw [p10,p90] band
  undercovers uniformly (~69% vs 80% nominal) across regimes, leads, and
  ride sizes; a single ×1.2 widening about p50 restores nominal coverage.
  Saves `data/quantile_gbm_fc.parquet`.
- `run_significance.py` — DM + MCB significance tests over the saved Q2/Q3/Q4
  forecast parquets (run last). Loss = scaled AE (MASE units); DM on
  per-origin-day differentials (Newey-West + HLN, lag 1 hourly / 4 daily —
  daily long buckets anti-conservative due to overlapping 112d windows);
  MCB = Nemenyi mean ranks over (series, origin) blocks. All headline
  claims survive: Q2S1 rec_none sole best-set member everywhere; Q2S2
  splice_cal sole best-set member everywhere; Q3 k02_aug contamination and
  long-lead aug gains significant, k04_fine≈full_fine n.s.; Q4 k=2 pooling
  gain significant, gone by k=4-8.
- `make_release.py` — builds the public dataset bundle → `../release/`
  (raw 5-min parquet, hourly/daily ground truth, series metadata, level
  ratios, qt-ride id crosswalk, covariates, protocol origin lists, all
  saved forecast parquets, stored production forecasts). Coarse
  third-party daily VALUES are excluded (Queue-Times terms don't grant
  redistribution, and their historical pages are bot-protected as of
  2026-09, so a user-side fetch script was investigated and abandoned);
  `--include-coarse` exists should permission ever arrive. `DATASHEET.md`
  is auto-extracted from `PAPER_DRAFT.md` Appendix A so the two can't
  drift. Upload steps and citation-placement checklist:
  `../RELEASE_GUIDE.md`.
- `run_prod_fidelity.py` — run AFTER run_prod_backtest.py. Scores the stored
  production forecasts (`forecast_trip_plan`: 82 next-day runs + frozen
  2026-05-31 long-lead snapshot) and checks reproduction fidelity
  (corr 0.88-0.93; deployed-as-served beats the retrained reproduction on
  identical targets — its L2/L3 were calibrated against a *fixed* stale
  2026-03-22 base, a more stable setup; so the backtest is conservative).

## Usage
```bash
python3 build_dataset.py             # after each server DB sync
python3 run_tier1.py --origins 3     # quick smoke test
python3 run_tier1.py                 # full rolling-origin run
python3 run_tier1_sf.py              # statsforecast run (~45s/origin)
python3 run_tier2_gbm.py             # global LightGBM (full run ~15 min)
python3 run_daily.py                 # daily 112d protocol, Tier-1 strategies
python3 run_prod_backtest.py         # production 3-layer reproduction
python3 run_prod_fidelity.py         # stored-forecast fidelity (after backtest)
python3 run_q2_calibration.py        # Q2 Stage 1 calibration factorial (~30 min)
python3 run_q2_grain.py              # Q2 Stage 2 data-grain conditions (~10 min)
python3 run_q3_curve.py              # Q3 learning curve (~25 min)
python3 run_q4_coldstart.py          # Q4 cold-start pooling (~20 min)
python3 run_q2_shape_layer.py        # shape-layer addendum (~1 min, after q2_grain)
python3 run_q6_covariates.py         # Q6 covariate ablation (~15 min)
python3 run_q5_models.py             # Q5 model classes (~4.5 h, needs torch/neuralforecast/chronos)
python3 run_quantile_gbm.py          # quantile GBM + coverage (~15 min)
python3 run_q4_chronos_coldstart.py  # chronos context curve (~10 min, after q4+q5)
python3 run_significance.py | tee results_significance.txt   # DM + MCB (~1 min)
python3 make_release.py              # build public release bundle → ../release/
```

## Forecast interface
Every method produces a DataFrame `[series_id, origin, ts, y_hat]`;
`harness.evaluate` + `harness.summarize` do the rest. Ground truth is the
hourly mean of 5-min STANDBY readings (zeros included, >=3 readings/hour).

## Known TODOs
- MASE scale currently averaged across origins per series; consider per-origin.
- Done 2026-09-24: intraday origins (00/09/13), series curation (80 series),
  statsforecast baselines, daily 112d runner, Tier-2 GBM, production
  backtest + stored-forecast fidelity check.
