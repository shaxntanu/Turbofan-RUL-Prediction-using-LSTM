# Experimental Results

**Snapshot:** 2026-10-11  
**Run IDs:** FD001 `20261011-161939`; FD002 `20261011-162103`; FD003 `20261011-162613`; FD004 `20261011-162754`.  
The downloadable dashboard data is [`web/public/results/benchmark.json`](../web/public/results/benchmark.json); detailed local JSON, CSV, model, and report artifacts are regenerated under ignored `outputs/` and `models/` directories.

## Protocol

- Seed 42; RUL capped at 125 cycles; 30-cycle input windows.
- Each training subset is split by engine, 80% for fitting and 20% for validation. Sensor filtering and MinMax scaling are fit on training engines only.
- Ridge regression over window summary features is compared with a stacked 64/32-unit LSTM. LSTM training uses Adam, batch size 64, at most 20 epochs, early stopping on validation MAE, and restores the best validation weights.
- The selected model is the one with lower validation MAE. Official test labels are not used to fit models or select the benchmark model. They are used for test targets and the explicitly retrospective fractional-life stratification described below.
- Test MAE/RMSE below aggregate every eligible sliding window from the official test trajectories; the first 29 cycles of each trajectory do not form a complete window. These are window-level results, not one terminal prediction per engine or the official per-engine PHM08 leaderboard score.
- The run used Python 3.12.10 and TensorFlow 2.21.0 on CPU. This is one seeded run, not a hyperparameter sweep.

## Model Comparison

All errors are cycles. Test-window counts are FD001 10,196; FD002 26,505; FD003 13,696; FD004 34,081.

| Subset | Ridge validation MAE | LSTM validation MAE | Ridge test MAE | LSTM test MAE | Selected | Selected test MAE / RMSE |
|---|---:|---:|---:|---:|---|---:|
| FD001 | 12.80 | 10.54 | 12.64 | 10.31 | LSTM | 10.31 / 13.94 |
| FD002 | 16.10 | 19.80 | 15.39 | 18.38 | Ridge | 15.39 / 19.36 |
| FD003 | 12.26 | 13.13 | 11.25 | 9.36 | Ridge | 11.25 / 15.57 |
| FD004 | 14.56 | 16.83 | 13.08 | 13.86 | Ridge | 13.08 / 18.00 |

FD003 is a useful reminder of the selection boundary: the LSTM has a lower test MAE there, but Ridge remains selected because it performed better on validation engines. No model was chosen by inspecting test performance. Across these runs, the LSTM is selected only for FD001; the baseline is stronger on validation for the other three subsets.

## Research Analyses

### Sensor Screening

PCA is fit to standardized retained training sensors; Pearson correlations with training RUL are descriptive associations, not causal importance.

| Subset | Retained sensors | Components for at least 95% variance | Variance at threshold | Highest absolute sensor/RUL correlation |
|---|---:|---:|---:|---|
| FD001 | 15 | 10 | 95.18% | sensor_11 |
| FD002 | 21 | 2 | 97.46% | sensor_16 |
| FD003 | 16 | 8 | 95.23% | sensor_11 |
| FD004 | 21 | 2 | 97.47% | sensor_14 |

### Fractional-Life Analysis

Each cell shows `shared Ridge MAE -> phase-specialized Ridge MAE`, in cycles. Test RUL and observed trajectory length are used retrospectively to assign windows to life thirds; this is not a deployable phase classifier. The specialized model has fewer training windows and is fit only on the corresponding training phase.

| Subset | Early | Middle | Late |
|---|---:|---:|---:|
| FD001 | 9.55 -> 2.72 | 14.34 -> 14.50 | 15.23 -> 8.71 |
| FD002 | 13.30 -> 2.41 | 17.10 -> 16.69 | 15.53 -> 11.93 |
| FD003 | 8.28 -> 1.27 | 12.21 -> 10.62 | 15.25 -> 12.76 |
| FD004 | 8.71 -> 1.19 | 13.85 -> 11.07 | 21.05 -> 16.61 |

The phase-specific comparison is exploratory: the early-life errors are small partly because RUL is capped, and stage membership depends on retrospective full-life information unavailable during online inference. C-MAPSS contains no overhaul timestamps, so these thirds are only a proxy for maintenance phases.

### Cross-Subset Ridge Transfer

Entries are target test MAE in cycles; rows are target subsets and columns are the training source. `pooled` is trained on the training engines from all four subsets.

| Target | FD001 source | FD002 source | FD003 source | FD004 source | Pooled |
|---|---:|---:|---:|---:|---:|
| FD001 | 12.64 | 13.19 | 15.94 | 19.41 | 15.66 |
| FD002 | 40.89 | 15.39 | 22.39 | 17.34 | 16.60 |
| FD003 | 15.61 | 12.78 | 11.25 | 14.75 | 10.98 |
| FD004 | 37.24 | 11.62 | 15.26 | 13.08 | 12.04 |

Source-specific scaling is retained, without refitting on target test data. The pooled model sees training data from each target subset, so pooled results are not zero-shot transfer to an unseen domain. All FD subsets are simulated variants of C-MAPSS, not independent real-world fleets.

### Optional Native Sensor Kernels

The C++17/pybind11 PCA and K-Means kernels were compared with scikit-learn using a seeded sample of 10,000 standardized training rows per subset. PCA timings include sklearn `fit_transform` and the equivalent native fit plus score projection; K-Means uses shared sklearn PCA scores, `k=4`, and ten initializations. Timings are best-of-three after one warm-up on this Windows/MSVC/OpenMP host.

| Subset | PCA sklearn (ms) | PCA C++ (ms) | Max PCA subspace angle | K-Means speedup | Adjusted Rand index |
|---|---:|---:|---:|---:|---:|
| FD001 | 8.73 | 13.62 | 0 deg | 1.21x | 0.994 |
| FD002 | 7.61 | 15.12 | 0 deg | 3.46x | 1.000 |
| FD003 | 5.25 | 16.43 | 0 deg | 1.74x | 0.996 |
| FD004 | 7.74 | 12.00 | 0 deg | 3.11x | 1.000 |

The PCA component subspaces match to numerical precision, but this native PCA implementation was slower on this host. The K-Means speedups are local microbenchmark results, not RUL or LSTM inference gains. Full per-run measurements and compiler metadata are in [`benchmark.json`](../web/public/results/benchmark.json).

### Prediction Intervals and Engine Robustness

Two-sided 95% split-conformal intervals are calibrated from validation residuals. Coverage and mean interval width are measured over test windows.

| Subset | Test coverage | Mean interval width (cycles) | Test engines flagged as MAE outliers* |
|---|---:|---:|---:|
| FD001 | 95.70% | 45.91 | 7 / 100 |
| FD002 | 95.97% | 60.25 | 2 / 237 |
| FD003 | 95.93% | 49.01 | 0 / 100 |
| FD004 | 95.66% | 52.98 | 2 / 237 |

*Outliers use the 1.5-IQR Tukey fence over per-engine MAE for engines with at least one complete window. Test coverage is descriptive and does not guarantee future or operational coverage.

## Reliability and Robustness Extension

**Run:** `20261018-150917`. Ridge results use seeds 42, 43, and 44; LSTM, MLP, and CNN challengers use seed 42 only. The table reports terminal-engine test MAE in cycles. Ridge entries are mean +/- sample standard deviation across seeds; each challenger is one run and is not a variance estimate.

| Subset | Ridge clean (3 seeds) | LSTM (1 seed) | MLP summary (1 seed) | Causal CNN (1 seed) |
|---|---:|---:|---:|---:|
| FD001 | 11.52 +/- 0.01 | 9.92 | 9.74 | 14.97 |
| FD002 | 13.31 +/- 0.03 | 15.85 | 11.87 | 15.96 |
| FD003 | 12.89 +/- 0.10 | 14.68 | 10.88 | 16.59 |
| FD004 | 16.50 +/- 0.14 | 20.37 | 12.87 | 21.01 |

The MLP results are an exploratory one-seed comparison, not evidence of a stable winner. Candidate selection uses validation-engine windows; official test labels are reserved for final scoring. XGBoost was requested but unavailable because the optional `xgboost` package was not installed. The CNN completed for all four subsets.

### Leave-One-Subset-Out Transfer

Each model is trained and calibrated using the other three C-MAPSS subsets, without training on the target subset. Results are terminal-engine test MAE in cycles; the LSTM uses one seed.

| Held-out subset | Ridge | LSTM |
|---|---:|---:|
| FD001 | 13.50 | 11.69 |
| FD002 | 15.48 | 13.60 |
| FD003 | 14.51 | 12.24 |
| FD004 | 17.79 | 20.08 |

These rows are not directly comparable to the baseline's overlapping-window metrics. FD001-FD004 are related simulator subsets, so this evaluates subset shift rather than transfer to independent fleets.

### Operational Sensitivity

Training a model specifically for ten observed cycles produced lower terminal MAE than truncating the normal 30-cycle model at inference in every subset: trained-versus-truncated MAE was FD001 15.57 vs. 22.93, FD002 15.02 vs. 21.00, FD003 14.97 vs. 22.16, and FD004 18.76 vs. 28.18. At a 30-cycle history, the corresponding trained-history Ridge terminal MAEs were 11.54, 13.33, 12.96, and 16.59.

Changing the RUL cap from 125 to 130 cycles increased fixed-Ridge terminal MAE on all four subsets: 11.54 to 12.12 (FD001), 13.33 to 14.11 (FD002), 12.96 to 13.76 (FD003), and 16.59 to 17.55 (FD004). The engine split and feature/scaling pipeline were held fixed.

The severe combined synthetic stress condition averaged two corruption scenarios. Ridge terminal MAE rose from clean values of 11.54, 13.33, 12.96, and 16.59 cycles to 37.64, 58.01, 32.19, and 57.20 cycles for FD001-FD004, respectively. These are controlled stressors, not a model of empirical failure prevalence.

The onset detector achieved 100% engine detection on its validation split, with false-alarm fractions from 8.0% to 45.0% and median detection delays from 11.5 to 51 cycles across subsets. Its threshold was selected on those same validation engines, so these values are tuning diagnostics, not independent performance estimates.

The run manifest and full aggregate tables are in the local `outputs/reliability/metrics/` directory; the dashboard export contains only the fields required by the web reliability view. Split-conformal coverage is descriptive under shift, corruption is synthetic, the FD subsets are related simulator domains, and N-CMAPSS was not run.

## Figures

| Subset | Predicted vs. actual | Fractional-life comparison | Prediction intervals |
|---|---|---|---|
| FD001 | [PNG](../web/public/results/fd001-predicted-vs-actual.png) | [PNG](../web/public/results/fd001-fractional-life-metrics.png) | [PNG](../web/public/results/fd001-prediction-intervals.png) |
| FD002 | [PNG](../web/public/results/fd002-predicted-vs-actual.png) | [PNG](../web/public/results/fd002-fractional-life-metrics.png) | [PNG](../web/public/results/fd002-prediction-intervals.png) |
| FD003 | [PNG](../web/public/results/fd003-predicted-vs-actual.png) | [PNG](../web/public/results/fd003-fractional-life-metrics.png) | [PNG](../web/public/results/fd003-prediction-intervals.png) |
| FD004 | [PNG](../web/public/results/fd004-predicted-vs-actual.png) | [PNG](../web/public/results/fd004-fractional-life-metrics.png) | [PNG](../web/public/results/fd004-prediction-intervals.png) |

The papers supplied with the repository predict turbine gas temperature using proprietary engine-health data. The C-MAPSS work here adapts selected methods to RUL prediction; it does not reproduce their data, target, or published numerical results.
