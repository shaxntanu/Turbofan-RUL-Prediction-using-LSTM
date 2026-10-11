# Reliability and Robustness Protocol

This document describes the implemented reliability study. The original benchmark remains in [RESULTS.md](RESULTS.md); this extension is run separately from the baseline pipeline and writes its outputs under `outputs/reliability/`.

## Reproduce

The default repeated study uses three seeds for the Ridge-family baselines and seed 42 for the LSTM. Extended model comparisons, the short-history protocols, online onset analysis, and the RUL-cap comparison run on the first seed. Leave-one-subset-out evaluation uses the first seed and trains only on the other selected subsets.

```powershell
python -m src.reliability_analysis --datasets FD001 FD002 FD003 FD004 --seeds 42 43 44 --lstm-seeds 42 --extended-models --with-cnn --epochs 20
```

Add `--with-xgboost` only after installing the optional `xgboost` package. The run records an unavailable-model status when that optional dependency is missing; it does not silently substitute another estimator. `--without-lstm` provides a lighter CPU run. Outputs include configuration JSON, per-seed and mean/standard-deviation metrics, predictions, CSV summaries, and Matplotlib figures. Generated outputs and fitted models are intentionally not committed.

## Protocol

- Split training trajectories by engine before making overlapping windows. Fit constant-channel filtering and scaling on training engines only; keep validation engines separate for model selection and calibration.
- Use capped RUL labels with the configured 125-cycle cap. The 125-versus-130 sensitivity uses the same engine split and a fixed Ridge pipeline; the test results are descriptive and do not choose the cap.
- Report terminal per-engine predictions separately from the eligible overlapping test-window predictions. PHM08 penalties are reported on these explicitly named units of aggregation; a window-sum is not the official per-engine challenge score.
- Select model variants, asymmetric-loss weights, EMA settings, and onset thresholds using validation data only. The official test RUL is used only after predictions are fixed, to compute metrics.
- Construct validation pseudo-censored endpoints deterministically from validation trajectories for terminal-style calibration and decision thresholds. Their target RUL values are used for validation, never for test-time decisions.

## Implemented Extensions

### Model and training comparisons

The optional first-seed comparison includes training-only sensor ranking plus Ridge; an asymmetric squared-loss Ridge fitted with an iterative reweighted least-squares approximation; past-only exponential smoothing followed by Ridge; degree-two polynomial Ridge on summary features; a summary-feature MLP; and an optional causal 1-D CNN. The simpler Ridge augmentation ablations add training-feature-scaled noise and/or channel dropout while retaining the original RUL targets. Settings are selected on validation metrics, not official test scores. The MLP may reach its iteration budget without convergence; the per-model selection JSON records that status. This is a comparison harness, not a reproduction of every architecture in the supplied papers. Bi-cLSTM is not implemented as an exact paper reproduction.

### Online onset and partial history

The onset gate consumes sensor-window summaries only. Its training/validation event label is a proxy: the first cycle whose *capped* RUL is below the configured cap. It is not an observed physical fault onset or an unsupervised change point. The gate threshold is selected on validation windows using the PHM08 penalty sum, then MAE as a tie-break. Test decisions do not read test RUL or future sensor cycles; test labels are used only to score the resulting decisions. Detection delay and false-alarm summaries therefore describe this capped-label proxy task.

Short-history results compare (1) a Ridge model trained at each shorter history length and (2) the normal-window model evaluated on a truncated history. Fixed-shape windows repeat the earliest observed row on the left. No future samples are used. These are separate protocols and should not be collapsed into a single “short-window” score.

### Stress, selective prediction, and transfer

Sensor stress includes channel/group dropout, stuck values, drift, additive noise, missing blocks with training-median imputation, reduced sampling, and selected combined corruptions. Severity parameters are recorded in `configuration.json`; corruptions preserve the target when a test target exists. These are controlled synthetic interventions, not estimated field failure rates.

Split-conformal absolute-residual intervals use validation engines for calibration. The study reports empirical test coverage and width, plus validation-threshold risk/coverage when abstaining on out-of-range sensor health scores. Coverage is not guaranteed under time dependence or domain shift; these are diagnostics, not deployment guarantees.

Leave-one-subset-out runs hold one FD subset out of fitting, scaling, and calibration and use the other selected subsets as source data. FD001-FD004 differ in simulated conditions and fault modes but are not independent aircraft fleets. A pooled model that includes the target subset is not a zero-shot result.

## Prior Work and Novelty Boundary

These extensions should not be described as a new RUL algorithm. Several ingredients already have direct C-MAPSS precedents:

- Sensor corruption and partial-sensor failure with imputation have been studied on C-MAPSS, including a 2026 FD001-focused framework ([PLOS ONE, 2026](https://doi.org/10.1371/journal.pone.0347312)). The stress suite here is a controlled evaluation, not a novel fault simulator.
- Cross-condition and cross-fault generalization is an established C-MAPSS topic; CDIG explicitly targets domain-invariant RUL prediction ([Reliability Engineering & System Safety, 2024](https://doi.org/10.1016/j.ress.2024.110606)). The project’s leave-one-subset-out Ridge/LSTM tests are a baseline transfer protocol, not a new domain-generalization method.
- Conformal intervals for RUL have already been evaluated on C-MAPSS, including split, normalized, non-exchangeable, and quantile methods ([Javanmardi & Hüllermeier, 2022](https://arxiv.org/abs/2212.14612)). This repository’s validation-residual intervals are a simple baseline and do not claim time-series coverage guarantees.
- A 2026 study systematically compares preprocessing and temporal design across all four C-MAPSS subsets ([Array, 2026](https://doi.org/10.1016/j.array.2026.100919)); recent architecture comparisons also cover Ridge, polynomial Ridge, XGBoost, CNN, and LSTM ([arXiv:2604.27234](https://arxiv.org/abs/2604.27234)). Those model families are not novel here.
- A 2026 CNN-GRU study explicitly frames a reliability-oriented framework around repeated-run robustness, generalization, interpretability, and all-four-subset evaluation ([Mechanical Systems and Signal Processing, 2026](https://doi.org/10.1016/j.ymssp.2026.114642)). This is especially close to the project’s evaluation framing and rules out claiming that the broad combination alone is novel.
- Early-warning and change detection on C-MAPSS are also active areas. A 2026 unsupervised LSTM-autoencoder study detects faults from persistent reconstruction error ([arXiv:2601.10269](https://arxiv.org/abs/2601.10269)). The supervised capped-RUL onset proxy in this repository is different, but should not be presented as a physical change-point detector.
- The supplied Bi-cLSTM paper proposes residual-corrected bidirectional LSTM RUL estimation ([arXiv:2603.00745](https://arxiv.org/abs/2603.00745)). Its method is not claimed as reproduced by the current implementation. The supplied papers that predict turbine gas temperature on proprietary health records are adapted only at the level of evaluation ideas; their targets, datasets, and reported results are not reproduced.

The plausible contribution is engineering and evaluation integration: engine-grouped leakage controls, both terminal and window-level reporting, validation-only selection/calibration, repeated runs, short-history and sensor-stress protocols, selective-risk summaries, and cross-subset holdout outputs in one inspectable pipeline. Whether this combination is a publishable novelty claim requires a systematic review and a broader, controlled comparison; no novelty or state-of-the-art claim is made here.

## Scope and Gaps

The included NASA [C-MAPSS repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) is a simulator benchmark. This CLI does not run N-CMAPSS: its data are not included, and no N-CMAPSS result is implied. Results should not be generalized to operational aircraft or used as maintenance advice without external validation, safety review, and fleet-specific calibration.
