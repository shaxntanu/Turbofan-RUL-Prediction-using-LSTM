# Methods and reproducibility

This document describes the implemented benchmark protocol. Its purpose is to make the recorded results interpretable and repeatable; it does not claim that one configuration is optimal for operational engines.

## Research questions

1. How do a window-summary Ridge baseline and an optional stacked LSTM compare on held-out engines in each C-MAPSS subset?
2. How do selected error and interval diagnostics change across simulated operating conditions, fault modes, history lengths, sensor corruption, and subset shift?
3. Which additional outputs can be regenerated and inspected using the repository's research and visualization tools?

The second and third questions are exploratory. The project does not use real fleet telemetry or an independent operational validation dataset.

## Dataset and labels

The repository includes `train_FD00x.txt`, `test_FD00x.txt`, and `RUL_FD00x.txt` for FD001-FD004 in `Ref/CMAPSSData/`. Each measurement row has 26 values: engine ID, cycle, three settings, and 21 sensors. FD001/FD003 use one simulated operating condition; FD002/FD004 vary operating conditions. FD003/FD004 include two simulated fault modes.

For a training row at cycle \(t\) on engine \(u\), the uncapped target is

\[
  RUL_{u,t} = T_u - t,
\]

where \(T_u\) is that training trajectory's final cycle. For a test row, the official final-cycle label \(r_u\) is added:

\[
  RUL_{u,t} = T^{obs}_u - t + r_u.
\]

The default configuration caps both targets at 125 cycles. The cap and 30-cycle input length are experimental settings, not universal physical constants.

## Partitioning and preprocessing

- Shuffle distinct training engine IDs with seed 42 and reserve 20% of the engines for validation; no engine contributes rows to both groups.
- Remove constant sensor channels based on the fit-engine rows only.
- Fit the configured MinMax scaler on fit-engine predictor columns (the three operational settings and retained sensors). Transform validation and official test values with that scaler.
- Keep cycles grouped and ordered by engine. Generate complete 30-cycle windows independently within each engine; assign each window the target at its last cycle.
- Do not fit preprocessing, select features, tune model variants, or calibrate intervals using official test labels.

This protocol's split is a single seeded partition. Its metrics do not estimate variation over every possible engine split.

## Models and selection

### Ridge baseline

`WindowRidgeBaseline` converts each input sequence to window-summary predictors and fits scikit-learn Ridge regression. It is the default deployable model in both the baseline experiment and the Streamlit uploaded-history forecast.

### LSTM

When TensorFlow is available and `train_lstm=True`, the pipeline trains the configured stacked 64/32-unit LSTM with dropout and a dense regression head. The default training configuration uses Adam, batches of 64, up to 100 epochs, and early stopping with a patience of 10 epochs; the published baseline snapshot used at most 20 epochs. The best validation weights are restored. TensorFlow is optional for the Ridge-only workflow.

Both candidates are scored on validation windows. The model with the lower validation MAE is selected. The already-separated official test data then provide the held-out metrics. The benchmark snapshot reports every candidate's test metrics to show comparison, but those scores do not choose the model.

## Metrics and uncertainty

- **MAE and RMSE:** Reported in RUL cycles at the named evaluation unit.
- **PHM08 asymmetric penalty:** Reported as an explicitly defined sum; the window-level sum is not the official per-engine challenge aggregation.
- **Terminal-engine evaluation:** Reliability runs retain one eligible final prediction per engine and report it separately from overlapping-window results.
- **Split-conformal intervals:** Use the empirical quantile of absolute validation residuals at 95% nominal coverage. Coverage and interval width on the test set are descriptive. Overlapping windows are dependent, and exchangeability or coverage under simulator/domain shift is not established.
- **Robustness summaries:** Include repeated Ridge seeds, single-seed optional challenger comparisons, synthetic sensor corruptions, short histories, capped-label onset proxies, and leave-one-subset-out evaluation. Each protocol's assumptions are documented in [`RELIABILITY_AND_ROBUSTNESS.md`](RELIABILITY_AND_ROBUSTNESS.md).

## Research-paper adaptations

The supplied papers study different targets and proprietary engine-health records. Implemented adaptations include sensor/PCA screening, polynomial trend and fractional-life summaries, model comparisons, prediction intervals, transfer tables, and error analyses. The repository does not claim to reproduce the papers' dataset, complete models, or numerical findings. Details and the novelty boundary are recorded in [`RESEARCH_TASK_MATRIX.md`](RESEARCH_TASK_MATRIX.md) and [`RELIABILITY_AND_ROBUSTNESS.md`](RELIABILITY_AND_ROBUSTNESS.md).

## Tools outside the primary RUL model

- **Matplotlib / Seaborn:** Generate per-subset sensor, prediction, error, interval, and research figures under `outputs/`.
- **Julia:** Recompute the fixed maintenance what-if scenario table from `data/maintenance_what_if_inputs.csv`. Its capacity and urgency cutoffs are analyst-selected assumptions; no cost, repair time, downtime, or post-service outcomes are modeled.
- **C++17 / pybind11:** Compare PCA and K-Means sensor-diagnostic kernels with scikit-learn. This experiment is not an RUL model or inference speedup claim.
- **ManimGx:** Source for a methods animation based on a measured sample trace. The bundled MP4 can be played without the renderer; its schematic model output is not a sample prediction.

## Reproduce the core results

Use Python 3.12 for the full training stack. See the root README for environment setup. From the repository root:

```powershell
python -m src.train_all --datasets FD001 FD002 FD003 FD004 --epochs 20
python scripts/export_web_data.py
```

For the optional repeated robustness study:

```powershell
python -m src.reliability_analysis --datasets FD001 FD002 FD003 FD004 --seeds 42 43 44 --lstm-seeds 42 --extended-models --with-cnn --epochs 20
```

The full robustness command can take substantially longer than the test suite. It creates local outputs and does not update the committed web snapshot until the export script is run. For the recorded protocols and values, see [`RESULTS.md`](RESULTS.md).
