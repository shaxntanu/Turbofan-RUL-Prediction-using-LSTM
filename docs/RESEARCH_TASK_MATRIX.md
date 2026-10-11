# Research Task Matrix

This matrix records implemented adaptations of the supplied papers. The papers predict turbine gas temperature (TGT) using proprietary engine-health/fleet data; this repository predicts RUL on simulated NASA C-MAPSS trajectories. Matching methods does not reproduce their datasets or published numerical results.

## Paper 1: Predictive Engine-Health Methods

| Paper method | C-MAPSS adaptation | Implementation | Status | Scope note |
|---|---|---|---|---|
| Pearson/PCA feature screening | Training-only sensor association and PCA variance screening | `src/research_analysis.py` | Implemented | Correlation is descriptive, not causal. |
| Detrending and degradation phases | Per-sensor polynomial trend analysis and fractional-life bins | `src/detrending.py`, `src/segmentation_analysis.py` | Implemented | No overhaul/maintenance timestamps; fractional life is a proxy. |
| Comparative predictive modeling | Window-summary Ridge and stacked many-to-one LSTM | `src/baseline_model.py`, `src/lstm_model.py`, `src/pipeline.py` | Implemented | Selected by validation MAE before official test scoring. |
| Robustness and outliers | Per-engine absolute error summaries and Tukey outlier counts | `src/research_analysis.py` | Implemented | Small engine counts limit tail inference. |
| Prediction uncertainty | Validation-calibrated split-conformal intervals | `src/uncertainty.py` | Implemented | Test coverage is descriptive; not a deployment guarantee. |
| Matplotlib results | Predicted/actual, intervals, residuals, sensor and phase figures | `src/visualizer.py` | Implemented | Figures are generated under `outputs/`. |

## Paper 2: Generalisable Health Models

| Paper method | C-MAPSS adaptation | Implementation | Status | Scope note |
|---|---|---|---|---|
| Domain-specific models | Independent FD001-FD004 pipelines | `src/train_all.py`, `src/pipeline.py` | Implemented | Each subset has its own engine-level split and scaler. |
| Source-to-target transfer | Source-trained Ridge evaluated on each other subset without target scaler refitting | `src/transfer_analysis.py` | Implemented | Four related simulated subsets are proxy domains, not real fleets. |
| Pooled generalization | Train on pooled source subsets and score each held-out subset | `src/transfer_analysis.py` | Implemented | Shared simulated dataset provenance remains a limitation. |
| Transfer metrics | MAE/RMSE matrix and dashboard heatmap | `src/transfer_analysis.py`, `scripts/export_web_data.py`, `web/src/main.jsx` | Implemented | Window-level metrics; not a fleet-level safety claim. |

## Evaluation Boundary

The C-MAPSS official test labels are used only after selecting a model on the training-engine validation split. Detrending, PCA/correlation screening, conformal calibration, and life-stage evaluation are adaptations motivated by the papers, not exact replications of their TGT models or fleet workflows. The Streamlit and web applications consume local pipeline artifacts; the web dashboard uses a committed JSON/figure bundle refreshed with `scripts/export_web_data.py`.
