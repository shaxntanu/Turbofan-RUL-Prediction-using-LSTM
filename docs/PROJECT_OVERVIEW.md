# Project definition and requirements

This document defines the current scope of the Turbofan Remaining Useful Life Observatory. It supersedes prospective acceptance criteria in the original [`PROJECT_REQUIREMENTS.md`](PROJECT_REQUIREMENTS.md) where they differ; that file remains as historical planning context.

## Identity and purpose

| Field | Current definition |
|---|---|
| Project name | Turbofan Remaining Useful Life Observatory |
| Problem | Estimate engine remaining useful life from multivariate run-to-failure trajectories. |
| Data | NASA C-MAPSS simulated turbofan engine subsets FD001-FD004, bundled in `Ref/CMAPSSData/`. |
| Primary users | Students, researchers, project evaluators, and maintainers exploring prognostics methods. |
| Form | Research-oriented Python pipeline with Streamlit and static React/Vite result applications. |
| Maturity | Reproducible benchmark/demo prototype; not a validated operational maintenance system. |

## Motivation and research boundary

RUL estimates can support maintenance planning, but this repository evaluates models on a controlled simulator dataset. It addresses an engineering need for an inspectable, repeatable comparison across all four C-MAPSS subsets, with engine-level splits, validation-based model selection, explicitly labeled evaluation units, and research outputs in one repository.

The supplied papers use proprietary engine-health records and predict turbine gas temperature. This project adapts selected evaluation ideas to simulated C-MAPSS RUL; it does not reproduce their data, targets, model results, or complete methods. The integration is not presented as a new RUL algorithm or a demonstrated research novelty. See [`RESEARCH_TASK_MATRIX.md`](RESEARCH_TASK_MATRIX.md) and [`RELIABILITY_AND_ROBUSTNESS.md`](RELIABILITY_AND_ROBUSTNESS.md).

## Objectives and evidence

| ID | Objective | Success evidence |
|---|---|---|
| O-01 | Load and validate the bundled four-subset C-MAPSS data. | Dataset files and schema checks; loader and validator tests. |
| O-02 | Build capped RUL targets and keep engines intact across fit and validation partitions. | Target/split implementation and tests; run metadata and protocol. |
| O-03 | Compare the Ridge baseline with an optional stacked LSTM without selecting on test labels. | Validation metrics, selected model, and held-out results in `docs/RESULTS.md` and exported artifacts. |
| O-04 | Report error, uncertainty, sensor, transfer, and reliability analyses with their evaluation limits. | `docs/RESULTS.md`, `docs/RELIABILITY_AND_ROBUSTNESS.md`, generated CSV/JSON/figures. |
| O-05 | Let users inspect the benchmark and run supported local analyses. | Streamlit and React/Vite applications; production build and application helper tests. |
| O-06 | Keep optional Julia and C++ tools isolated from core RUL inference. | Separate scenario script, optional C++ extension, and documented install/build commands. |
| O-07 | Let another maintainer understand and verify repository behavior. | Architecture, methodology, limitations, traceability, contributor guidance, and CI checks. |

Numerical thresholds are not set as universal acceptance criteria: C-MAPSS subsets differ, and the recorded results are benchmark observations rather than operational safety targets.

## Functional requirements

| Requirement | Current behavior |
|---|---|
| Load FD001-FD004 measurements and test labels. | Implemented; bundled files can be overridden with `CMAPSS_DATA_DIR`. |
| Validate schema and per-engine cycle order. | Implemented in `src/data_validator.py`. |
| Construct capped training/test RUL labels. | Implemented in `src/target_construction.py`. |
| Split by engine and fit preprocessing on training engines. | Implemented in the experiment pipeline. |
| Generate fixed-length per-engine sequences. | Implemented; the default history is 30 cycles. |
| Train Ridge and optionally LSTM; choose by validation MAE. | Implemented; TensorFlow is required only for LSTM training. |
| Evaluate and plot predictions and research analyses. | Implemented; outputs are written under ignored `outputs/` and `models/`. |
| Browse committed benchmark and reliability results. | Implemented in Streamlit and the static web dashboard. |
| Forecast uploaded sensor histories. | Implemented in the Streamlit app with the saved Ridge model; the upload forecast does not use the LSTM artifact. |
| Regenerate the maintenance what-if table. | Implemented as a Julia script; the apps read the committed scenario table. |
| Run native PCA/K-Means diagnostics. | Optional C++17/pybind11 extension; not part of RUL prediction. |

## Nonfunctional requirements

- **Evaluation integrity:** Split engines before fitting preprocessing; use validation data for model choice and calibration; report test results only after choices are fixed.
- **Reproducibility:** Record seeds and package versions; pin the principal Python and JavaScript dependencies; include the benchmark data and regeneration commands.
- **Portability:** Support the documented Python 3.12 training environment; allow a Ridge-only path when TensorFlow is unavailable.
- **Maintainability:** Keep data, models, analyses, interfaces, and optional native tooling in separate modules.
- **Honest reporting:** Identify simulator data, exploratory analyses, synthetic stress, window versus terminal metrics, and unverified deployment assumptions.
- **Data handling:** Do not treat uploaded or exported data as flight records or as a maintenance recommendation.

## Assumptions and constraints

- C-MAPSS trajectories approximate degradation and are generated by a simulator; no operational aircraft telemetry is included.
- A complete training comparison requires the supported TensorFlow/Python combination. Ridge and static dashboards need no TensorFlow runtime.
- Full reliability studies can be computationally expensive; optional model challengers and the C++/Julia tools have separate dependencies.
- The official benchmark test labels are available locally and support retrospective evaluation; they do not constitute an external validation set.
- Model weights and new experiment outputs are local by default and are ignored by Git. The web dashboard uses a committed export snapshot.
- Public application URLs are recorded in the README, but account settings and current hosted service state are outside this repository's source control.

## Explicitly out of scope

- Airworthiness, flight operations, maintenance decisions, service scheduling, or safety-critical deployment.
- Real-time sensor ingestion, an authenticated prediction API, aircraft integration, or fleet data storage.
- Claims of state-of-the-art accuracy, real-fleet generalization, measured societal impact, or exact reproduction of supplied papers.
- N-CMAPSS evaluation; its data are not bundled and the current CLI does not implement that dataset.
