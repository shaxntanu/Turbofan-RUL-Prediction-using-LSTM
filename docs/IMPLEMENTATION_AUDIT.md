# Implementation Audit

**Reviewed:** 2026-10-11  
**Branch:** `main`  
**Scope:** source implementation, completed local runs, generated dashboard bundle, and documentation. This audit distinguishes executed work from items not verified here.

## Status

| Area | Status | Evidence / boundary |
|---|---|---|
| Dataset loading and validation | Implemented and executed | NASA C-MAPSS FD001-FD004 train, test, and RUL files loaded in the completed runs. |
| Targets, preprocessing, and windows | Implemented and executed | Capped RUL labels, engine-level train/validation split, training-only sensor filtering/scaling, and 30-cycle windows. |
| Ridge baseline and LSTM | Implemented and executed | All four subsets trained and compared; saved local artifacts are under ignored `models/`. |
| Model selection and test evaluation | Implemented and executed | Selection uses validation MAE; selected models are then scored on held-out test labels. See [RESULTS.md](RESULTS.md). |
| Research adaptations | Implemented and executed | PCA/sensor screening, trend and fractional-life analyses, engine robustness, conformal intervals, and Ridge source/pooled transfer. |
| Matplotlib reports | Implemented and executed | Per-subset figures are generated under ignored `outputs/`; the dashboard’s primary result figures are committed under `web/public/results/`. |
| Streamlit application | Implemented | Interactive subset exploration, baseline and LSTM run controls, saved result display, and uploaded-window inference. |
| React/Vite dashboard | Implemented | Overview, subset/model comparison, transfer matrix, methodology, and figure views use the committed benchmark bundle. |
| Archify workflow | Delivered and visually checked | `docs/turbofan-pipeline-showcase.workflow.json` validates; `docs/turbofan-pipeline.html` passed the Archify showcase and viewport checks. |
| Python syntax | Verified | `compileall` is run as a syntax check; it is not a test suite. |
| Automated tests | Not run | Unit and integration tests were not run for this delivery. |
| C++ inference / latency benchmark | Not implemented | Not part of the delivered Python/Streamlit/web workflow; no C++ runtime claim is made. |
| LaTeX manuscript | Intentionally omitted | Per request, no `.tex` file was created. |

## Measured Run

The published snapshot uses seed 42, 30-cycle windows, a 125-cycle RUL cap, and a 20-epoch maximum for LSTM runs, with early stopping and restored best validation weights. Python 3.12.10 and TensorFlow 2.21.0 ran on CPU. The full per-model MAE/RMSE comparison, prediction intervals, research summaries, and caveats are in [RESULTS.md](RESULTS.md) and `web/public/results/benchmark.json`.

The model selected by validation MAE was LSTM for FD001 and Ridge for FD002-FD004. LSTM did have a lower test MAE than Ridge on FD003, but Ridge remains the reported selection because its validation MAE was lower. No choice was made from test performance.

## Limitations

- Reported test metrics aggregate eligible 30-cycle windows from official test trajectories; they are not the one-terminal-prediction-per-engine PHM08 leaderboard protocol.
- FD001-FD004 transfer is a simulated proxy-domain study, not transfer between independent operators or real fleets.
- Fractional-life stages approximate maintenance phases; C-MAPSS has no overhaul records. Test RUL is used to assign evaluation windows to those retrospective phases.
- Split-conformal coverage is measured on the same held-out test windows and is descriptive, not a deployment guarantee.
- This is a single seeded training run with a 20-epoch ceiling, not a hyperparameter search or a claim of state-of-the-art performance.
- `models/` and `outputs/` are git-ignored. Re-run the documented pipeline to recreate model binaries and detailed local reports; the web benchmark JSON and selected figures are versioned.

## Reproduction

From the repository root with the pinned Python dependencies installed:

```powershell
python -m src.train_all --epochs 20
python -c "from src.config import ProjectConfig; from src.transfer_analysis import run_transfer_analysis; run_transfer_analysis(config=ProjectConfig(dataset='FD001'), train_lstm=False)"
python scripts/export_web_data.py
```

The second command runs the Ridge transfer study independently. The web dashboard has its own `web/package-lock.json`; use `npm ci` and `npm run build` from `web/` for a production build.
