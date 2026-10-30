# Implementation Audit

**Reviewed:** 2026-10-31
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
| Streamlit application | Implemented | Interactive subset exploration, baseline and LSTM run controls, saved result display, uploaded-window inference, and optional native-kernel benchmark. |
| React/Vite dashboard | Implemented | Overview, subset/model comparison, transfer matrix, methodology, figure views, and native-kernel parity use the committed benchmark bundle. |
| Archify workflow | Delivered and visually checked | `docs/turbofan-pipeline-showcase.workflow.json` validates; `docs/turbofan-pipeline.html` passed the Archify showcase and viewport checks. |
| Python syntax | Verified | `compileall` is run as a syntax check; it is not a test suite. |
| Automated tests and CI | Verified | GitHub Actions passed the Python 3.12 and 3.14 suites, C++ wheel build, dependency checks, and web production build for `560cf9d`. |
| Optional C++ sensor kernels | Implemented and executed | C++17/pybind11 PCA and K-Means compared with scikit-learn on seeded 10k-row training samples from all four subsets. PCA fit-plus-projection subspaces match; PCA was slower on this host, while K-Means ranged from 1.21x to 3.46x faster. These are local microbenchmarks, not RUL/LSTM inference speedups. |
| LaTeX manuscript | Intentionally omitted | Per request, no `.tex` file was created. |

## Measured Run

The refreshed snapshot uses seed 42, 30-cycle windows, a 125-cycle RUL cap, and a 20-epoch maximum for LSTM runs, with early stopping and restored best validation weights. The FD001-FD004 run IDs are `20261031-002409`, `20261031-002605`, `20261031-003010`, and `20261031-003157`. Python 3.12.10 and TensorFlow 2.21.0 ran on CPU. The full per-model MAE/RMSE comparison, prediction intervals, research summaries, and caveats are in [RESULTS.md](RESULTS.md); run `python scripts/export_web_data.py` to refresh the static dashboard bundle.

The model selected by validation MAE was LSTM for FD001 and Ridge for FD002-FD004. LSTM did have a lower test MAE than Ridge on FD003, but Ridge remains the reported selection because its validation MAE was lower. No choice was made from test performance.

The optional native module was built with MSVC and OpenMP. Its PCA principal subspaces matched scikit-learn on the sampled training rows, comparing fit plus score projection on both paths; the C++ covariance/Jacobi implementation was slower than sklearn's full-SVD PCA on this host. K-Means used shared sklearn PCA scores and had high label agreement (adjusted Rand index 0.994-1.000). Timings are machine/compiler-dependent; details are in `web/public/results/benchmark.json`.

## Latest Verification (2026-10-31)

- Fresh baseline training completed for all four subsets. Selection remains validation-based: LSTM for FD001 and Ridge for FD002-FD004. The measured test-window MAEs were 10.31, 15.39, 11.25, and 13.08 cycles, respectively.
- The reliability CLI completed run `20261030-231526` on FD001-FD004, including repeated Ridge seeds, held-out-subset evaluation, short-history protocols, synthetic sensor stress, risk coverage, onset detection, and RUL-cap sensitivity. XGBoost was unavailable and was recorded as optional rather than silently replaced.
- The GitHub Actions repository checks for `560cf9d` passed Python 3.12 and 3.14, dependency scans, the optional C++ wheel, and the Vite production build. The Docker workflow built the Streamlit image and received HTTP 200 from its health and root routes.
- All 16 Mermaid diagrams in the architecture and study documents rendered and passed SVG checks. The existing interactive conformal-calibration figure was reviewed at desktop/mobile sizes in light/dark themes and with keyboard controls; no duplicate figure was added.
- The Streamlit responsive-layout tests passed (7 tests). The public Streamlit URL remains inaccessible to this audit session, so hosted account/deployment settings and hosted interaction are not verified.

## Limitations

- Reported test metrics aggregate eligible 30-cycle windows from official test trajectories; they are not the one-terminal-prediction-per-engine PHM08 leaderboard protocol.
- FD001-FD004 transfer is a simulated proxy-domain study, not transfer between independent operators or real fleets.
- Fractional-life stages approximate maintenance phases; C-MAPSS has no overhaul records. Test RUL is used to assign evaluation windows to those retrospective phases.
- Split-conformal coverage is measured on the same held-out test windows and is descriptive, not a deployment guarantee.
- This is a single seeded training run with a 20-epoch ceiling, not a hyperparameter search or a claim of state-of-the-art performance.
- `models/` and `outputs/` are git-ignored. Re-run the documented pipeline to recreate model binaries and detailed local reports; the web benchmark JSON and selected figures are versioned.
- Local Windows Docker Desktop remains unavailable because its service fails before the engine pipe starts; the hosted image-build and HTTP smoke workflow passes. No local Docker Desktop reset or data deletion was performed.

## Reproduction

From the repository root with the pinned Python dependencies installed:

```powershell
python -m src.train_all --epochs 20
python -c "from src.config import ProjectConfig; from src.transfer_analysis import run_transfer_analysis; run_transfer_analysis(config=ProjectConfig(dataset='FD001'), train_lstm=False)"
python scripts/export_web_data.py
```

The second command runs the Ridge transfer study independently. The web dashboard has its own `web/package-lock.json`; use `npm ci` and `npm run build` from `web/` for a production build.
