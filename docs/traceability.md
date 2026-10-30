# Requirements traceability

This matrix links current requirements to project objectives, implementation, verification evidence, and status. A passing helper test or successful build only verifies the scope exercised by that check; it does not imply full end-to-end or field verification.

| Requirement ID | Requirement | Objective | Implementation | Verification | Evidence | Status |
|---|---|---|---|---|---|---|
| REQ-DATA-01 | Read C-MAPSS measurements and reject invalid schema/trajectory data. | O-01 | `src/data_loader.py`, `src/data_validator.py` | Unit tests cover schema and sequential-cycle validation; no clean full-dataset pipeline run was repeated in this audit. | `tests/test_data_loader.py`; bundled files under `Ref/CMAPSSData/` | Partially Verified |
| REQ-LABEL-01 | Construct training and test RUL labels, with the configured cap. | O-02 | `src/target_construction.py` | Unit tests check per-engine training and test label calculations. | `tests/test_data_loader.py` | Verified |
| REQ-SPLIT-01 | Split by engine and fit preprocessing on the fit partition only. | O-02 | `src/data_splitter.py`, `src/pipeline.py`, `src/normalization.py` | Code and recorded protocol inspected; no dedicated end-to-end leakage regression test. | `docs/methodology.md`; `docs/RESULTS.md` | Partially Verified |
| REQ-MODEL-01 | Compare Ridge and optional TensorFlow LSTM candidates. | O-03 | `src/baseline_model.py`, `src/lstm_model.py`, `src/pipeline.py` | Wrapper/helper tests and a recorded benchmark snapshot; full training was not rerun in this audit. | `tests/test_research_extensions.py`; `docs/RESULTS.md` | Partially Verified |
| REQ-EVAL-01 | Select candidates using validation data and report named test metrics. | O-03 | `src/pipeline.py`, `src/evaluator.py`, `src/uncertainty.py` | Code and saved result protocol inspected; CI does not run a full model experiment. | `docs/RESULTS.md`; `docs/methodology.md` | Partially Verified |
| REQ-RESEARCH-01 | Provide reliability, sensor, transfer, and robustness analyses. | O-04 | `src/reliability_analysis.py`, `src/research_analysis.py`, `src/transfer_analysis.py` | Helper tests pass. The four committed subset snapshots report run `20261018-150917` and available data; the study was not rerun during this audit. | `tests/test_reliability.py`; `web/public/results/reliability/fd001.json`–`fd004.json`; `docs/RELIABILITY_AND_ROBUSTNESS.md` | Partially Verified |
| REQ-EXPORT-01 | Export portable static data for the web dashboard. | O-05 | `scripts/export_web_data.py` | Export tests pass. | `tests/test_export_web_data.py` | Verified |
| REQ-STREAMLIT-01 | Render combined-subset, reliability, video, and maintenance views. | O-05 | `src/streamlit_dashboard.py` | Unit tests cover data/table behavior; no hosted or browser-driven interaction test. | `tests/test_streamlit_dashboard.py` | Partially Verified |
| REQ-WEB-01 | Build the React/Vite static dashboard. | O-05 | `web/` | Local production build and hosted web build passed; CI also completed the production dependency audit for commit `04dacab`. | `npm run build` in `web/`; [hosted checks for `04dacab`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41082526123) | Verified |
| REQ-CMAKE-01 | Build and check optional C++ PCA/K-Means kernels. | O-06 | `cpp_engine/`, `src/cpp_bridge.py` | Native wheel built successfully; parity tests passed in the repository test run. | `pip wheel --no-deps ./cpp_engine`; `tests/test_cpp_engine.py` | Verified |
| REQ-JULIA-01 | Regenerate the maintenance what-if table from the committed input snapshot. | O-06 | `scripts/maintenance_what_if.jl` | Hosted Julia workflow regenerated the scenario table and confirmed it matches the committed CSV; local Julia is unavailable. | [Julia scenario verification for `6d99649`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41030696910); `.github/workflows/julia-what-if.yml` | Verified |
| REQ-DOC-01 | Explain scope, architecture, methods, limitations, and evidence to a new maintainer. | O-07 | `README.md`, `docs/` | Documentation links and consistency are reviewed in final audit. | `docs/README.md`; `docs/REPOSITORY_AUDIT.md` | Partially Verified |
| REQ-CI-01 | Run Python tests on supported Ridge/training runtimes and build the web app. | O-07 | `.github/workflows/ci.yml` | GitHub-hosted Python 3.12/3.14 and web jobs passed, including dependency scans and the Python 3.12 C++ wheel build. | [Hosted checks for `04dacab`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41082526123); `.github/workflows/ci.yml` | Verified |
| REQ-PROD-01 | Provide validated aircraft maintenance recommendations or production RUL service. | — | Not implemented | Out of scope; no operational data, system integration, or safety validation exists. | `docs/limitations.md` | Not Applicable |

## Interpretation

- **Verified** means the cited test or build ran successfully for the stated behavior in this audit.
- **Partially Verified** means only unit/helper coverage, code inspection, or a prior recorded result supports the requirement.
- **Untested** means the implementation exists but its stated verification has not run.
- **Planned** is reserved for work not implemented; it is not used to imply a commitment or delivery date.
- **Not Applicable** means the item is explicitly outside project scope.

The initial repository test run completed with **29 passed**. It emitted one expected scikit-learn convergence warning from a test that deliberately limits an MLP to three iterations. The new CI workflow was added after that baseline run and still needs its first hosted execution.
