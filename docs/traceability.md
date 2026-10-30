# Requirements traceability

This matrix links current requirements to project objectives, implementation, verification evidence, and status. A passing helper test or successful build only verifies the scope exercised by that check; it does not imply full end-to-end or field verification.

| Requirement ID | Requirement | Objective | Implementation | Verification | Evidence | Status |
|---|---|---|---|---|---|---|
| REQ-DATA-01 | Read C-MAPSS measurements and reject invalid schema/trajectory data. | O-01 | `src/data_loader.py`, `src/data_validator.py` | Unit tests cover schema and sequential-cycle validation; fresh FD001-FD004 training and reliability runs loaded the bundled domains successfully. | `tests/test_data_loader.py`; `docs/RESULTS.md`; bundled files under `Ref/CMAPSSData/` | Verified |
| REQ-LABEL-01 | Construct training and test RUL labels, with the configured cap. | O-02 | `src/target_construction.py` | Unit tests check per-engine training and test label calculations. | `tests/test_data_loader.py` | Verified |
| REQ-SPLIT-01 | Split by engine and fit preprocessing on the fit partition only. | O-02 | `src/data_splitter.py`, `src/pipeline.py`, `src/normalization.py` | Code and recorded protocol inspected; no dedicated end-to-end leakage regression test. | `docs/methodology.md`; `docs/RESULTS.md` | Partially Verified |
| REQ-MODEL-01 | Compare Ridge and optional TensorFlow LSTM candidates. | O-03 | `src/baseline_model.py`, `src/lstm_model.py`, `src/pipeline.py` | Fresh full training completed on FD001-FD004; the hosted Python 3.12 and 3.14 CI tests also passed. | `tests/test_research_extensions.py`; `docs/RESULTS.md` | Verified |
| REQ-EVAL-01 | Select candidates using validation data and report named test metrics. | O-03 | `src/pipeline.py`, `src/evaluator.py`, `src/uncertainty.py` | Fresh full-subset runs recorded validation-based model selection and official test metrics; the results document states the window-level scoring protocol. | `docs/RESULTS.md`; `docs/methodology.md` | Verified |
| REQ-RESEARCH-01 | Provide reliability, sensor, transfer, and robustness analyses. | O-04 | `src/reliability_analysis.py`, `src/research_analysis.py`, `src/transfer_analysis.py` | Fresh reliability CLI completed for all four subsets as run `20261030-231526`; refreshed tables cover repeated seeds, held-out subsets, sensor stress, risk coverage, short histories, onset, and cap sensitivity. | `tests/test_reliability.py`; `web/public/results/reliability/fd001.json`–`fd004.json`; `docs/RELIABILITY_AND_ROBUSTNESS.md`; `docs/RESULTS.md` | Verified |
| REQ-EXPORT-01 | Export portable static data for the web dashboard. | O-05 | `scripts/export_web_data.py` | Export tests pass. | `tests/test_export_web_data.py` | Verified |
| REQ-STREAMLIT-01 | Render combined-subset, reliability, video, and maintenance views. | O-05 | `src/streamlit_dashboard.py` | Unit tests cover data/table behavior; no hosted or browser-driven interaction test. | `tests/test_streamlit_dashboard.py` | Partially Verified |
| REQ-WEB-01 | Build the React/Vite static dashboard. | O-05 | `web/` | Hosted production build and production dependency audit passed for commit `560cf9d`. | [Repository checks for `560cf9d`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41108973945); `web/` | Verified |
| REQ-CMAKE-01 | Build and check optional C++ PCA/K-Means kernels. | O-06 | `cpp_engine/`, `src/cpp_bridge.py` | Native wheel built successfully; parity tests passed in the repository test run. | `pip wheel --no-deps ./cpp_engine`; `tests/test_cpp_engine.py` | Verified |
| REQ-JULIA-01 | Regenerate the maintenance what-if table from the committed input snapshot. | O-06 | `scripts/maintenance_what_if.jl` | Hosted Julia workflow regenerated the scenario table and confirmed it matches the committed CSV; local Julia is unavailable. | [Julia scenario verification for `6d99649`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41030696910); `.github/workflows/julia-what-if.yml` | Verified |
| REQ-DOC-01 | Explain scope, architecture, methods, limitations, and evidence to a new maintainer. | O-07 | `README.md`, `docs/` | Local documentation-link audit found no missing targets; all 16 Mermaid blocks rendered successfully. The current results and verification records are refreshed in this audit. | `docs/README.md`; `docs/REPOSITORY_AUDIT.md`; `docs/IMPLEMENTATION_AUDIT.md` | Verified |
| REQ-CI-01 | Run Python tests on supported Ridge/training runtimes and build the web app. | O-07 | `.github/workflows/ci.yml` | GitHub-hosted Python 3.12/3.14 and web jobs passed, including dependency scans and the Python 3.12 C++ wheel build. | [Repository checks for `560cf9d`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41108973945); `.github/workflows/ci.yml` | Verified |
| REQ-DOCKER-01 | Build the Streamlit image and verify its health and root routes. | O-07 | `Dockerfile`, `.dockerignore`, `.github/workflows/docker.yml` | Hosted Docker build and HTTP smoke checks passed; the local Windows Docker Engine remains unavailable. | [Docker image check for `0dc35d0`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41107892668); `.github/workflows/docker.yml` | Verified |
| REQ-PROD-01 | Provide validated aircraft maintenance recommendations or production RUL service. | — | Not implemented | Out of scope; no operational data, system integration, or safety validation exists. | `docs/limitations.md` | Not Applicable |

## Interpretation

- **Verified** means the cited test or build ran successfully for the stated behavior in this audit.
- **Partially Verified** means only unit/helper coverage, code inspection, or a prior recorded result supports the requirement.
- **Untested** means the implementation exists but its stated verification has not run.
- **Planned** is reserved for work not implemented; it is not used to imply a commitment or delivery date.
- **Not Applicable** means the item is explicitly outside project scope.

## Verification Update (2026-10-31)

- Fresh baseline runs completed for FD001-FD004 and are published in [`RESULTS.md`](RESULTS.md) and the exported dashboard bundle.
- Reliability run `20261030-231526` completed for all four subsets; the current snapshots are exported under `web/public/results/reliability/`.
- The hosted repository checks for `560cf9d` passed Python 3.12, Python 3.14, and the web production build. The Docker image workflow also built the image and received HTTP 200 from both Streamlit routes.
- All 16 Mermaid blocks across the architecture and study documents rendered successfully. The static calibration figure was reviewed at desktop/mobile sizes in both themes and with keyboard input.
- Hosted Streamlit access and local Windows Docker startup remain environment/deployment limitations; see [`REPOSITORY_AUDIT.md`](REPOSITORY_AUDIT.md).
