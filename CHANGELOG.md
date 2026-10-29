# Changelog

This file records significant user-facing and reproducibility changes. Experiment-specific measurements remain versioned in [`docs/RESULTS.md`](docs/RESULTS.md); Git history provides file-level change details.

## Unreleased

### Added

- Python 3.12/3.14 tests, optional native-extension build, and web production build in GitHub Actions.
- A development dependency manifest that keeps pytest out of the deployed app requirements.
- Current architecture, project scope, methodology, benchmarks, limitations, traceability, SDG assessment, and roadmap documentation.
- Contributor, security-reporting, code-of-conduct, and citation guidance.

### Clarified

- The project is a simulator-based research/demo prototype; the results are not operational maintenance guidance.
- The Streamlit upload forecast uses Ridge, while TensorFlow LSTM training is optional.
- Included dataset and paper files have source-specific terms distinct from the MIT license for repository software.
