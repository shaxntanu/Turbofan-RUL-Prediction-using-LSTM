# Runtime and platform compatibility

**Evidence reviewed:** 2026-11-02

This matrix separates local execution, CI verification, and unverified deployment settings. A successful build does not establish compatibility on every operating system or in every hosted account.

| Runtime or platform | Intended role | Status | Evidence | Notes |
|---|---|---|---|---|
| Python 3.12 | Full training, tests, Streamlit, and optional C++ extension | Verified locally; CI configured | Local Python 3.12.10 virtual environment; recent full Python suite and compile checks; CI matrix includes 3.12. | TensorFlow 2.21.0 is included on this path. The recorded model snapshot ran on CPU. |
| Python 3.14 | Ridge, tests, compilation, and Streamlit without LSTM training | CI Verified in the last recorded hosted run | The requirements marker excludes TensorFlow when Python is 3.14 or newer. CI has a Python 3.14 job. | TensorFlow 2.21 is not available under this manifest for 3.14. LSTM training is disabled; Ridge remains available. The current commit's hosted run was not retrieved during this audit because the GitHub API connection timed out. |
| Node.js 24 | Vite production build and static dashboard | Verified locally; CI configured | Local Node v24.19.0 and npm 11.17.0; Vite production build passed in this audit. CI installs Node 24 and runs npm ci, npm audit --omit=dev, and npm run build. | A hosted build verifies compilation, not the current Vercel deployment revision. |
| C++17 with pybind11 | Optional PCA and K-Means sensor analysis | Build Verified | The optional extension has a Python package build; past local Windows/MSVC/OpenMP and hosted Linux CI builds are documented in the repository audit and results. CI builds it on Python 3.12 only. | It is not part of RUL model training or inference. Exact compiler and CPU timings are host-specific. |
| Julia 1.10 | Regenerate maintenance what-if scenario CSV | CI Verified; locally unavailable | The Julia workflow installs Julia 1.10, runs scripts/maintenance_what_if.jl, and checks the committed CSV diff. Prior hosted verification is linked in the repository audit. | Julia was not found on this local machine. The hosted apps read the committed CSV and do not require Julia at runtime. |
| Docker Engine 28.5.1 | Local Streamlit container | Verified locally | On 2026-11-02 the image built locally and both Streamlit health and root endpoints returned HTTP 200. The repository audit records the 4.04 GB image size. | This is a local smoke check. It does not verify Streamlit Community Cloud settings or make the container a hardened production image. |
| Streamlit Community Cloud | Hosted Streamlit application | Unverified deployment | A prior direct check reached the service access/not-found page. The repository's Docker workflow verifies the container separately. | The public app's current configuration, Python version, secrets, and access state were not inspected. |
| Vercel | Static React results dashboard and bundled study pages | Build Verified; current deployed revision unverified | Local production build passed. The Vercel URL was checked in an earlier audit, but this push's deployment revision was not verified. | The web app consumes committed static result snapshots. |
| macOS, mobile operating systems, and other Linux distributions | Possible developer or viewer environments | Untested | No direct compatibility run is recorded for these targets. | Do not infer support from the Ubuntu CI job or browser layout inspection. |

## Canonical commands

- Python tests: python -m pytest -q
- Python compilation: python -m compileall -q src scripts tests
- Local Markdown targets: python scripts/check_docs.py
- Dashboard build: from web, run npm ci followed by npm run build
- Julia snapshot: julia scripts/maintenance_what_if.jl, then review the CSV diff
- Container smoke path: use the Docker workflow or the root README Docker commands

The GitHub Actions workflows are the canonical cross-environment checks. A green result applies only to the workflow jobs and commit that actually completed. The status for the newly pushed commits was not available during this documentation pass because GitHub API and connected GitHub requests timed out.

There is no hardware or firmware target in this repository, so board, sensor, PCB, and embedded runtime compatibility do not apply.
