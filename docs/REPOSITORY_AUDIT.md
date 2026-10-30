# Repository audit

**Audit date:** 2026-10-30

**Initial revision:** `17bc488` (`main`, clean and aligned with `origin/main`)
**Implementation revision covered by this follow-up:** `4813ee4` (`main`, pushed to `origin/main`)

This report records local checks and the hosted CI and Vercel observations listed below. A successful push does not verify deployment settings or the Streamlit service.

## Repository Overview

The repository is a research-oriented turbofan Remaining Useful Life (RUL) benchmark and demonstration project. It combines a modular Python pipeline, NASA C-MAPSS FD001-FD004 data, an optional TensorFlow LSTM, a Ridge baseline, research/reliability analyses, Matplotlib outputs, Streamlit, a static React/Vite dashboard, an optional C++ sensor-kernel benchmark, a Julia what-if analysis, and a ManimGx animation source.

It is a software/research hybrid focused on simulated prognostics. It is not an embedded system, aircraft integration, or operational maintenance product. The root MIT license applies to project software; included data and papers have separate source-specific terms.

## Initial Condition

- The repository already had a clean, substantial implementation and 29 passing Python tests. It was not an empty scaffold and did not need a rewrite.
- The primary workflow, research results, and dashboard artifacts existed. Core experiment outputs and model weights were git-ignored; the web app used a committed static snapshot.
- Python dependencies were pinned, but pytest and the optional local C++ package were both in the app/runtime manifest. The Docker image and Streamlit system-package manifest also installed a compiler for that optional extension. A standalone development dependency file did not exist.
- GitHub Actions contained a Julia scenario-table check but no Python test, native-extension, or web-build CI.
- Documentation was extensive but split between current results and earlier prospective specifications. There was no single current architecture map, requirements-to-evidence matrix, maturity audit, or documentation index.
- The dataset specification made unsupported blanket reuse claims and cited a dataset description inconsistent with the bundled dataset readme.
- The repository had no contributor, security-reporting, code-of-conduct, changelog, citation, or issue/PR templates.
- The initial revision was `17bc488`, on `main`, clean and synced with `origin/main`; the local history contained 260 commits and no tags. This audit did not inspect GitHub Releases or account settings.

## Major Problems Found

1. Runtime and development dependencies were mixed, including pytest and an optional C++ build in the app manifest; the app container installed a compiler for it.
2. No automated CI exercised the Python tests, optional C++ build, or web production build.
3. Current architecture and historical/aspirational specifications were difficult to distinguish.
4. Requirements, implementation, verification, and remaining gaps were not mapped in one place.
5. Dataset license/usage assertions exceeded the evidence included in the repository.
6. No maintainer-facing security reporting or community contribution guidance existed.
7. The Streamlit upload forecast uses Ridge even when an LSTM comparison is available; this behavior was not prominent in the project documentation.
8. The npm and Python advisory sources could not be reached from the audit environment, leaving dependency vulnerability status unresolved.

No confirmed core model defect was found in the inspected pipeline, and the existing test suite passed. That is not proof of correctness for every full experiment, upload, or hosted deployment.

## Architecture Findings

- The Python pipeline is modular across configuration, data loading/validation, targets, engine splitting, preprocessing, sequence construction, models, evaluation, analyses, and visualization.
- The standard run separates fit and validation engines before preprocessing, selects the model on validation MAE, and reports official test metrics afterwards. The reliability study distinguishes terminal-engine from overlapping-window evaluation.
- The Streamlit app can run supported local experiments and uses saved Ridge artifacts for uploaded-history forecasts. The web app is a static results viewer; it fetches committed bundle data rather than calling an inference service.
- Julia, C++, and ManimGx serve separate analysis/diagnostic/media roles and are optional for core Ridge and static dashboard use.
- Generated `outputs/` and `models/` remain local; the static web export can become stale until regenerated and committed.

## Improvements Implemented

- Moved pytest out of the Streamlit runtime manifest into `requirements-dev.txt`.
- Removed the optional C++ package build from the app runtime manifest; CI and local contributors build it explicitly when needed.
- Removed the now-unneeded compiler installation from the Docker and Streamlit system-package configuration.
- Added read-only GitHub Actions CI for Python 3.12/3.14, C++ extension build on Python 3.12, and the Vite production build.
- Added current architecture, project-scope, methodology, benchmark, limitations, traceability, roadmap, SDG, and audit documentation.
- Added three retrospective ADRs and a documentation index.
- Added contributor setup, security-reporting, code-of-conduct, changelog, and software-citation files, plus issue and PR templates.
- Clarified historic planning documents and corrected dataset scope, citation, and third-party rights language without changing `LICENSE`.
- Updated the root README with current documentation and verification links; no contribution section was added.
- Kept the implementation and documentation changes in individual commits authored as Shantanu and pushed them to `origin/main`, following the user's explicit instruction. The latest hosted CI run and live Vercel check are recorded below.

## Files Added

- `.github/workflows/ci.yml`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `requirements-dev.txt`
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `CITATION.cff`
- `docs/README.md`, `docs/PROJECT_OVERVIEW.md`, `docs/architecture.md`, `docs/methodology.md`, `docs/benchmarks.md`, `docs/limitations.md`, `docs/traceability.md`, `docs/roadmap.md`, `docs/sdg-alignment.md`
- `docs/decisions/ADR-001-engine-level-validation.md`
- `docs/decisions/ADR-002-static-results-dashboard.md`
- `docs/decisions/ADR-003-isolate-optional-tooling.md`
- `docs/REPOSITORY_AUDIT.md`

## Files Modified

- `README.md`
- `requirements.txt`
- `docs/DATASET_AND_DATA_PIPELINE.md`
- `docs/PROJECT_REQUIREMENTS.md`
- `docs/SYSTEM_ARCHITECTURE.md`
- `docs/TECHNOLOGY_STACK.md`
- `Dockerfile`

## Files Removed

- `packages.txt` (it installed only `build-essential`, which the app no longer needs)

The underlying models, published metric snapshots, dataset, and license remain unchanged. Subsequent Methods-page work adds explanatory and media UI without changing how model outputs are produced.

## Subsequent Dashboard Integration

- The Vercel Methods page now embeds the data-linked split-conformal calibration explainer and a collapsed, captioned ManimGx video player.
- `docs/figures/build_conformal_calibration.mjs` generates the static SVG preview and the standalone interactive page bundled at `web/public/conformal-calibration.html`.
- The root README, methodology, and documentation index link to the figure and interactive page. The root README retains only the Vercel and Streamlit badges and has no Contributions section.

## Testing Performed

- `python -m pytest -q` using the repository Python 3.12.10 environment.
- `python -m compileall -q src scripts tests animations`.
- `npm run build` from `web/`.
- `python -m pip check`.
- Built the optional C++ wheel with `python -m pip wheel --no-deps --wheel-dir <temporary-directory> ./cpp_engine`.
- Scanned 78 relative Markdown links across 28 tracked Markdown files; every target resolved.
- Parsed the CI and Julia workflows, issue forms, and `CITATION.cff` as YAML and checked expected top-level fields.
- Inspected source for TODO/FIXME markers, dynamic evaluation, unsafe HTML insertion, subprocess/system calls, broad exception handling, and debug configuration.
- Scanned tracked text files for common credential and private-key patterns; no matches were reported.
- Reviewed CI workflow permissions and setup steps. `actionlint` was not installed, so GitHub Actions-specific semantics were not linted locally; the hosted workflow result is recorded below.
- Reviewed the Mermaid source in the current architecture diagrams; no Mermaid renderer/parser was available for rendered validation.

## Results at Initial Audit Pass

- Python suite: **29 passed**, one expected scikit-learn convergence warning from a test that deliberately limits an MLP to three iterations.
- Python compilation: passed.
- Web production build: passed; Vite emitted the expected `dist/` bundle.
- Dependency consistency: `pip check` reported no broken requirements.
- C++ package build: wheel built successfully for CPython 3.12 on Windows; the repository tests passed with the native module available.
- `npm audit --omit=dev`: inconclusive; the npm advisory endpoint reset the connection.
- `pip-audit --local`: inconclusive; the PyPI advisory request timed out. No dependency vulnerability-free claim is made.
- Julia scenario generation was not run because Julia is not installed in this environment. The existing workflow and Python scenario reconciliation tests remain in place.
- A clean Python 3.14 environment install was attempted, but package resolution stopped when the configured index could not provide `fonttools`; the 3.14 compatibility path therefore remains unverified locally.
- The Docker CLI was installed, but the Docker daemon was unavailable; the modified image was not built.
- The full four-subset training and reliability studies were not rerun. Browser checks cover the Vercel Methods and calibration views; the Streamlit deployment could not be accessed.

## Verification Update at `1759605` (2026-10-30)

- Repository `.venv` Python 3.12.10: `python -m pytest -q` — **29 passed**, with one expected scikit-learn convergence warning from the deliberately three-iteration MLP test; `compileall` and `pip check` passed.
- System Python 3.14.7: `pytest` — **27 passed, 2 skipped**, with the same expected convergence warning; `compileall` and `pip check` passed. `pip check` printed an ignored invalid `~treamlit` distribution warning in the user-level site-packages directory, then reported no broken requirements.
- `npm run build` passed. Vite emitted the MP4, poster, external VTT captions, interactive calibration page, and dashboard bundles.
- The latest GitHub Actions run, [Repository checks for `1759605`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41005793081), completed successfully: Python 3.12 tests/compile/C++ wheel, Python 3.14 tests/compile, and the Vite production build all passed. The optional C++ build is skipped on Python 3.14 by design.
- Browser inspection of the live Vercel dashboard confirmed that the Methods calibration iframe loads, reports FD001's saved q95 and observed coverage, and that the video plays with its caption visible. The standalone calibration page also loaded with working controls. Local interaction checks confirmed subset and light/dark synchronization, including FD004.
- Responsive review of the dashboard at 1440, 768, and 390 pixels found **0 FAIL**. The design inspector reported two warnings (remote IBM Plex font fallback in its inspection browser and a high share of small dashboard labels); its source lint reported no FAIL and 21 heuristic warnings for section kickers and the ordered method-step labels.
- The public Streamlit URL redirected to Streamlit's “You do not have access to this app or it does not exist” page. This session could not verify its hosted app or settings; the message does not distinguish access restrictions from a missing deployment.

## Verification Update at `6d99649` (2026-10-30)

- The repository Python 3.12 environment passed the full suite: **33 passed**, with one expected scikit-learn convergence warning from the deliberately three-iteration MLP test. `compileall` and `npm run build` also passed.
- [Repository checks for `6d99649`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41030664857) passed all three jobs: Python 3.12 tests/compile/C++ wheel, Python 3.14 tests/compile, and the Vite production build.
- [Julia scenario verification for `6d99649`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41030696910) installed Julia, regenerated the scenario table, and confirmed it matched the committed CSV. Julia is unavailable in the local environment, so this verification was hosted.
- A bounded end-to-end Ridge smoke test and uploaded-history forecast contract test are now part of the Python suite. The latter covers case-insensitive RUL input, cycle metadata, and bounded prediction intervals.
- The deployed results fetch issue was traced to the original 3.67 MB `benchmark.json`, which included all reliability tables. The main snapshot is now about 45 KB; four reliability snapshots load on demand by subset. Streamlit merges those snapshots into its expected in-memory bundle. The live Vercel dashboard and Reliability view loaded the four subsets and FD001 reliability metrics.
- The standalone split-conformal calibration page exposes working subset and illustrative-prediction controls and reports saved benchmark calibration metrics. Its interactive figure is generated from project-owned source; the diagram's rank pegs are schematic, while q95 and observed coverage come from the selected stored run.
- The production dashboard passed the responsive design inspector at 1440, 768, and 390 pixels with **0 FAIL**. It reported one remaining heuristic warning: a high share of small secondary labels in this dense research dashboard.
## Verification Update at `4813ee4` (2026-10-30)

- Reinstalled the declared Python dependencies and reran the full Python suite: **33 passed**, with one expected scikit-learn convergence warning from the deliberately three-iteration MLP test. `compileall`, `pip check`, and the Vite production build passed.
- [Repository checks for `4813ee4`](https://github.com/shaxntanu/Turbofan-RUL-Prediction-using-LSTM/actions/runs/41032865270) passed Python 3.12 tests/compile/C++ wheel, Python 3.14 tests/compile, and the Vite production build.
- The advisory review identified and patched the repo's declared dependencies: Streamlit `1.66.1` supports Pillow `13.0.0`; certifi is pinned at `2026.10.23`; pytest is pinned at `9.0.3`; and the C++ build backend requires setuptools `>=84.1.0`. These versions address the reviewed [Pillow advisory](https://github.com/advisories/GHSA-5x8w-8jj5-q52h), [certifi advisory](https://github.com/advisories/GHSA-6q48-6c7w-9gw2), [pytest security fix](https://github.com/pytest-dev/pytest/commit/95d8423bd24992deea5b9df32555fa1741679e2c), and [setuptools advisory](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w).
- `pip-audit` scanned 95 installed Python packages after the updates and reported **0 known vulnerabilities**. The npm production-dependency scan (`npm audit --omit=dev`) also reported **0 vulnerabilities**. These are point-in-time checks, not continuous monitoring.
- The local C++ wheel was produced after installing patched setuptools `84.2.1`; the hosted CI wheel job also passed. A first local isolated-build attempt timed out fetching build dependencies from PyPI, so the local retry used the already installed, patched build backend.

## Security Findings

- No common API-key, token, or private-key patterns were found in the tracked text scan. The scan is not a complete secret-detection guarantee.
- The new CI workflow requests `contents: read`, does not publish packages/releases, and has no deployment credentials.
- Streamlit parses user-uploaded CSV/TXT in memory. The app is intended for trusted demonstration use and does not provide a hardened multi-tenant boundary.
- The static dashboard has no prediction API or account system. Hosted service settings and secrets were not inspected.
- No license was changed. Dataset and research papers are documented as third-party material with source-specific terms.
- Point-in-time Python and npm advisory scans are clear after the reviewed dependency updates; no automated recurring advisory scanner is configured.

## Dependency Findings

- Direct Python, web, and animation dependencies are separated across manifests; `requirements.txt` now excludes pytest and the optional C++ build.
- `pip check` found no installed dependency conflicts.
- The optional C++ package wheel built successfully. The repository root is an application/research project rather than a distributable Python package, so no root package metadata was added.
- The web application has a lockfile. Python dependencies are directly pinned but lack a full transitive lockfile.
- No SBOM is checked in. There is no root release/package build, and the project spans separate deployed, training, animation, native, and web dependency environments. A generated environment snapshot would not stay current. Revisit CycloneDX/SPDX generation if a release or artifact-distribution process is introduced.
- Point-in-time advisory review found and fixed affected declared packages; post-update scans reported no known vulnerabilities in 95 installed Python packages and no production npm findings. Python dependencies still lack a full transitive lockfile, so results can change as package advisories are published.

## Documentation Improvements

- Added a source-checked system architecture and repository map.
- Separated current project scope from historical aspirations, with explicit implemented/optional/out-of-scope behavior.
- Documented labels, split boundaries, preprocessing, model selection, metrics, uncertainty, reproducibility, and research adaptations.
- Added limitations for simulator external validity, metric units, synthetic stress, maintenance what-if assumptions, upload security, and stale static bundles.
- Added evidence-backed benchmarks, traceability statuses, retrospective ADRs, roadmap, and a restrained SDG 9.5 assessment.
- Corrected the bundled dataset readme reference and removed unsupported blanket license claims.

## Research Improvements

- Marked current versus retrospective, synthetic, and proxy analyses directly in the architecture/methods documentation.
- Identified which results are one-seed, multi-seed, window-level, terminal-engine, or local microbenchmark outputs.
- Clarified that supplied papers' datasets, targets, and numerical results are not reproduced.
- Made the validation and test-label boundary explicit and mapped each objective to evidence.

## Reproducibility Improvements

- Added development-only test dependencies and an automated CI configuration for both the Python fallback and full training Python paths.
- Added exact local test/build commands to the root README and contributor guide.
- Documented result regeneration, optional tool setup, environment limitations, and the need to refresh static exports explicitly.
- Recorded successful hosted CI and Julia scenario regeneration; full model/reliability studies remain pending.
- The configured package index previously lacked `fonttools`, so a clean local Python 3.14 install remains unverified; the hosted Python 3.14 CI install, compile, and tests passed.

## Traceability Status

See [`traceability.md`](traceability.md). At the original audit completion, four requirements had direct passing test/build evidence, eight were partially verified, and production maintenance recommendations were not applicable/out of scope. The subsequent Python/web and hosted CI results are listed above. “Verified” applies only to the named check, not an entire subsystem.

## Remaining Limitations

- No full training or reliability study was rerun; published metrics remain the dated snapshots in `docs/RESULTS.md`.
- The latest hosted Python/web CI and Julia scenario checks passed for code revision `6d99649`; future pushes have their own CI runs.
- A clean local Python 3.14 environment install previously stopped when the configured package index did not provide `fonttools`; the hosted Python 3.14 CI install and test job has since passed.
- The post-update Python scan reported no known vulnerabilities across 95 installed packages, and npm production dependencies reported zero. These checks do not replace recurring scans; deployment settings were not inspected.
- Local Julia is unavailable, but the hosted Julia workflow regenerated and verified its committed scenario CSV. The Vercel dashboard and calibration page were checked live. At the last direct Streamlit URL check, Streamlit returned an access/not-found page, so the hosted deployment remains unverified.
- The Docker image build needs to be run on a host with an active Docker daemon.
- No independent operational engine data, hardware validation, or maintenance outcomes exist.
- Streamlit uploaded-history prediction uses Ridge only; public applications are not production safety systems.
- Third-party dataset and paper reuse terms should be confirmed at their source before redistribution.
- The Python dependency set is not transitively locked, and no SBOM is generated. Bounded end-to-end Ridge and upload-contract tests now exist, but full model/reliability runs remain unverified.
- The new Mermaid diagrams were reviewed as source but not rendered or parser-validated.

## Initial Maturity Score

The scores below use the prompt's weighted dimensions. They are an internal engineering rubric, not an external certification.

| Dimension | Weight | Initial | Evidence at `17bc488` |
|---|---:|---:|---|
| Architecture & Modularity | 15 | 11 | Strong module separation and current diagrams, but no single canonical architecture reference. |
| Documentation | 15 | 9 | Extensive docs, mixed with older prospective requirements and no index/traceability. |
| Code Quality | 15 | 11 | Modular, functioning source; development/test dependency boundary was unclear. |
| Testing | 15 | 9 | 29 tests passed locally; no general Python/web CI or end-to-end experiment test. |
| Security | 10 | 7 | No obvious secrets found in a simple scan; no security policy and upload boundary undocumented. |
| Reproducibility | 10 | 7 | Pinned direct dependencies and dated results; no general CI or canonical current method page. |
| Open Source Governance | 10 | 3 | MIT license existed; contributor, security, citation, conduct, and templates were absent. |
| Research Rigor & Traceability | 10 | 7 | Strong protocols/results, but no integrated requirement-to-evidence matrix or audit. |
| **Total** | **100** | **64** | |

## Final Maturity Score

| Dimension | Weight | Final | Evidence and remaining gap |
|---|---:|---:|---|
| Architecture & Modularity | 15 | 13 | Current diagrams and boundaries documented; no API or deployment integration test. |
| Documentation | 15 | 14 | Canonical index, methods, limitations, and audit added; historical source documents remain. |
| Code Quality | 15 | 12 | Runtime/dev dependencies separated and boundaries explicit; no linter/type-check policy. |
| Testing | 15 | 12 | 33 Python 3.12 tests pass, including a bounded real-data pipeline smoke test and upload contract; Python 3.12/3.14 and web CI jobs pass; full studies and hosted Streamlit remain unverified. |
| Security | 10 | 8 | Reporting guidance added and point-in-time Python/npm scans are clear after dependency fixes; no recurring scanner or hosted-settings review exists. |
| Reproducibility | 10 | 8 | CI paths and commands documented; Python transitive lock and repeated independent full runs are absent. |
| Open Source Governance | 10 | 8 | Contributor, security, conduct, citation, changelog, issue, and PR guidance added; no release process. |
| Research Rigor & Traceability | 10 | 9 | Evidence matrix and claim boundaries added; no independent fleet data or full fresh study run. |
| **Total** | **100** | **84** | |

## Recommended Next Actions

1. Monitor future Python/web CI runs and fix any regressions.
2. Add recurring Python/npm advisory checks and refresh dependency pins when fixes are published.
3. Restore or confirm the Streamlit deployment and its access settings; the Vercel dashboard was verified live, but deployment settings were not inspected.
4. Confirm reuse terms for every third-party dataset and paper before further redistribution.
5. Pursue independent operational data and external validation before making maintenance or safety claims.
