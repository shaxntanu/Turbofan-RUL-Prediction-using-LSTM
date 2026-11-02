# AI-assisted change audit

**Audit date:** 2026-11-02

This record describes the substantial repository changes made during the research-grade architecture and assurance pass. It focuses on engineering changes and verification evidence rather than listing routine edits.

## Change record

| Change | Files or areas | Requirement and reason | Validation | Remaining uncertainty |
|---|---|---|---|---|
| Add dependency-free local Markdown target checking | scripts/check_docs.py, tests/test_docs_check.py, .github/workflows/ci.yml, CONTRIBUTING.md | Catch missing local links and images in repository documentation and run the check in CI and contributor workflows. | Six focused tests pass. The command checked 34 Markdown files with no missing local targets. | External URLs and heading fragments are intentionally outside the checker. The GitHub workflow result for the latest commit could not be retrieved. |
| Add research, compatibility, and artifact provenance records | docs/research-sources.md, docs/compatibility.md, docs/generated-artifacts.md, docs/README.md | Record the source and task limits of the papers and dataset, tested runtime scope, and generation paths for result bundles and visual assets. | Local link checker, source review, and comparison against manifests, workflows, and committed artifacts. | The Archify source version and edit/export command, Anatomy kit revision, and an authoritative public record for the supplied 2026 conference paper remain unverified. |
| Register prompt-referenced skills and tools | docs/AI_SKILLS_REFERENCE.md | Distinguish a skill being named from it being installed or used, and document project-specific applicability. | Compared the inventory with the supplied master prompt, current host skills, project files, and existing audit notes. | Several external README skill packages were not installed. The Shpigford repository URL could not be confirmed. KiCad tools are not applicable because there is no hardware design in this repository. |
| Add an assumptions and failure-mode register | docs/assumptions.md, docs/limitations.md | Make the C-MAPSS, metric, calibration, what-if, static snapshot, upload, and optional-runtime assumptions explicit. | Reviewed against the experiment protocols, app behavior, and deployment limitations. | Operational and confidential-data workflows were not available for validation. |
| Extend root navigation and traceability | README.md, docs/README.md, docs/traceability.md | Make the new evidence documents discoverable and map the new documentation checks to requirements. | Local Markdown target check and root README constraint scan. | The README remains a project summary; the documentation guide is the full index. No Contribution section was added. |
| Record the assistance and verification trail | docs/AI_CHANGE_AUDIT.md | Preserve the major changes, source checks, validations, and unresolved evidence from this pass. | Compared with tool outputs, changed files, commits, and repository status. | The record covers this audit pass and does not claim independent review by another engineer. |

## External sources and specialized guidance

- NASA's [C-MAPSS catalog record](https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data) states that the data are simulator-generated, the catalog currently has no downloadable payload, and a license is not specified.
- The supplied paper references were compared with the NASA-hosted 2008 [damage propagation paper](https://doi.org/10.1109/PHM.2008.4711414) and the 2025 publisher version of record for [A Framework for an ML-Based Predictive Turbofan Engine Health Model](https://www.mdpi.com/2226-4310/12/8/725). The project documents its C-MAPSS RUL results as methodological adaptations, not numerical replications of in-service TGT studies.
- The official [ManimGx documentation](https://manimgx.academa.ai/) was reviewed for the existing media provenance. No new video was rendered.
- The referenced writing and editing guidance was read and applied to the documentation pass. Taste was used for the responsive UI review. Diagram Design was considered, but no duplicate diagram was created.
- Crawl4AI was not available in the local environment and was not used. Ordinary web search and direct source pages were used for the source checks.

## Verification performed

- Repository Python suite: 39 passed, with one expected scikit-learn convergence warning from the test that deliberately limits an MLP to three iterations.
- Python module compilation and pip dependency consistency check passed.
- Local Markdown link and image checker passed across 34 Markdown files; its six focused tests passed.
- Vite production build passed on Node v24.19.0.
- The local production dashboard was inspected at desktop, tablet, and mobile sizes, including dark mode, combined-subset results, reliability, the study diagram, and Methods content. The review found no console errors or horizontal overflow and one warning for compact analytical labels.
- A tracked and unignored text-file scan found no em dash characters. The root README still has no Contribution heading.
- Local Docker Engine 28.5.1 built the Streamlit image and both the health and root endpoints returned HTTP 200 in the separate Docker smoke check.

## Checks that could not be completed

- The local Python advisory request stalled without a response and was interrupted. npm audit --omit=dev returned an advisory endpoint error. Both current local advisory scans are inconclusive. Earlier hosted scans apply only to their recorded commit.
- GitHub API requests timed out, so the CI run status for the newly pushed commits was unavailable.
- Julia is not installed locally. The previously recorded Julia workflow verified the committed maintenance scenario CSV on its own run.
- The Streamlit public URL's current settings and app access remain unverified. A prior URL check reached the service access/not-found page.
- No independent operational engine dataset, maintenance outcomes, physical hardware, firmware, or confidential deployment configuration was available.

## Generated outputs and change scope

No new experiment results, fitted models, dataset files, Archify diagrams, ManimGx videos, or LaTeX files were generated in this pass. The model code and published metric snapshots were not changed. The new validator only checks documentation paths. The existing visuals were retained because they explain distinct project concepts.

Commits 7ab7016 and 19543d8 were pushed to origin/main with Shantanu as author. This assurance pass is tracked in a separate atomic commit.
