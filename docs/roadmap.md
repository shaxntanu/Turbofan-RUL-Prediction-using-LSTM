# Roadmap

This roadmap separates completed repository work from possible follow-up research. Items beyond the first group are proposals; they do not imply access to data, external validation, or a delivery commitment.

## Completed in this repository audit

- [x] Separate pytest from the deployed app requirements into `requirements-dev.txt`.
- [x] Add read-only CI for Python 3.12/3.14, the optional native extension on Python 3.12, and the Vite production build.
- [x] Document the current architecture, objectives, methods, benchmark evidence, limitations, traceability, and verification status.
- [x] Add contributor, security, citation, and repository governance guidance.
- [x] Clarify which existing planning documents are historical and which are current references.

## Short-term engineering work

- [ ] Review the first hosted CI run and fix any platform-specific failures it exposes.
- [ ] Add a compact end-to-end Ridge smoke test that loads a small real subset, writes expected artifacts, and verifies split boundaries without running full LSTM training.
- [ ] Add an automated test for the uploaded-history forecast contract, including its current Ridge-only behavior and validation checks.
- [ ] Run and record the Julia scenario generator on a supported environment; verify the generated CSV against the committed snapshot.
- [ ] Use an advisory-scanning service in an environment with registry access, then review findings before changing pinned dependencies.

## Medium-term research work

- [ ] Evaluate on an independently sourced prognostics dataset, subject to its terms and data availability; keep it as a separate protocol from C-MAPSS.
- [ ] Repeat model comparisons over multiple engine-level splits and enough seeds to quantify split and training variability.
- [ ] Prespecify a small number of hypotheses and ablations before using the public test labels for further exploration.
- [ ] Compare calibration under explicit temporal and subset shift, with coverage metrics and uncertainty methods appropriate to the dependence structure.
- [ ] Add a documented data provenance record for each result bundle, including input checksums and source revision.

## Long-term research and deployment goals

- [ ] Validate any proposed model using representative, authorized operational data and an independent review protocol.
- [ ] Define calibrated abstention, human review, update monitoring, and safety controls before exposing predictions to maintenance workflows.
- [ ] Consider service integration only after data governance, security, reliability, and domain-owner requirements are defined.

The long-term items require external data, domain expertise, and safety review. No accuracy or maintenance-impact targets are set until those inputs exist.
