# Roadmap

This roadmap separates completed repository work from possible follow-up research. Items beyond the first group are proposals; they do not imply access to data, external validation, or a delivery commitment.

## Completed in this repository audit

- [x] Separate pytest from the deployed app requirements into `requirements-dev.txt`.
- [x] Add read-only CI for Python 3.12/3.14, the optional native extension on Python 3.12, and the Vite production build.
- [x] Document the current architecture, objectives, methods, benchmark evidence, limitations, traceability, and verification status.
- [x] Add contributor, security, citation, and repository governance guidance.
- [x] Clarify which existing planning documents are historical and which are current references.
- [x] Review hosted CI across Python 3.12/3.14, the optional C++ wheel, and the Vite build; all jobs passed for `6d99649`.
- [x] Add a bounded real-data Ridge pipeline smoke test with artifact and engine-split checks.
- [x] Add upload-forecast contract coverage for validation, cycle metadata, and bounded predictions.
- [x] Run the Julia scenario generator in hosted CI and verify its regenerated CSV matches the committed snapshot.

## Short-term engineering work

- [ ] Complete Python and npm dependency advisory scans in an environment with registry access, then review findings before changing pinned dependencies.

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
