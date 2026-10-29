## Summary

Describe the user-visible or research behavior changed and why.

## Verification

- [ ] `python -m compileall -q src scripts tests` (when Python code changes)
- [ ] `python -m pytest -q` (when Python behavior changes)
- [ ] `npm run build` from `web/` (when web assets or data change)
- [ ] Other checks and exact commands:

## Research and data notes

- Evaluation unit (window / terminal engine / other):
- Dataset, seed, and protocol (if results changed):
- Test-label use and leakage controls (if experiments changed):
- Are results synthetic, simulated, or from operational data?

## Risks and documentation

- [ ] User-facing behavior and limitations are documented.
- [ ] No credentials, private telemetry, or unreviewed generated artifacts are included.
- [ ] Claims are supported by the included evidence.

## Deployment

Describe whether a deployment setting or hosted result bundle needs an update.
