# Contributing

Thank you for helping improve the Turbofan RUL Observatory. Contributions should preserve the distinction between simulated benchmark results and evidence from operational engines.

## Set up a development environment

Use Python 3.12 for LSTM training and the optional native extension. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

For the complete training and optional C++ environment, install `requirements-training.txt`. The React dashboard has a separate environment:

```powershell
cd web
npm ci
npm run dev
```

## Before opening a pull request

Run the checks that apply to your changes:

```powershell
python -m compileall -q src scripts tests
python -m pytest -q
```

```powershell
cd web
npm ci
npm run build
```

For changes to Julia scenario generation, run `julia scripts/maintenance_what_if.jl` and verify that the committed scenario table is the intended result. For the optional C++ extension, build it with `python -m pip install ./cpp_engine` and run `python -m pytest tests/test_cpp_engine.py -q`.

## Research and data expectations

- Keep training, validation, and official test roles explicit. Do not fit preprocessing, choose models, or tune thresholds using official test labels.
- State whether a score is per-window or per-engine, and document the aggregation rule.
- Record dataset subset, seed, configuration, software versions, and commands for new results.
- Label synthetic stressors and retrospective life-stage analyses; do not describe them as field observations.
- Do not claim a paper reproduction, operational benefit, or algorithmic novelty without evidence.
- Do not commit private telemetry, credentials, model artifacts, or large generated outputs. Update the static web export only when its provenance and figures are intentionally reviewed.
- Keep changes focused. If a full experiment cannot be rerun, say which existing artifact or check supports the change and what remains unverified.

## Pull requests

Use the repository pull request template. Include the problem, behavior changed, verification performed, and any result or deployment caveats. Link related issues when applicable. Do not include secrets or private operational data in issues, logs, or screenshots.
