# ADR-003: Keep research demonstration tools outside core RUL inference

## Status

Accepted; retrospective record.

## Context

The repository contains a C++17/pybind11 sensor-kernel benchmark, a Julia maintenance what-if script, and a ManimGx source for the methods animation. None of these components is part of the Ridge/LSTM RUL prediction interface.

## Decision

Keep their dependencies and execution paths optional. The core Streamlit app and static results dashboard must remain usable without the C++ extension, Julia runtime, or ManimGx renderer. Commit generated scenario and animation artifacts when they are reviewed for hosted playback.

## Alternatives

- Make every tool a mandatory dependency of the primary Python environment.
- Include C++ diagnostics or scenario allocation in RUL model fitting and inference.
- Generate animation or scenario results during application startup.

## Rationale

The repository's current manifests and code isolate the tools by purpose, while the apps consume committed outputs where needed. **Retrospective interpretation based on current repository evidence.** The historical rationale was not recorded.

## Consequences and tradeoffs

- The standard app has a smaller runtime dependency surface and does not require native compilers or secondary runtimes.
- Optional analyses require their own setup and are not automatically refreshed in hosted apps.
- Their outputs must be described as diagnostics, simulation, or media rather than RUL model performance.

## Evidence

- `requirements-animation.txt`
- `cpp_engine/pyproject.toml`
- `scripts/maintenance_what_if.jl`
- `.github/workflows/julia-what-if.yml`
- `docs/architecture.md`
