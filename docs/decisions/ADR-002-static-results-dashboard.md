# ADR-002: Serve the web dashboard from a committed results bundle

## Status

Accepted; retrospective record.

## Context

The React/Vite dashboard displays benchmark tables, figures, transfer results, reliability summaries, and maintenance scenarios. Experiment outputs under `outputs/` and fitted models under `models/` are generated locally and ignored by Git.

## Decision

Export the reviewed results and figures into `web/public/results/` and have the static dashboard read that committed bundle. Regenerate it with `scripts/export_web_data.py` after changing the source experiments.

## Alternatives

- Add a server/API that loads fitted models and computes results on demand.
- Commit all local outputs and model weights directly.
- Keep the dashboard coupled to a developer's ignored local `outputs/` directory.

## Rationale

The checked-in design allows the web application to run as a static site without a live Python prediction service. **Retrospective interpretation based on current repository evidence.** The original rationale was not recorded in Git.

## Consequences and tradeoffs

- A deployment can render the published snapshot without access to local training artifacts.
- The bundle can become stale until regenerated, reviewed, and committed.
- The web app is a results viewer, not a live inference endpoint.
- Exported files should be reviewed for local paths and private information before commit.

## Evidence

- `scripts/export_web_data.py`
- `web/src/main.jsx`
- `web/public/results/`
- `docs/architecture.md`
