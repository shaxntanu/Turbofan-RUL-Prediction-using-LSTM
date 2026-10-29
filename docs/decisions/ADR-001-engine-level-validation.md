# ADR-001: Keep validation engines separate from training engines

## Status

Accepted; retrospective record.

## Context

C-MAPSS contains multiple ordered observations for each engine. A row-level split could place windows from the same engine in both fit and validation data, creating correlated partitions and overstating generalization to unseen engines.

## Decision

Shuffle distinct training engine IDs with the configured seed, split the IDs, and fit sensor filtering and scaling on the fit-engine rows only. Generate windows within each engine after the split.

## Alternatives

- Randomly split rows or windows.
- Fit preprocessing on all training trajectories before splitting engines.
- Use a different group-aware cross-validation design.

## Rationale

The current implementation preserves engine boundaries and keeps validation data out of fitted preprocessing. This supports evaluation on held-out engines while retaining the official test partition for final scoring.

**Retrospective interpretation based on current repository evidence.** The historical decision record was not present in Git when this ADR was written.

## Consequences and tradeoffs

- The split is reproducible for a fixed seed and engine ordering.
- A single 80/20 partition does not estimate split-to-split variability.
- Engine grouping prevents one leakage path but does not make overlapping windows statistically independent.

## Evidence

- `src/data_splitter.py`
- `src/pipeline.py`
- `docs/methodology.md`
- `tests/test_data_loader.py` (target/validation helper coverage; full pipeline split leakage is not directly regression-tested)
