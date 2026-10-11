from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_model import WindowRidgeBaseline
from src.evaluator import Evaluator

LIFE_STAGES = ("early", "middle", "late")


def _stage_for_windows(
    units: np.ndarray,
    cycles: np.ndarray,
    lifetime_by_unit: dict[int, float],
) -> np.ndarray:
    lifetimes = np.asarray([lifetime_by_unit[int(unit)] for unit in units], dtype=float)
    fractions = np.clip((cycles.astype(float) - 1) / np.maximum(lifetimes - 1, 1), 0, 1)
    return np.asarray(LIFE_STAGES, dtype=object)[np.minimum((fractions * 3).astype(int), 2)]


def run_fractional_life_analysis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_units: np.ndarray,
    train_cycles: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_units: np.ndarray,
    test_cycles: np.ndarray,
    train_lifetimes: dict[int, float],
    test_lifetimes: dict[int, float],
    output_dir: str | Path,
    *,
    minimum_stage_windows: int = 50,
    max_rul: float | None = None,
) -> dict:
    """Compare a shared baseline with Ridge models specialized by life stage."""
    if not (len(X_train) == len(y_train) == len(train_units) == len(train_cycles)):
        raise ValueError("Training windows, targets, units, and endpoint cycles must align")
    if not (len(X_test) == len(y_test) == len(test_units) == len(test_cycles)):
        raise ValueError("Test windows, targets, units, and endpoint cycles must align")
    if not len(X_train) or not len(X_test):
        raise ValueError("Fractional-life analysis requires non-empty train and test windows")

    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_stages = _stage_for_windows(train_units, train_cycles, train_lifetimes)
    test_stages = _stage_for_windows(test_units, test_cycles, test_lifetimes)
    shared_model = WindowRidgeBaseline().fit(X_train, y_train)
    rows = []

    for stage in LIFE_STAGES:
        train_mask = train_stages == stage
        test_mask = test_stages == stage
        if not test_mask.any():
            continue

        shared_predictions = shared_model.predict(X_test[test_mask])
        shared_predictions = (
            np.clip(shared_predictions, 0.0, max_rul)
            if max_rul is not None
            else np.maximum(shared_predictions, 0.0)
        )
        rows.append(
            {
                "life_stage": stage,
                "model_scope": "shared_full_life",
                "training_windows": int(len(X_train)),
                **Evaluator.compute_metrics(y_test[test_mask], shared_predictions),
            }
        )

        if int(train_mask.sum()) < minimum_stage_windows:
            continue
        phase_model = WindowRidgeBaseline().fit(X_train[train_mask], y_train[train_mask])
        phase_predictions = phase_model.predict(X_test[test_mask])
        phase_predictions = (
            np.clip(phase_predictions, 0.0, max_rul)
            if max_rul is not None
            else np.maximum(phase_predictions, 0.0)
        )
        rows.append(
            {
                "life_stage": stage,
                "model_scope": "phase_specialized",
                "training_windows": int(train_mask.sum()),
                **Evaluator.compute_metrics(y_test[test_mask], phase_predictions),
            }
        )
        phase_model.save(model_dir / f"ridge_{stage}.joblib")

    metrics = pd.DataFrame(rows)
    metrics_path = output_dir / "fractional_life_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    result = {
        "schema_version": 1,
        "stage_boundaries": {"early": "[0, 1/3)", "middle": "[1/3, 2/3)", "late": "[2/3, 1]"},
        "stage_assignment": "window endpoint cycle divided by engine lifetime; test RUL is used only to stratify evaluation",
        "protocols": ["shared_full_life", "phase_specialized"],
        "minimum_stage_training_windows": int(minimum_stage_windows),
        "metrics_path": str(metrics_path),
        "metrics": rows,
        "limitation": (
            "Fractional-life phases are a proxy for overhaul-defined segments; C-MAPSS has no "
            "overhaul boundaries or in-service maintenance history."
        ),
    }
    (output_dir / "fractional_life_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
