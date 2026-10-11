from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Evaluator:
    @staticmethod
    def phm08_score(y_true, y_pred) -> float:
        """Sum the PHM08 asymmetric penalty function over the supplied predictions."""
        error = np.asarray(y_pred, dtype=float).ravel() - np.asarray(y_true, dtype=float).ravel()
        penalties = np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0))
        return float(np.sum(penalties))

    @classmethod
    def compute_metrics(cls, y_true, y_pred) -> dict[str, float]:
        truth = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        if truth.size == 0 or truth.size != pred.size:
            raise ValueError("y_true and y_pred must be non-empty arrays of equal length")
        if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
            raise ValueError("Metric inputs must contain only finite values")
        return {
            "mae": float(mean_absolute_error(truth, pred)),
            "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
            "r2": float(r2_score(truth, pred)) if truth.size > 1 else 0.0,
            "mean_error": float(np.mean(truth - pred)),
            "median_absolute_error": float(np.median(np.abs(truth - pred))),
            "phm08_score": cls.phm08_score(truth, pred),
            "n_samples": int(truth.size),
        }

    @staticmethod
    def analyze_errors(y_true, y_pred) -> dict[str, float]:
        errors = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
        return {
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "max_error": float(np.max(np.abs(errors))),
        }

    @classmethod
    def per_engine_metrics(cls, y_true, y_pred, units) -> pd.DataFrame:
        truth = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        unit_ids = np.asarray(units).ravel()
        if not (truth.size == pred.size == unit_ids.size):
            raise ValueError("y_true, y_pred, and units must have equal lengths")
        rows = []
        for unit in np.unique(unit_ids):
            mask = unit_ids == unit
            metrics = cls.compute_metrics(truth[mask], pred[mask])
            rows.append({"unit": int(unit), **metrics})
        return pd.DataFrame(rows)
