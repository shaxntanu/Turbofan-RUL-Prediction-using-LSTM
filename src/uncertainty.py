from __future__ import annotations

import math

import numpy as np


class SplitConformalRUL:
    """Distribution-free absolute-residual intervals calibrated on validation engines."""

    def __init__(self, nominal_coverage: float = 0.95, eta: float = 50.0):
        if not 0.0 < nominal_coverage < 1.0:
            raise ValueError("nominal_coverage must be between 0 and 1")
        self.nominal_coverage = nominal_coverage
        self.eta = eta
        self.radius: float | None = None

    def fit(self, y_true, y_pred) -> "SplitConformalRUL":
        truth = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        if truth.size == 0 or truth.size != pred.size:
            raise ValueError("Calibration arrays must be non-empty and the same length")
        if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
            raise ValueError("Calibration arrays must contain only finite values")

        residuals = np.abs(truth - pred)
        rank = min(math.ceil((len(residuals) + 1) * self.nominal_coverage), len(residuals))
        self.radius = float(np.partition(residuals, rank - 1)[rank - 1])
        return self

    def predict_interval(
        self,
        predictions,
        max_rul: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.radius is None:
            raise ValueError("Fit the interval on validation residuals first")
        pred = np.asarray(predictions, dtype=float).ravel()
        lower = np.maximum(0.0, pred - self.radius)
        upper = pred + self.radius
        if max_rul is not None:
            upper = np.minimum(max_rul, upper)
        return lower, np.maximum(lower, upper)

    def metrics(self, y_true, lower, upper) -> dict[str, float]:
        truth = np.asarray(y_true, dtype=float).ravel()
        low = np.asarray(lower, dtype=float).ravel()
        high = np.asarray(upper, dtype=float).ravel()
        if not truth.size or not (truth.size == low.size == high.size):
            raise ValueError("Truth and interval arrays must be non-empty and the same length")

        coverage = float(np.mean((truth >= low) & (truth <= high)))
        mean_width = float(np.mean(high - low))
        truth_range = float(np.ptp(truth))
        normalized_width = mean_width / truth_range if truth_range > 0 else 0.0
        if coverage >= self.nominal_coverage:
            cwc = mean_width
        else:
            penalty = math.exp(min(50.0, self.eta * (self.nominal_coverage - coverage)))
            cwc = mean_width * (1.0 + penalty)
        return {
            "nominal_coverage": float(self.nominal_coverage),
            "picp": coverage,
            "mpi_width_cycles": mean_width,
            "nmpi_width": float(normalized_width),
            "cwc_cycles": float(cwc),
            "interval_radius_cycles": float(self.radius or 0.0),
        }
