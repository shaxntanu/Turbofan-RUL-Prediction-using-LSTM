from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class WindowRidgeBaseline:
    """Fast regression baseline over summary features from each sensor window."""

    SUMMARY_NAMES = ("first", "mean", "std", "slope", "last")

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    @classmethod
    def summarize(cls, sequences: np.ndarray) -> np.ndarray:
        if sequences.ndim != 3:
            raise ValueError("Expected sequences shaped (samples, window, features)")
        window = sequences.shape[1]
        positions = np.arange(window, dtype=np.float32)
        centered = positions - positions.mean()
        denominator = float(np.dot(centered, centered)) or 1.0
        slope = np.einsum("ntf,t->nf", sequences, centered) / denominator
        return np.concatenate(
            [
                sequences[:, 0, :],
                sequences.mean(axis=1),
                sequences.std(axis=1),
                slope,
                sequences[:, -1, :],
            ],
            axis=1,
        ).astype(np.float32, copy=False)

    def fit(self, sequences: np.ndarray, targets: np.ndarray):
        if len(sequences) != len(targets) or not len(targets):
            raise ValueError("Baseline requires matching non-empty training sequences and targets")
        self.model.fit(self.summarize(sequences), np.asarray(targets, dtype=float))
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        return self.model.predict(self.summarize(sequences)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"alpha": self.alpha, "model": self.model}, path)

    @classmethod
    def load(cls, path: str | Path) -> "WindowRidgeBaseline":
        artifact = joblib.load(path)
        baseline = cls(alpha=float(artifact["alpha"]))
        baseline.model = artifact["model"]
        return baseline
