from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ProjectConfig


class SequenceGenerator:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def generate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "RUL",
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y, _, _ = self.generate_with_metadata(df, feature_cols, target_col)
        return X, y

    def generate_with_metadata(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "RUL",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        window = self.config.window_size
        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        unit_parts: list[np.ndarray] = []
        cycle_parts: list[np.ndarray] = []

        ordered = df.sort_values(["unit", "cycle"])
        for unit, group in ordered.groupby("unit", sort=False):
            if len(group) < window:
                continue
            values = group[feature_cols].to_numpy(dtype=np.float32, copy=True)
            windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
            windows = windows.transpose(0, 2, 1).copy()
            X_parts.append(windows)
            if target_col in group:
                targets = group[target_col].to_numpy(dtype=np.float32, copy=True)
                y_parts.append(targets[window - 1 :])
            unit_parts.append(np.full(len(windows), unit, dtype=np.int32))
            cycle_parts.append(group["cycle"].to_numpy(dtype=np.int32)[window - 1 :])

        n_features = len(feature_cols)
        if not X_parts:
            return (
                np.empty((0, window, n_features), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int32),
            )

        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts) if len(y_parts) == len(X_parts) else np.empty((0,), dtype=np.float32)
        units = np.concatenate(unit_parts)
        cycles = np.concatenate(cycle_parts)
        return X, y, units, cycles

    def latest_windows(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the latest complete window for each engine with enough history."""
        windows: list[np.ndarray] = []
        units: list[int] = []
        for unit, group in df.sort_values(["unit", "cycle"]).groupby("unit", sort=False):
            if len(group) < self.config.window_size:
                continue
            values = group[feature_cols].tail(self.config.window_size).to_numpy(
                dtype=np.float32,
                copy=True,
            )
            windows.append(values)
            units.append(int(unit))
        if not windows:
            return (
                np.empty((0, self.config.window_size, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        return np.stack(windows), np.asarray(units, dtype=np.int32)
