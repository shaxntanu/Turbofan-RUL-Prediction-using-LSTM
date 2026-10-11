from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.config import ProjectConfig


class DataCleaner:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def sensor_variance(self, df: pd.DataFrame) -> pd.Series:
        sensors = [name for name in df.columns if name.startswith("sensor_")]
        return df[sensors].var().sort_values(ascending=False)

    def remove_constant_sensors(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        if not self.config.remove_constant_sensors:
            return df.copy(), []
        variances = self.sensor_variance(df)
        removed = variances[variances < self.config.constant_sensor_threshold].index.tolist()
        return df.drop(columns=removed).copy(), removed

    @staticmethod
    def align_features(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
        """Preserve a source model's feature order and reject missing columns."""
        missing = [name for name in features if name not in df.columns]
        if missing:
            raise ValueError(f"Input is missing model features: {', '.join(missing)}")
        return df
