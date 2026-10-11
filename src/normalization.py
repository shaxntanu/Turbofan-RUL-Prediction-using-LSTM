from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.config import ProjectConfig


class Normalizer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.scaler = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        scaler_type = MinMaxScaler if self.config.scaler_type == "minmax" else StandardScaler
        self.scaler = scaler_type()
        self.scaler.fit(df[feature_cols])
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError("Fit the scaler on training data before transforming data")
        missing = [name for name in feature_cols if name not in df.columns]
        if missing:
            raise ValueError(f"Input is missing scaler features: {', '.join(missing)}")
        result = df.copy()
        result.loc[:, feature_cols] = self.scaler.transform(result[feature_cols])
        return result

    def save(self, path: str | Path) -> None:
        if self.scaler is None:
            raise ValueError("Cannot save an unfitted scaler")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)

    def load(self, path: str | Path):
        self.scaler = joblib.load(path)
        return self
