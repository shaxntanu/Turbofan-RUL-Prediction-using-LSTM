import pandas as pd
import numpy as np
import joblib
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class Normalizer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.scaler = None

    def fit(self, df: pd.DataFrame, feature_cols: list):
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        if self.config.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

        self.scaler.fit(df[feature_cols])
        logger.info(f"Scaler {type(self.scaler).__name__} fitted on training data.")

    def transform(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        if self.scaler is None:
            raise ValueError("Scaler must be fitted before transforming data.")

        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.transform(df[feature_cols])
        return df_scaled

    def save(self, path: str):
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")

    def load(self, path: str):
        self.scaler = joblib.load(path)
        logger.info(f"Scaler loaded from {path}")
