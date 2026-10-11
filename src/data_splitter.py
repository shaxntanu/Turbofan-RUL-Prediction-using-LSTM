from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def split_engines(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        units = df["unit"].drop_duplicates().to_numpy(copy=True)
        if len(units) < 2:
            raise ValueError("At least two engines are required for an engine-level split")

        np.random.default_rng(self.config.random_seed).shuffle(units)
        n_train = min(max(int(self.config.train_ratio * len(units)), 1), len(units) - 1)
        train_units = units[:n_train]
        val_units = units[n_train:]
        train_df = df[df["unit"].isin(train_units)].copy()
        val_df = df[df["unit"].isin(val_units)].copy()
        logger.info("Engine split: %d train, %d validation", len(train_units), len(val_units))
        return train_df, val_df
