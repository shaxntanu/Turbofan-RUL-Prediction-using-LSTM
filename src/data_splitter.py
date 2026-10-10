import pandas as pd
import numpy as np
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class DataSplitter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def split_engines(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits engines into training and validation sets at the engine level.
        """
        np.random.seed(self.config.random_seed)
        unique_units = df['unit'].unique()
        np.random.shuffle(unique_units)

        n_train = int(self.config.train_ratio * len(unique_units))

        train_units = unique_units[:n_train]
        val_units = unique_units[n_train:]

        train_df = df[df['unit'].isin(train_units)]
        val_df = df[df['unit'].isin(val_units)]

        logger.info(f"Split engines: {len(train_units)} train, {len(val_units)} val")
        return train_df, val_df
