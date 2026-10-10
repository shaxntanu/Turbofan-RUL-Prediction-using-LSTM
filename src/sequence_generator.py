import pandas as pd
import numpy as np
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class SequenceGenerator:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def generate(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'RUL') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates sliding window sequences for LSTM.
        X shape: (n_sequences, window_size, n_features)
        y shape: (n_sequences,)
        """
        sequences = []
        targets = []

        # Ensure sorting by unit and cycle
        df = df.sort_values(['unit', 'cycle'])

        for unit in df['unit'].unique():
            unit_data = df[df['unit'] == unit]
            features = unit_data[feature_cols].values
            targets_vals = unit_data[target_col].values

            # Sliding window
            for i in range(len(unit_data) - self.config.window_size + 1):
                sequences.append(features[i : i + self.config.window_size])
                targets.append(targets_vals[i + self.config.window_size - 1])

        return np.array(sequences), np.array(targets)
