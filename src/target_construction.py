import pandas as pd
import numpy as np
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class TargetConstructor:
    @staticmethod
    def compute_training_rul(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes RUL for training data: max_cycle - current_cycle
        """
        df = df.copy()
        # Find max cycle for each engine
        max_cycles = df.groupby('unit')['cycle'].max()
        # Map max cycle back to each row and subtract current cycle
        df['RUL'] = df['unit'].map(max_cycles) - df['cycle']
        return df

    @staticmethod
    def compute_test_rul(df: pd.DataFrame, final_ruls: pd.Series) -> pd.DataFrame:
        """
        Computes RUL for test data: max_test_cycle - current_cycle + final_rul
        """
        df = df.copy()
        # Map final RUL from the provided Series (indexed by 0...N-1)
        # We assume the order in final_ruls matches the order of unique units in test_df
        units = df['unit'].unique()
        unit_to_final_rul = {unit: final_ruls.iloc[i] for i, unit in enumerate(units)}

        max_cycles = df.groupby('unit')['cycle'].max()

        def calc_row_rul(row):
            unit = row['unit']
            cycle = row['cycle']
            return max_cycles[unit] - cycle + unit_to_final_rul[unit]

        df['RUL'] = df.apply(calc_row_rul, axis=1)
        return df
