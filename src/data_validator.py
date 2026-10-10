import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate(df: pd.DataFrame, dataset_name: str) -> Tuple[bool, List[str]]:
        issues = []

        # 1. Column Count Check
        if df.shape[1] != 26:
            issues.append(f"Column mismatch: Expected 26 columns, got {df.shape[1]}")

        # 2. Missing Values Check
        if df.isnull().any().any():
            null_counts = df.isnull().sum().to_dict()
            issues.append(f"Missing values found: {null_counts}")

        # 3. Sequential Cycles Check
        for unit in df['unit'].unique():
            unit_df = df[df['unit'] == unit]
            cycles = unit_df['cycle'].values
            if not np.all(cycles == np.arange(1, len(cycles) + 1)):
                issues.append(f"Unit {unit}: Cycles are not sequential")
                break # Only report first instance to avoid spamming

        # 4. Data Types Check
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            issues.append("Non-numeric data types found in dataset")

        return (len(issues) == 0, issues)
