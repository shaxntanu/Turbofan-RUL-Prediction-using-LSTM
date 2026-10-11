from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


class DataValidator:
    REQUIRED_COLUMNS = [
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *[f"sensor_{i}" for i in range(1, 22)],
    ]

    @classmethod
    def validate(cls, df: pd.DataFrame, dataset_name: str) -> Tuple[bool, List[str]]:
        issues: list[str] = []
        if len(df.columns) != 26:
            issues.append(f"Column mismatch: Expected 26 columns, got {len(df.columns)}")
        missing_columns = [column for column in cls.REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {', '.join(missing_columns)}")
            return False, issues

        numeric = df[cls.REQUIRED_COLUMNS].select_dtypes(include=[np.number])
        if numeric.shape[1] != len(cls.REQUIRED_COLUMNS):
            issues.append("Non-numeric data types found in dataset")
            return False, issues
        if df[cls.REQUIRED_COLUMNS].isna().any().any():
            issues.append("Missing values found in the measurement data")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            issues.append("Infinite values found in the measurement data")

        if (df["cycle"] < 1).any():
            issues.append("Cycle values must be positive")
        if df.duplicated(["unit", "cycle"]).any():
            issues.append("Duplicate engine-cycle records found")

        for unit, group in df.groupby("unit", sort=False):
            cycles = group["cycle"].to_numpy()
            expected = np.arange(1, len(group) + 1)
            if not np.array_equal(cycles, expected):
                issues.append(f"Unit {unit}: Cycles are not sequential")
                break

        return not issues, issues
