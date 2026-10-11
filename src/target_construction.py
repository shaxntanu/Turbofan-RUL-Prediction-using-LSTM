from __future__ import annotations

import pandas as pd


class TargetConstructor:
    @staticmethod
    def compute_training_rul(
        df: pd.DataFrame,
        clip_value: int | None = None,
    ) -> pd.DataFrame:
        result = df.copy()
        last_cycles = result.groupby("unit")["cycle"].transform("max")
        result["RUL"] = (last_cycles - result["cycle"]).astype(float)
        if clip_value is not None:
            result["RUL"] = result["RUL"].clip(upper=clip_value)
        return result

    @staticmethod
    def compute_test_rul(
        df: pd.DataFrame,
        final_ruls: pd.Series,
        clip_value: int | None = None,
    ) -> pd.DataFrame:
        result = df.copy()
        units = result["unit"].drop_duplicates().tolist()
        final_values = pd.Series(final_ruls).reset_index(drop=True)
        if len(units) != len(final_values):
            raise ValueError(
                f"Test set has {len(units)} engines but the RUL file has "
                f"{len(final_values)} labels"
            )

        final_by_unit = dict(zip(units, final_values.astype(float), strict=True))
        last_cycles = result.groupby("unit")["cycle"].transform("max")
        result["RUL"] = last_cycles - result["cycle"] + result["unit"].map(final_by_unit)
        if clip_value is not None:
            result["RUL"] = result["RUL"].clip(upper=clip_value)
        return result
