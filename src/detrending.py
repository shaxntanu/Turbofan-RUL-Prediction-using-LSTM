from __future__ import annotations

import numpy as np
import pandas as pd


def polynomial_detrend_by_engine(
    frame: pd.DataFrame,
    sensor_columns: list[str],
    degree: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove each sensor's per-engine polynomial trend without using RUL labels."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    missing = [column for column in sensor_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {', '.join(missing)}")

    result = frame.copy()
    coefficient_rows: list[dict] = []
    for unit, group in result.groupby("unit", sort=False):
        row_index = group.index
        cycles = group["cycle"].to_numpy(dtype=float)
        span = float(np.ptp(cycles))
        x = (cycles - cycles.min()) / span if span else np.zeros_like(cycles)
        fit_degree = min(degree, len(group) - 1)
        for sensor in sensor_columns:
            values = group[sensor].to_numpy(dtype=float)
            coefficients = np.polyfit(x, values, deg=fit_degree)
            trend = np.polyval(coefficients, x)
            result.loc[row_index, sensor] = values - trend
            coefficient_rows.append(
                {
                    "unit": int(unit),
                    "sensor": sensor,
                    "degree": fit_degree,
                    "coefficients": ";".join(f"{value:.10g}" for value in coefficients),
                }
            )

    return result, pd.DataFrame(coefficient_rows)
