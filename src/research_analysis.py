from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_sensor_screening(
    train_frame: pd.DataFrame,
    sensor_columns: list[str],
    output_dir: str | Path,
    variance_threshold: float = 0.95,
    max_rows: int = 50000,
) -> dict:
    """Rank sensor/RUL associations and measure PCA compression on training rows."""
    if "RUL" not in train_frame:
        raise ValueError("Sensor screening requires a training RUL column")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correlations = train_frame[sensor_columns].corrwith(train_frame["RUL"], method="pearson")
    correlation_table = pd.DataFrame(
        {
            "sensor": correlations.index,
            "pearson_r": correlations.to_numpy(),
            "absolute_pearson_r": np.abs(correlations.to_numpy()),
            "variance": train_frame[sensor_columns].var().reindex(correlations.index).to_numpy(),
        }
    ).sort_values("absolute_pearson_r", ascending=False)
    correlation_table.to_csv(output_dir / "sensor_screening.csv", index=False)

    values = train_frame[sensor_columns]
    if len(values) > max_rows:
        values = values.sample(n=max_rows, random_state=42)
    scaled = StandardScaler().fit_transform(values)
    pca = PCA(svd_solver="full").fit(scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    pca_table = pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": cumulative,
        }
    )
    pca_table.to_csv(output_dir / "pca_variance.csv", index=False)
    result = {
        "method": "standardized sensor PCA; Pearson correlation with training RUL",
        "rows_used": int(len(values)),
        "sensor_count": int(len(sensor_columns)),
        "variance_threshold": float(variance_threshold),
        "components_to_threshold": components,
        "cumulative_variance_at_threshold": float(cumulative[components - 1]),
        "top_sensors_by_absolute_pearson": correlation_table.head(8).to_dict("records"),
    }
    (output_dir / "sensor_screening.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    return result


def summarize_engine_robustness(per_engine: pd.DataFrame, metric: str = "mae") -> dict[str, float]:
    """Summarize fleet variation and flag Tukey-IQR outliers by engine."""
    values = per_engine[metric].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Per-engine {metric} values must be non-empty and finite")
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = float(q3 - q1)
    upper_fence = float(q3 + 1.5 * iqr)
    outliers = values > upper_fence
    return {
        "metric": metric,
        "engine_count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "upper_tukey_fence": upper_fence,
        "outlier_engines": int(np.sum(outliers)),
        "outlier_fraction": float(np.mean(outliers)),
        "max": float(np.max(values)),
    }
