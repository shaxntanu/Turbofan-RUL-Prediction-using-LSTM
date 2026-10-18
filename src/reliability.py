from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.baseline_model import WindowRidgeBaseline
LIFE_STAGES = ("early", "middle", "late")
ABSTENTION_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.5)
SEVERITY_LEVELS = ("mild", "moderate", "severe")
SEVERITY = {"mild": 1, "moderate": 2, "severe": 3}


def leave_one_subset_sources(datasets, target: str) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(name).upper() for name in datasets))
    target = target.upper()
    if len(selected) < 2 or target not in selected:
        raise ValueError("LOFO requires at least two subsets and a target included in the subset list")
    sources = tuple(name for name in selected if name != target)
    if target in sources:
        raise AssertionError("Target subset must be excluded from LOFO sources")
    return sources


@dataclass(frozen=True)
class CorruptionScenario:
    name: str
    kind: str
    severity: str
    apply: Callable[[pd.DataFrame], pd.DataFrame]


def phm08_contributions(y_true, y_pred) -> np.ndarray:
    """Return one PHM08 challenge penalty per prediction (prediction minus truth)."""
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    if truth.size == 0 or truth.size != pred.size:
        raise ValueError("PHM08 inputs must be non-empty arrays of equal length")
    if not (np.isfinite(truth).all() and np.isfinite(pred).all()):
        raise ValueError("PHM08 inputs must contain only finite values")
    error = pred - truth
    return np.where(error < 0, np.expm1(-error / 13.0), np.expm1(error / 10.0))


def terminal_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one latest complete window and its endpoint label for each engine."""
    sequences, units, cycles, targets = [], [], [], []
    ordered = frame.sort_values(["unit", "cycle"])
    for unit, group in ordered.groupby("unit", sort=False):
        if len(group) < window_size:
            continue
        tail = group.tail(window_size)
        sequences.append(tail[feature_columns].to_numpy(dtype=np.float32, copy=True))
        units.append(int(unit))
        cycles.append(int(tail["cycle"].iloc[-1]))
        targets.append(float(tail["RUL"].iloc[-1]))
    n_features = len(feature_columns)
    if not sequences:
        return (
            np.empty((0, window_size, n_features), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    return (
        np.stack(sequences),
        np.asarray(units, dtype=np.int32),
        np.asarray(cycles, dtype=np.int32),
        np.asarray(targets, dtype=np.float32),
    )


def validation_censor_windows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
    *,
    seed: int,
    max_rul: float = 125.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one reproducible pseudo-censored endpoint per validation engine."""
    rng = np.random.default_rng(seed)
    sequences, units, cycles, targets = [], [], [], []
    ordered = frame.sort_values(["unit", "cycle"])
    for unit, group in ordered.groupby("unit", sort=False):
        if len(group) < window_size:
            continue
        candidates = group.iloc[window_size - 1 :]
        requested_rul = rng.uniform(0.0, max_rul)
        distance = np.abs(candidates["RUL"].to_numpy(dtype=float) - requested_rul)
        nearest = np.flatnonzero(distance == distance.min())
        endpoint_position = int(nearest[rng.integers(len(nearest))])
        endpoint = window_size - 1 + endpoint_position
        prefix = group.iloc[: endpoint + 1]
        tail = prefix.tail(window_size)
        sequences.append(tail[feature_columns].to_numpy(dtype=np.float32, copy=True))
        units.append(int(unit))
        cycles.append(int(tail["cycle"].iloc[-1]))
        targets.append(float(tail["RUL"].iloc[-1]))
    if not sequences:
        return (
            np.empty((0, window_size, len(feature_columns)), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    return (
        np.stack(sequences),
        np.asarray(units, dtype=np.int32),
        np.asarray(cycles, dtype=np.int32),
        np.asarray(targets, dtype=np.float32),
    )


def prediction_frame(
    dataset: str,
    model: str,
    units: np.ndarray,
    cycles: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    scenario: str = "clean",
    kind: str = "clean",
    severity: str = "none",
) -> pd.DataFrame:
    pred = np.asarray(predictions, dtype=float).ravel()
    true = np.asarray(truth, dtype=float).ravel()
    if not (len(units) == len(cycles) == len(true) == len(pred)):
        raise ValueError("Prediction metadata and values must have equal lengths")
    result = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "scenario": scenario,
            "corruption_type": kind,
            "severity": severity,
            "unit": np.asarray(units, dtype=int),
            "cycle": np.asarray(cycles, dtype=int),
            "rul_true": true,
            "rul_predicted": pred,
            "residual_pred_minus_true": pred - true,
            "absolute_error": np.abs(pred - true),
            "phm08_score_contribution": phm08_contributions(true, pred),
        }
    )
    if lower is not None and upper is not None:
        result["interval_lower"] = np.asarray(lower, dtype=float).ravel()
        result["interval_upper"] = np.asarray(upper, dtype=float).ravel()
        result["interval_covered"] = (
            (true >= result["interval_lower"]) & (true <= result["interval_upper"])
        )
        result["interval_width"] = result["interval_upper"] - result["interval_lower"]
    return result


def terminal_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        raise ValueError("Terminal metrics require at least one engine prediction")
    truth = frame["rul_true"].to_numpy(dtype=float)
    pred = frame["rul_predicted"].to_numpy(dtype=float)
    contributions = frame["phm08_score_contribution"].to_numpy(dtype=float)
    values: dict[str, float | int] = {
        "terminal_engine_mae": float(np.mean(np.abs(pred - truth))),
        "terminal_engine_rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "phm08_score_sum": float(contributions.sum()),
        "phm08_score_mean": float(contributions.mean()),
        "n_engines": int(len(frame)),
    }
    if "interval_covered" in frame:
        values["interval_coverage"] = float(frame["interval_covered"].mean())
        values["mean_interval_width"] = float(frame["interval_width"].mean())
    return values


def training_augmentation(
    sequences: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    copies: int = 1,
    channel_dropout: float = 0.08,
    noise_fraction: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Append train-only channel-dropout/noise views without changing their labels."""
    if len(sequences) != len(targets) or not len(targets):
        raise ValueError("Augmentation requires matching non-empty training sequences and labels")
    if copies < 1 or not 0.0 <= channel_dropout <= 1.0 or noise_fraction < 0:
        raise ValueError("Invalid augmentation configuration")
    rng = np.random.default_rng(seed)
    center = np.median(sequences, axis=(0, 1), keepdims=True)
    feature_scale = np.std(sequences, axis=(0, 1), keepdims=True)
    feature_scale = np.maximum(feature_scale, 1e-3)
    augmented = []
    for _ in range(copies):
        values = sequences.copy()
        mask = rng.random((len(values), 1, values.shape[2])) < channel_dropout
        values = np.where(mask, center, values)
        values += rng.normal(size=values.shape).astype(np.float32) * feature_scale * noise_fraction
        augmented.append(values)
    return np.concatenate([sequences, *augmented]), np.tile(np.asarray(targets), copies + 1)


def sensor_corruption_scenarios(
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
) -> list[CorruptionScenario]:
    """Create deterministic test-time sensor faults using training-only fill/scales."""
    sensors = sorted(
        [name for name in feature_columns if name.startswith("sensor_")],
        key=lambda name: int(name.split("_")[1]),
    )
    if not sensors:
        return []
    medians = training_frame[sensors].median().to_dict()
    scales = training_frame[sensors].std(ddof=0).replace(0, np.nan)
    scales = scales.fillna(1.0).to_dict()
    groups = [
        [name for name in sensors if 1 <= int(name.split("_")[1]) <= 7],
        [name for name in sensors if 8 <= int(name.split("_")[1]) <= 14],
        [name for name in sensors if 15 <= int(name.split("_")[1]) <= 21],
    ]
    groups = [group for group in groups if group]
    scenarios: list[CorruptionScenario] = []

    def float_sensor_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        result = frame.copy()
        for column in columns:
            result[column] = result[column].astype(float)
        return result

    def dropout(columns: list[str], probability: float, scenario_seed: int):
        def apply(frame: pd.DataFrame) -> pd.DataFrame:
            result = float_sensor_frame(frame, columns)
            rng = np.random.default_rng(scenario_seed)
            mask = rng.random((len(result), len(columns))) < probability
            result.loc[:, columns] = np.where(
                mask,
                np.asarray([medians[name] for name in columns]),
                result[columns].to_numpy(),
            )
            return result
        return apply

    for index, sensor in enumerate(sensors):
        scenarios.append(
            CorruptionScenario(
                f"dropout_{sensor}_moderate",
                "individual_dropout",
                "moderate",
                dropout([sensor], 0.6, seed + index),
            )
        )

    for group_index, columns in enumerate(groups):
        group_name = f"sensor_group_{group_index + 1}"
        for level, probability in zip(SEVERITY_LEVELS, (0.25, 0.6, 1.0), strict=True):
            scenarios.append(
                CorruptionScenario(
                    f"dropout_{group_name}_{level}",
                    "group_dropout",
                    level,
                    dropout(columns, probability, seed + 100 + group_index * 10 + SEVERITY[level]),
                )
            )

    for group_index, columns in enumerate(groups):
        group_name = f"sensor_group_{group_index + 1}"
        for level, onset in zip(SEVERITY_LEVELS, (180, 120, 60), strict=True):
            def stuck(frame: pd.DataFrame, names=columns, onset_cycle=onset):
                result = float_sensor_frame(frame, names)
                mask = result["cycle"] >= onset_cycle
                result.loc[mask, names] = [medians[name] for name in names]
                return result
            scenarios.append(
                CorruptionScenario(
                    f"stuck_{group_name}_{level}", "stuck", level, stuck
                )
            )

    for level, magnitude in zip(SEVERITY_LEVELS, (0.5, 1.0, 2.0), strict=True):
        def drift(frame: pd.DataFrame, factor=magnitude):
            result = float_sensor_frame(frame, sensors)
            elapsed = np.maximum(result["cycle"].to_numpy(dtype=float) - 1.0, 0.0)
            ramp = np.minimum(elapsed / 200.0, 1.0)
            for name in sensors:
                result[name] += factor * scales[name] * ramp
            return result
        scenarios.append(CorruptionScenario(f"drift_all_{level}", "drift", level, drift))

    for level, multiplier in zip(SEVERITY_LEVELS, (0.1, 0.25, 0.5), strict=True):
        def noise(frame: pd.DataFrame, factor=multiplier, severity_offset=SEVERITY[level]):
            result = float_sensor_frame(frame, sensors)
            rng = np.random.default_rng(seed + severity_offset)
            values = rng.normal(size=(len(result), len(sensors)))
            values *= np.asarray([scales[name] for name in sensors]) * factor
            result.loc[:, sensors] = result[sensors].to_numpy() + values
            return result
        scenarios.append(CorruptionScenario(f"noise_all_{level}", "noise", level, noise))

    for level, (start_cycle, length) in zip(
        SEVERITY_LEVELS, ((100, 10), (100, 25), (100, 50)), strict=True
    ):
        def missing_block(frame: pd.DataFrame, start=start_cycle, block_length=length):
            result = float_sensor_frame(frame, sensors)
            missing = result["cycle"].between(start, start + block_length - 1)
            result.loc[missing, sensors] = [medians[name] for name in sensors]
            return result
        scenarios.append(
            CorruptionScenario(f"missing_time_block_{level}", "missing_block", level, missing_block)
        )

    for level, stride in zip(SEVERITY_LEVELS, (2, 3, 4), strict=True):
        def reduced_sampling(frame: pd.DataFrame, step=stride):
            sampled = []
            for _, group in frame.sort_values(["unit", "cycle"]).groupby("unit", sort=False):
                subset = group.iloc[::step]
                if subset.index[-1] != group.index[-1]:
                    subset = pd.concat([subset, group.tail(1)])
                sampled.append(subset)
            return pd.concat(sampled, ignore_index=True)
        scenarios.append(
            CorruptionScenario(f"sampling_stride_{stride}", "reduced_sampling", level, reduced_sampling)
        )

    combo_seed = seed + 900
    scenarios.append(
        CorruptionScenario(
            "combined_severe_dropout_noise",
            "combined",
            "severe",
            lambda frame: _combine_dropout_noise(
                frame, groups, sensors, medians, scales, seed=combo_seed, noise_fraction=0.25
            ),
        )
    )
    scenarios.append(
        CorruptionScenario(
            "combined_severe_drift_missing_block",
            "combined",
            "severe",
            lambda frame: _combine_drift_missing(
                frame, sensors, medians, scales, horizon=200, start_cycle=100, block_length=50
            ),
        )
    )
    return scenarios


def _combine_dropout_noise(frame, groups, sensors, medians, scales, *, seed, noise_fraction):
    result = frame.copy()
    for name in sensors:
        result[name] = result[name].astype(float)
    columns = list(dict.fromkeys(groups[0] + groups[-1]))
    result.loc[:, columns] = [medians[name] for name in columns]
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(len(result), len(sensors)))
    noise *= np.asarray([scales[name] for name in sensors]) * noise_fraction
    result.loc[:, sensors] = result[sensors].to_numpy() + noise
    result.loc[:, columns] = [medians[name] for name in columns]
    return result


def _combine_drift_missing(frame, sensors, medians, scales, *, horizon, start_cycle, block_length):
    result = frame.copy()
    for name in sensors:
        result[name] = result[name].astype(float)
    elapsed = np.maximum(result["cycle"].to_numpy(dtype=float) - 1.0, 0.0)
    ramp = np.minimum(elapsed / horizon, 1.0)
    for name in sensors:
        result[name] += 2.0 * scales[name] * ramp
    missing = result["cycle"].between(start_cycle, start_cycle + block_length - 1)
    result.loc[missing, sensors] = [medians[name] for name in sensors]
    return result


def window_health_scores(training_windows: np.ndarray, evaluation_windows: np.ndarray) -> np.ndarray:
    """Score sensor-window departure using only the source/training window distribution."""
    if not len(training_windows) or not len(evaluation_windows):
        return np.zeros(len(evaluation_windows), dtype=float)
    train_rows = training_windows.reshape(-1, training_windows.shape[-1]).astype(float)
    center = np.median(train_rows, axis=0)
    q25, q75 = np.quantile(train_rows, [0.25, 0.75], axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-3)
    eval_center = np.median(evaluation_windows, axis=1)
    robust_z = np.abs((eval_center - center) / scale)
    exceed = robust_z > 3.0
    return exceed.mean(axis=1) + 0.05 * np.maximum(robust_z - 3.0, 0).mean(axis=1)


def validation_abstention_thresholds(
    validation_scores: np.ndarray,
    fractions: tuple[float, ...] = ABSTENTION_FRACTIONS,
) -> dict[float, float]:
    scores = np.asarray(validation_scores, dtype=float).ravel()
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Validation health scores must be non-empty and finite")
    return {
        float(fraction): float(np.quantile(scores, 1.0 - fraction, method="lower"))
        if fraction
        else float("inf")
        for fraction in fractions
    }


def risk_coverage_rows(
    truth: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[float, float],
) -> list[dict]:
    truth = np.asarray(truth, dtype=float).ravel()
    pred = np.asarray(predictions, dtype=float).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if not (len(truth) == len(pred) == len(scores)):
        raise ValueError("Risk/coverage inputs must have equal lengths")
    rows = []
    for requested, threshold in thresholds.items():
        flagged = scores > threshold if requested else np.zeros(len(scores), dtype=bool)
        retained = ~flagged
        rows.append(
            {
                "requested_abstention_fraction": requested,
                "validation_score_threshold": threshold,
                "n_predictions": int(len(scores)),
                "n_flagged": int(flagged.sum()),
                "flagged_fraction": float(flagged.mean()) if len(flagged) else 0.0,
                "coverage_fraction": float(retained.mean()) if len(retained) else 0.0,
                "retained_mae": (
                    float(np.mean(np.abs(truth[retained] - pred[retained])))
                    if retained.any()
                    else float("nan")
                ),
            }
        )
    return rows


def online_stage_predictions(
    train_windows: np.ndarray,
    train_targets: np.ndarray,
    train_units: np.ndarray,
    train_cycles: np.ndarray,
    train_lifetimes: dict[int, float],
    evaluation_windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict life stage from each current window; test lifetime/RUL is never consulted."""
    units = np.asarray(train_units, dtype=int)
    cycles = np.asarray(train_cycles, dtype=float)
    lifetime = np.asarray([train_lifetimes[int(unit)] for unit in units], dtype=float)
    fraction = np.clip(cycles / np.maximum(lifetime, 1.0), 0.0, 1.0)
    labels = np.asarray(LIFE_STAGES, dtype=object)[np.minimum((fraction * 3).astype(int), 2)]
    summaries = WindowRidgeBaseline.summarize(train_windows)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    classifier.fit(summaries, labels)
    predicted_stages = classifier.predict(WindowRidgeBaseline.summarize(evaluation_windows))

    shared = WindowRidgeBaseline().fit(train_windows, train_targets)
    predictions = shared.predict(evaluation_windows)
    for stage in LIFE_STAGES:
        mask = labels == stage
        if int(mask.sum()) < 50 or not np.any(predicted_stages == stage):
            continue
        specialized = WindowRidgeBaseline().fit(train_windows[mask], np.asarray(train_targets)[mask])
        evaluation_mask = predicted_stages == stage
        predictions[evaluation_mask] = specialized.predict(evaluation_windows[evaluation_mask])
    return predicted_stages, predictions
