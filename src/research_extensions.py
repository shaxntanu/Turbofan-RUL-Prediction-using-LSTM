from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baseline_model import WindowRidgeBaseline
from src.evaluator import Evaluator


ASYMMETRIC_WEIGHTS = (0.5, 1.0, 2.0)
EMA_ALPHAS = (0.15, 0.35, 0.65)
SHORT_HISTORY_LENGTHS = (10, 20, 30)


def causal_ema_windows(sequences: np.ndarray, alpha: float) -> np.ndarray:
    """Smooth each observed window forward in time, without a future-looking filter."""
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or not 0.0 < alpha <= 1.0:
        raise ValueError("Expected (samples, time, features) windows and alpha in (0, 1]")
    smoothed = np.empty_like(values)
    if not len(values):
        return smoothed
    smoothed[:, 0] = values[:, 0]
    for time_index in range(1, values.shape[1]):
        smoothed[:, time_index] = (
            alpha * values[:, time_index]
            + (1.0 - alpha) * smoothed[:, time_index - 1]
        )
    return smoothed


def select_training_features(
    sequences: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    *,
    top_sensors: int = 10,
) -> tuple[list[int], pd.DataFrame]:
    """Rank sensors by train-window association; always retain operating settings."""
    values = np.asarray(sequences, dtype=float)
    y = np.asarray(targets, dtype=float).ravel()
    if values.ndim != 3 or len(values) != len(y) or values.shape[2] != len(feature_names):
        raise ValueError("Feature-selection inputs have inconsistent shapes")
    if not len(y):
        raise ValueError("Feature selection requires training windows")
    means = values.mean(axis=1)
    rows = []
    for index, name in enumerate(feature_names):
        correlation = np.corrcoef(means[:, index], y)[0, 1]
        score = abs(float(correlation)) if np.isfinite(correlation) else 0.0
        rows.append({"feature": name, "train_abs_target_correlation": score, "index": index})
    ranking = pd.DataFrame(rows).sort_values(
        ["train_abs_target_correlation", "feature"], ascending=[False, True]
    )
    sensor_indices = [
        int(row.index)
        for row in ranking.itertuples()
        if row.feature.startswith("sensor_")
    ][:top_sensors]
    settings = [
        index for index, name in enumerate(feature_names) if name.startswith("setting_")
    ]
    selected = sorted(set(sensor_indices + settings))
    if not selected:
        selected = [int(ranking.iloc[0]["index"])]
    ranking["selected"] = ranking["index"].isin(selected)
    return selected, ranking.reset_index(drop=True)


class SelectedWindowRidge:
    def __init__(self, feature_indices: list[int], alpha: float = 10.0):
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.estimator = WindowRidgeBaseline(alpha=alpha)

    def fit(self, sequences: np.ndarray, targets: np.ndarray):
        self.estimator.fit(sequences[:, :, self.feature_indices], targets)
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        return self.estimator.predict(sequences[:, :, self.feature_indices])


class AsymmetricWindowRidge:
    """Iteratively reweighted Ridge approximation to asymmetric squared loss."""

    def __init__(self, overestimate_weight: float, ridge_alpha: float = 10.0, iterations: int = 6):
        if overestimate_weight < 0 or iterations < 1:
            raise ValueError("Asymmetric weight must be non-negative and iterations positive")
        self.overestimate_weight = float(overestimate_weight)
        self.ridge_alpha = float(ridge_alpha)
        self.iterations = int(iterations)
        self.scaler = StandardScaler()
        self.regressor = Ridge(alpha=self.ridge_alpha)

    def fit(self, sequences: np.ndarray, targets: np.ndarray):
        features = WindowRidgeBaseline.summarize(sequences)
        target = np.asarray(targets, dtype=float).ravel()
        if len(features) != len(target) or not len(target):
            raise ValueError("Asymmetric Ridge requires matching non-empty training data")
        design = self.scaler.fit_transform(features)
        weights = np.ones(len(target), dtype=float)
        for _ in range(self.iterations):
            self.regressor.fit(design, target, sample_weight=weights)
            residual = self.regressor.predict(design) - target
            weights = np.where(residual > 0, 1.0 + self.overestimate_weight, 1.0)
        self.regressor.fit(design, target, sample_weight=weights)
        return self

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        features = WindowRidgeBaseline.summarize(sequences)
        return self.regressor.predict(self.scaler.transform(features)).astype(np.float32)


def select_asymmetric_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    weights: tuple[float, ...] = ASYMMETRIC_WEIGHTS,
) -> tuple[AsymmetricWindowRidge, float, list[dict]]:
    """Choose the overestimation penalty using validation engines only."""
    candidates = []
    for weight in weights:
        model = AsymmetricWindowRidge(weight).fit(x_train, y_train)
        metrics = Evaluator.compute_metrics(y_validation, model.predict(x_validation))
        candidates.append({"overestimate_weight": weight, **metrics, "model": model})
    selected = min(candidates, key=lambda row: (row["phm08_score"], row["mae"]))
    return selected["model"], float(selected["overestimate_weight"]), [
        {key: value for key, value in row.items() if key != "model"} for row in candidates
    ]


def history_windows(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    model_window: int,
    observed_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build endpoint windows from only the most recent available observations.

    If the model expects a longer input, left padding repeats the earliest observed
    row in that prefix. No future or unobserved trajectory values are introduced.
    """
    if model_window < 1 or observed_history < 1 or observed_history > model_window:
        raise ValueError("History must be positive and cannot exceed the model input length")
    sequences, units, cycles, targets = [], [], [], []
    for unit, group in frame.sort_values(["unit", "cycle"]).groupby("unit", sort=False):
        values = group[feature_names].to_numpy(dtype=np.float32, copy=True)
        labels = group["RUL"].to_numpy(dtype=np.float32, copy=True)
        cycle_values = group["cycle"].to_numpy(dtype=np.int32, copy=True)
        for endpoint in range(observed_history - 1, len(group)):
            start = max(0, endpoint - observed_history + 1)
            prefix = values[start : endpoint + 1]
            padding = model_window - len(prefix)
            if padding:
                prefix = np.concatenate(
                    [np.repeat(prefix[:1], padding, axis=0), prefix], axis=0
                )
            sequences.append(prefix)
            units.append(int(unit))
            cycles.append(int(cycle_values[endpoint]))
            targets.append(float(labels[endpoint]))
    if not sequences:
        return (
            np.empty((0, model_window, len(feature_names)), dtype=np.float32),
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


def fit_online_onset_detector(
    sequences: np.ndarray, targets: np.ndarray, *, rul_cap: float
) -> LogisticRegression:
    """Fit an onset classifier on training windows labeled by capped-RUL transition."""
    labels = np.asarray(targets, dtype=float).ravel() < float(rul_cap)
    if len(labels) != len(sequences) or np.unique(labels).size < 2:
        raise ValueError("Onset training requires both pre-onset and post-onset examples")
    detector = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0),
    )
    detector.fit(WindowRidgeBaseline.summarize(sequences), labels.astype(int))
    return detector


def onset_probability(detector, sequences: np.ndarray) -> np.ndarray:
    return detector.predict_proba(WindowRidgeBaseline.summarize(sequences))[:, 1]


def onset_detection_report(
    units: np.ndarray,
    cycles: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    rul_cap: float,
    threshold: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    truth = np.asarray(targets, dtype=float).ravel() < rul_cap
    probability = np.asarray(probabilities, dtype=float).ravel()
    predicted = probability >= threshold
    if not (len(units) == len(cycles) == len(truth) == len(predicted)):
        raise ValueError("Onset evaluation metadata and values must align")
    per_engine = []
    for unit in np.unique(units):
        mask = np.asarray(units) == unit
        order = np.argsort(np.asarray(cycles)[mask])
        engine_cycles = np.asarray(cycles)[mask][order]
        engine_truth = truth[mask][order]
        engine_predicted = predicted[mask][order]
        actual = engine_cycles[np.flatnonzero(engine_truth)[0]] if engine_truth.any() else np.nan
        detected_positions = np.flatnonzero(engine_predicted)
        detected = bool(len(detected_positions))
        detected_cycle = int(engine_cycles[detected_positions[0]]) if detected else np.nan
        per_engine.append(
            {
                "unit": int(unit),
                "onset_cycle_by_rul_cap": actual,
                "first_predicted_onset_cycle": detected_cycle,
                "detected": detected,
                "detection_delay_cycles": detected_cycle - actual if detected and np.isfinite(actual) else np.nan,
                "false_alarm_before_onset": bool(detected and np.isfinite(actual) and detected_cycle < actual),
            }
        )
    events = pd.DataFrame(per_engine)
    summary = {
        "window_accuracy": float(accuracy_score(truth, predicted)),
        "window_precision": float(precision_score(truth, predicted, zero_division=0)),
        "window_recall": float(recall_score(truth, predicted, zero_division=0)),
        "window_f1": float(f1_score(truth, predicted, zero_division=0)),
        "engine_detection_rate": float(events["detected"].mean()) if len(events) else 0.0,
        "engine_false_alarm_fraction": float(events["false_alarm_before_onset"].mean()) if len(events) else 0.0,
        "median_detection_delay_cycles": float(events["detection_delay_cycles"].median()),
        "onset_threshold": float(threshold),
        "rul_cap_used_for_onset_label": float(rul_cap),
        "n_engines": int(len(events)),
    }
    return summary, events


@dataclass
class OnlineOnsetGate:
    detector: object
    shared_model: WindowRidgeBaseline
    post_onset_model: WindowRidgeBaseline

    def predict(self, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probabilities = onset_probability(self.detector, sequences)
        predictions = self.shared_model.predict(sequences)
        detected = probabilities >= 0.5
        if detected.any():
            predictions[detected] = self.post_onset_model.predict(sequences[detected])
        return predictions, probabilities


def fit_online_onset_gate(
    sequences: np.ndarray,
    targets: np.ndarray,
    *,
    rul_cap: float,
) -> OnlineOnsetGate:
    target = np.asarray(targets, dtype=float).ravel()
    detector = fit_online_onset_detector(sequences, target, rul_cap=rul_cap)
    post_onset = target < rul_cap
    if post_onset.sum() < 10:
        raise ValueError("Too few post-onset training windows for a gated regressor")
    shared = WindowRidgeBaseline().fit(sequences, target)
    specialized = WindowRidgeBaseline().fit(sequences[post_onset], target[post_onset])
    return OnlineOnsetGate(detector, shared, specialized)
