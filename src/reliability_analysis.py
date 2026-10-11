from __future__ import annotations

import argparse
import json
import logging
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_model import WindowRidgeBaseline
from src.config import DATASET_NAMES, PROJECT_ROOT, ProjectConfig
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.data_splitter import DataSplitter
from src.data_validator import DataValidator
from src.evaluator import Evaluator
from src.normalization import Normalizer
from src.reliability import (
    ABSTENTION_FRACTIONS,
    CorruptionScenario,
    leave_one_subset_sources,
    online_stage_predictions,
    prediction_frame,
    risk_coverage_rows,
    sensor_corruption_scenarios,
    terminal_metrics,
    terminal_windows,
    training_augmentation,
    validation_abstention_thresholds,
    validation_censor_windows,
    window_health_scores,
)
from src.sequence_generator import SequenceGenerator
from src.target_construction import TargetConstructor
from src.trainer import Trainer
from src.uncertainty import SplitConformalRUL

logger = logging.getLogger(__name__)
MODEL_NAMES = ("ridge_clean", "ridge_augmented", "lstm_clean")


def _bound(values: np.ndarray, config: ProjectConfig) -> np.ndarray:
    upper = config.rul_clip_value if config.clip_rul else None
    return np.clip(values, 0.0, upper) if upper is not None else np.maximum(values, 0.0)


def _predict(model, values: np.ndarray, config: ProjectConfig) -> np.ndarray:
    if isinstance(model, WindowRidgeBaseline):
        predictions = model.predict(values)
    else:
        predictions = np.asarray(
            model.model.predict(values, batch_size=config.batch_size, verbose=0),
            dtype=np.float32,
        ).ravel()
    return _bound(predictions, config)


def _validate_and_label(dataset: str, config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = DataLoader(replace(config, dataset=dataset))
    raw_train = loader.load_train_data()
    raw_test = loader.load_test_data()
    for split, frame in (("train", raw_train), ("test", raw_test)):
        valid, issues = DataValidator.validate(frame, dataset)
        if not valid:
            raise ValueError(f"Invalid {dataset} {split} data: {'; '.join(issues)}")
    final_rul = loader.load_rul_data()
    if raw_test["unit"].nunique() != len(final_rul):
        raise ValueError(f"{dataset} test trajectories and RUL labels do not align")
    cap = config.rul_clip_value if config.clip_rul else None
    return (
        TargetConstructor.compute_training_rul(raw_train, cap),
        TargetConstructor.compute_test_rul(raw_test, final_rul, cap),
    )


def _fit_suite(
    labeled_training: pd.DataFrame,
    config: ProjectConfig,
    *,
    seed: int,
    train_lstm: bool,
) -> dict:
    seeded_config = replace(config, random_seed=seed)
    train_frame, validation_frame = DataSplitter(seeded_config).split_engines(labeled_training)
    train_frame, removed = DataCleaner(seeded_config).remove_constant_sensors(train_frame)
    validation_frame = validation_frame.drop(columns=removed, errors="ignore")
    features = [name for name in train_frame if name not in {"unit", "cycle", "RUL"}]
    if not features:
        raise ValueError("No sensor or operating-setting features remain after cleaning")

    normalizer = Normalizer(seeded_config).fit(train_frame, features)
    train_scaled = normalizer.transform(train_frame, features)
    validation_scaled = normalizer.transform(validation_frame, features)
    generator = SequenceGenerator(seeded_config)
    x_train, y_train, train_units, train_cycles = generator.generate_with_metadata(
        train_scaled, features
    )
    x_validation, y_validation, _, _ = generator.generate_with_metadata(
        validation_scaled, features
    )
    if not len(x_train) or not len(x_validation):
        raise ValueError("The engine split produced an empty window set")

    ridge = WindowRidgeBaseline().fit(x_train, y_train)
    augmented_x, augmented_y = training_augmentation(x_train, y_train, seed=seed + 500)
    augmented_ridge = WindowRidgeBaseline().fit(augmented_x, augmented_y)
    models = {"ridge_clean": ridge, "ridge_augmented": augmented_ridge}
    if train_lstm:
        import tensorflow as tf

        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        lstm_config = replace(seeded_config, random_seed=seed)
        from src.lstm_model import LSTMModel

        lstm = LSTMModel(lstm_config, input_shape=(config.window_size, len(features)))
        Trainer(lstm_config).train(lstm, x_train, y_train, x_validation, y_validation)
        models["lstm_clean"] = lstm

    censor_x, censor_units, censor_cycles, censor_y = validation_censor_windows(
        validation_scaled,
        features,
        config.window_size,
        seed=seed + 1700,
        max_rul=float(config.rul_clip_value if config.clip_rul else 125),
    )
    calibration = {
        name: _predict(model, censor_x, seeded_config) for name, model in models.items()
    }
    validation_window_metrics = {
        name: Evaluator.compute_metrics(
            y_validation, _predict(model, x_validation, seeded_config)
        )
        for name, model in models.items()
    }
    clean_model_names = [name for name in ("ridge_clean", "lstm_clean") if name in models]
    selected_clean_model = min(
        clean_model_names,
        key=lambda name: validation_window_metrics[name]["mae"],
    )
    health_validation = window_health_scores(x_train, censor_x)
    abstention_thresholds = validation_abstention_thresholds(health_validation)
    train_lifetimes = {
        int(unit): float(cycles)
        for unit, cycles in train_frame.groupby("unit", sort=False)["cycle"].max().items()
    }
    return {
        "models": models,
        "features": features,
        "removed_sensors": removed,
        "normalizer": normalizer,
        "train_frame": train_frame,
        "validation_frame": validation_frame,
        "train_windows": x_train,
        "train_targets": y_train,
        "train_units": train_units,
        "train_cycles": train_cycles,
        "train_scaled": train_scaled,
        "validation_scaled": validation_scaled,
        "validation_windows": x_validation,
        "validation_targets": y_validation,
        "calibration_windows": censor_x,
        "calibration_units": censor_units,
        "calibration_cycles": censor_cycles,
        "calibration_targets": censor_y,
        "calibration_predictions": calibration,
        "validation_window_metrics": validation_window_metrics,
        "selected_clean_model": selected_clean_model,
        "abstention_thresholds": abstention_thresholds,
        "health_validation": health_validation,
        "train_lifetimes": train_lifetimes,
        "seed": seed,
        "config": seeded_config,
    }


def _evaluation_windows(frame: pd.DataFrame, suite: dict) -> tuple:
    scaled = suite["normalizer"].transform(frame, suite["features"])
    return terminal_windows(scaled, suite["features"], suite["config"].window_size), scaled


def _model_interval(suite: dict, model_name: str, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    calibration = SplitConformalRUL(nominal_coverage=0.95).fit(
        suite["calibration_targets"], suite["calibration_predictions"][model_name]
    )
    lower, upper = calibration.predict_interval(
        predictions,
        max_rul=suite["config"].rul_clip_value if suite["config"].clip_rul else None,
    )
    return lower, upper, float(calibration.radius or 0.0)


def _metric_row(
    frame: pd.DataFrame,
    *,
    dataset: str,
    model: str,
    protocol: str,
    suite: dict,
    source_subsets: tuple[str, ...] = (),
) -> dict:
    return {
        "dataset": dataset,
        "model": model,
        "protocol": protocol,
        "seed": suite["seed"],
        "selected_clean_model": suite["selected_clean_model"],
        "source_subsets": ";".join(source_subsets),
        **terminal_metrics(frame),
    }


def _save_model_artifacts(suite: dict, output_dir: Path, dataset: str) -> None:
    import joblib

    for name, model in suite["models"].items():
        target = output_dir / "models" / dataset / f"seed-{suite['seed']}"
        target.mkdir(parents=True, exist_ok=True)
        if isinstance(model, WindowRidgeBaseline):
            model.save(target / f"{name}.joblib")
        else:
            model.save(str(target / f"{name}.keras"))
    suite["normalizer"].save(
        output_dir / "models" / dataset / f"seed-{suite['seed']}" / "scaler.joblib"
    )


def _targets_unchanged(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    reference = before.set_index(["unit", "cycle"])["RUL"]
    keys = pd.MultiIndex.from_frame(after[["unit", "cycle"]])
    if not keys.isin(reference.index).all():
        return False
    expected = reference.reindex(keys).to_numpy(dtype=float)
    observed = after["RUL"].to_numpy(dtype=float)
    return bool(np.array_equal(expected, observed))


def _run_in_domain(
    datasets: tuple[str, ...],
    config: ProjectConfig,
    seeds: tuple[int, ...],
    lstm_seeds: set[int],
) -> tuple[list[dict], list[pd.DataFrame], list[dict], list[pd.DataFrame], list[dict], list[pd.DataFrame]]:
    repeated_rows, terminal_outputs = [], []
    stress_rows, stress_outputs = [], []
    online_rows, online_outputs = [], []
    for dataset in datasets:
        training, testing = _validate_and_label(dataset, config)
        for seed in seeds:
            suite = _fit_suite(
                training,
                replace(config, dataset=dataset),
                seed=seed,
                train_lstm=seed in lstm_seeds,
            )
            (test_windows, test_units, test_cycles, test_truth), test_scaled = _evaluation_windows(
                testing, suite
            )
            if not len(test_windows):
                raise ValueError(f"{dataset} contains no complete terminal test windows")
            generator = SequenceGenerator(suite["config"])
            overlapping_x, overlapping_y, overlapping_units, overlapping_cycles = generator.generate_with_metadata(
                test_scaled, suite["features"]
            )
            clean_predictions = {
                name: _predict(model, test_windows, suite["config"])
                for name, model in suite["models"].items()
            }
            overlap_predictions = {
                name: _predict(model, overlapping_x, suite["config"])
                for name, model in suite["models"].items()
            }
            for model_name, prediction in clean_predictions.items():
                lower, upper, radius = _model_interval(suite, model_name, prediction)
                frame = prediction_frame(
                    dataset,
                    model_name,
                    test_units,
                    test_cycles,
                    test_truth,
                    prediction,
                    lower=lower,
                    upper=upper,
                    scenario="clean",
                )
                frame.insert(1, "seed", seed)
                terminal_outputs.append(frame)
                row = _metric_row(
                    frame,
                    dataset=dataset,
                    model=model_name,
                    protocol="in_domain_terminal_engine",
                    suite=suite,
                )
                row["validation_calibration_radius"] = radius
                repeated_rows.append(row)
                overlap = Evaluator.compute_metrics(overlapping_y, overlap_predictions[model_name])
                repeated_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "protocol": "in_domain_overlapping_windows",
                        "seed": seed,
                        "selected_clean_model": suite["selected_clean_model"],
                        "source_subsets": dataset,
                        "overlapping_window_mae": overlap["mae"],
                        "overlapping_window_rmse": overlap["rmse"],
                        "overlapping_window_phm08_sum": overlap["phm08_score"],
                        "n_windows": overlap["n_samples"],
                    }
                )

            if seed == seeds[0]:
                _save_model_artifacts(suite, config.output_dir / "reliability", dataset)
                stress_scenarios = [
                    CorruptionScenario("clean", "clean", "none", lambda frame: frame.copy()),
                    *sensor_corruption_scenarios(
                        suite["train_frame"], suite["features"], seed=seed + 2500
                    ),
                ]
                for scenario in stress_scenarios:
                    stress_frame = scenario.apply(testing)
                    if not _targets_unchanged(testing, stress_frame):
                        raise AssertionError(f"{scenario.name} modified test targets")
                    terminal_x, units, cycles, truth = terminal_windows(
                        suite["normalizer"].transform(stress_frame, suite["features"]),
                        suite["features"],
                        suite["config"].window_size,
                    )
                    if not len(terminal_x):
                        continue
                    anomaly = window_health_scores(suite["train_windows"], terminal_x)
                    for model_name, model in suite["models"].items():
                        prediction = _predict(model, terminal_x, suite["config"])
                        lower, upper, radius = _model_interval(suite, model_name, prediction)
                        frame = prediction_frame(
                            dataset,
                            model_name,
                            units,
                            cycles,
                            truth,
                            prediction,
                            lower=lower,
                            upper=upper,
                            scenario=scenario.name,
                            kind=scenario.kind,
                            severity=scenario.severity,
                        )
                        frame["anomaly_score"] = anomaly
                        stress_outputs.append(frame)
                        row = {
                            "dataset": dataset,
                            "model": model_name,
                            "scenario": scenario.name,
                            "corruption_type": scenario.kind,
                            "severity": scenario.severity,
                            "seed": seed,
                            "calibration_radius_cycles": radius,
                            **terminal_metrics(frame),
                        }
                        threshold = suite["abstention_thresholds"][0.2]
                        flagged = anomaly >= threshold
                        row["abstention_threshold_validation_20pct"] = threshold
                        row["flagged_predictions"] = int(flagged.sum())
                        row["flagged_fraction"] = float(flagged.mean())
                        stress_rows.append(row)
                        for risk_row in risk_coverage_rows(
                            truth, prediction, anomaly, suite["abstention_thresholds"]
                        ):
                            stress_rows.append(
                                {
                                    "dataset": dataset,
                                    "model": model_name,
                                    "scenario": scenario.name,
                                    "corruption_type": scenario.kind,
                                    "severity": scenario.severity,
                                    "seed": seed,
                                    "record_type": "risk_coverage",
                                    **risk_row,
                                }
                            )

                stage_predictions, online_prediction = online_stage_predictions(
                    suite["train_windows"],
                    suite["train_targets"],
                    suite["train_units"],
                    suite["train_cycles"],
                    suite["train_lifetimes"],
                    overlapping_x,
                )
                shared_prediction = _predict(
                    suite["models"]["ridge_clean"], overlapping_x, suite["config"]
                )
                for stage in ("early", "middle", "late"):
                    stage_mask = stage_predictions == stage
                    if not stage_mask.any():
                        continue
                    for model_name, prediction in (
                        ("shared_ridge", shared_prediction),
                        ("online_stage_ridge", _bound(online_prediction, suite["config"])),
                    ):
                        online_rows.append(
                            {
                                "dataset": dataset,
                                "model": model_name,
                                "protocol": "all_test_windows_grouped_by_online_predicted_stage",
                                "predicted_stage": stage,
                                "seed": seed,
                                **Evaluator.compute_metrics(
                                    overlapping_y[stage_mask], prediction[stage_mask]
                                ),
                            }
                        )
                for model_name, prediction in (
                    ("shared_ridge", shared_prediction),
                    ("online_stage_ridge", _bound(online_prediction, suite["config"])),
                ):
                    values = Evaluator.compute_metrics(overlapping_y, prediction)
                    online_rows.append(
                        {
                            "dataset": dataset,
                            "model": model_name,
                            "protocol": "all_test_windows_grouped_by_online_predicted_stage",
                            "seed": seed,
                            **values,
                        }
                    )
                stage_frame = pd.DataFrame(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "unit": overlapping_units,
                        "cycle": overlapping_cycles,
                        "rul_true": overlapping_y,
                        "predicted_online_stage": stage_predictions,
                        "rul_predicted_online_stage": _bound(online_prediction, suite["config"]),
                        "rul_predicted_shared_ridge": shared_prediction,
                    }
                )
                online_outputs.append(stage_frame)
                logger.info("Completed repeated in-domain run %s seed=%d", dataset, seed)

    return (
        repeated_rows,
        terminal_outputs,
        stress_rows,
        stress_outputs,
        online_rows,
        online_outputs,
    )


def _pool_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    pooled, offset = [], 0
    for frame in frames:
        item = frame.copy()
        item["unit"] = item["unit"].astype(int) + offset
        pooled.append(item)
        offset = int(item["unit"].max()) + 1
    return pd.concat(pooled, ignore_index=True)


def _compare_existing_protocols(heldout: pd.DataFrame, benchmark_path: Path) -> pd.DataFrame:
    if heldout.empty or not benchmark_path.is_file():
        return pd.DataFrame()
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    transfer_rows = benchmark.get("transfer", {}).get("rows", [])
    comparisons = []
    for _, row in heldout.iterrows():
        dataset = row["dataset"]
        baseline = benchmark.get("datasets", {}).get(dataset, {})
        ridge = baseline.get("model_metrics", {}).get("ridge", {})
        selected = baseline.get("test_metrics", {})
        same_subset_transfer = next(
            (
                item
                for item in transfer_rows
                if item.get("source_dataset") == dataset
                and item.get("target_dataset") == dataset
                and item.get("model") == "ridge"
            ),
            {},
        )
        comparisons.append(
            {
                "dataset": dataset,
                "heldout_model": row["model"],
                "heldout_source_subsets": row.get("source_subsets", ""),
                "heldout_terminal_engine_mae": row.get("terminal_engine_mae"),
                "heldout_terminal_engine_rmse": row.get("terminal_engine_rmse"),
                "heldout_phm08_score_sum": row.get("phm08_score_sum"),
                "existing_in_domain_ridge_overlapping_window_mae": ridge.get("mae"),
                "existing_in_domain_ridge_overlapping_window_rmse": ridge.get("rmse"),
                "existing_in_domain_selected_model": baseline.get("selected_model"),
                "existing_in_domain_selected_overlapping_window_mae": selected.get("mae"),
                "existing_in_domain_selected_overlapping_window_rmse": selected.get("rmse"),
                "existing_same_subset_transfer_ridge_overlapping_window_mae": same_subset_transfer.get("mae"),
                "existing_same_subset_transfer_ridge_overlapping_window_rmse": same_subset_transfer.get("rmse"),
                "protocol_note": (
                    "Held-out-source terminal-engine metrics are not directly comparable to the prior "
                    "overlapping-window in-domain/transfer figures."
                ),
            }
        )
    return pd.DataFrame(comparisons)


def _run_heldout_domains(
    datasets: tuple[str, ...],
    config: ProjectConfig,
    seed: int,
    *,
    train_lstm: bool,
) -> tuple[list[dict], list[pd.DataFrame], list[dict], list[pd.DataFrame]]:
    domains = {name: _validate_and_label(name, config) for name in datasets}
    metric_rows, prediction_frames, stress_rows, stress_frames = [], [], [], []
    for target in datasets:
        sources = leave_one_subset_sources(datasets, target)
        if target in sources or len(sources) != len(datasets) - 1:
            raise AssertionError("Leave-one-subset-out source exclusion failed")
        pooled_source_training = _pool_sources([domains[name][0] for name in sources])
        suite = _fit_suite(
            pooled_source_training,
            replace(config, dataset=target),
            seed=seed,
            train_lstm=train_lstm,
        )
        target_test = domains[target][1].drop(
            columns=suite["removed_sensors"], errors="ignore"
        )
        terminal_x, units, cycles, truth = terminal_windows(
            suite["normalizer"].transform(target_test, suite["features"]),
            suite["features"],
            suite["config"].window_size,
        )
        if not len(terminal_x):
            raise ValueError(f"{target} has no complete windows for held-out evaluation")
        for model_name, model in suite["models"].items():
            prediction = _predict(model, terminal_x, suite["config"])
            lower, upper, radius = _model_interval(suite, model_name, prediction)
            frame = prediction_frame(
                target,
                model_name,
                units,
                cycles,
                truth,
                prediction,
                lower=lower,
                upper=upper,
                scenario="heldout_clean",
            )
            frame["protocol"] = "leave_one_subset_out"
            frame["source_subsets"] = ";".join(sources)
            prediction_frames.append(frame)
            row = _metric_row(
                frame,
                dataset=target,
                model=model_name,
                protocol="leave_one_subset_out_terminal_engine",
                suite=suite,
                source_subsets=sources,
            )
            row["calibration_radius_cycles"] = radius
            row["target_training_used"] = False
            metric_rows.append(row)

        target_frame = target_test
        scenarios = sensor_corruption_scenarios(
            suite["train_frame"], suite["features"], seed=seed + 3900
        )
        for scenario in scenarios:
            if scenario.kind != "combined":
                continue
            corrupted = scenario.apply(target_frame)
            if not _targets_unchanged(target_frame, corrupted):
                raise AssertionError(f"{scenario.name} modified held-out targets")
            stress_x, stress_units, stress_cycles, stress_truth = terminal_windows(
                suite["normalizer"].transform(corrupted, suite["features"]),
                suite["features"],
                suite["config"].window_size,
            )
            anomaly = window_health_scores(suite["train_windows"], stress_x)
            for model_name, model in suite["models"].items():
                prediction = _predict(model, stress_x, suite["config"])
                lower, upper, radius = _model_interval(suite, model_name, prediction)
                frame = prediction_frame(
                    target,
                    model_name,
                    stress_units,
                    stress_cycles,
                    stress_truth,
                    prediction,
                    lower=lower,
                    upper=upper,
                    scenario=scenario.name,
                    kind=scenario.kind,
                    severity=scenario.severity,
                )
                frame["anomaly_score"] = anomaly
                frame["source_subsets"] = ";".join(sources)
                stress_frames.append(frame)
                row = {
                    "dataset": target,
                    "model": model_name,
                    "protocol": "leave_one_subset_out_corrupted_terminal_engine",
                    "source_subsets": ";".join(sources),
                    "target_training_used": False,
                    "scenario": scenario.name,
                    "corruption_type": scenario.kind,
                    "severity": scenario.severity,
                    "seed": seed,
                    "calibration_radius_cycles": radius,
                    **terminal_metrics(frame),
                }
                threshold = suite["abstention_thresholds"][0.2]
                flagged = anomaly >= threshold
                row["abstention_threshold_validation_20pct"] = threshold
                row["flagged_predictions"] = int(flagged.sum())
                row["flagged_fraction"] = float(flagged.mean())
                stress_rows.append(row)
        logger.info("Completed held-out target %s; sources=%s", target, ",".join(sources))
    return metric_rows, prediction_frames, stress_rows, stress_frames


def _write_plots(stress_metrics: pd.DataFrame, risk_metrics: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    record_type = stress_metrics.get(
        "record_type", pd.Series(index=stress_metrics.index, dtype=object)
    )
    records = stress_metrics.loc[record_type != "risk_coverage"]
    if not records.empty:
        datasets = sorted(records["dataset"].dropna().unique())
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
        for axis, dataset in zip(axes.flat, datasets, strict=False):
            subset = records[records["dataset"] == dataset]
            summary = subset.groupby(["corruption_type", "model"], as_index=False)[
                "terminal_engine_mae"
            ].mean()
            kinds = list(summary["corruption_type"].drop_duplicates())
            x = np.arange(len(kinds))
            models = list(summary["model"].drop_duplicates())
            width = 0.8 / max(len(models), 1)
            for model_index, model in enumerate(models):
                values = [
                    float(summary.loc[
                        (summary["corruption_type"] == kind) & (summary["model"] == model),
                        "terminal_engine_mae",
                    ].mean())
                    for kind in kinds
                ]
                axis.bar(x - 0.4 + width * (model_index + 0.5), values, width, label=model)
            axis.set_title(dataset)
            axis.set_ylabel("Terminal-engine MAE (cycles)")
            axis.set_xticks(x, kinds, rotation=30, ha="right")
            axis.grid(axis="y", alpha=0.2)
        if datasets:
            axes.flat[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures_dir / "sensor_corruption_terminal_mae.png", dpi=160)
        plt.close(fig)

    if not risk_metrics.empty:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
        for axis, dataset in zip(axes.flat, sorted(risk_metrics["dataset"].dropna().unique()), strict=False):
            subset = risk_metrics[risk_metrics["dataset"] == dataset]
            for model, group in subset.groupby("model"):
                summary = group.groupby("requested_abstention_fraction", as_index=False).agg(
                    coverage=("coverage_fraction", "mean"), risk=("retained_mae", "mean")
                )
                axis.plot(summary["coverage"], summary["risk"], marker="o", label=model)
            axis.set_title(dataset)
            axis.set_xlabel("Retained prediction coverage")
            axis.set_ylabel("Retained terminal MAE (cycles)")
            axis.grid(alpha=0.2)
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "validation_threshold_risk_coverage.png", dpi=160)
        plt.close(fig)


def run_reliability_study(
    datasets: tuple[str, ...] = DATASET_NAMES,
    config: ProjectConfig | None = None,
    *,
    seeds: tuple[int, ...] = (42, 43, 44),
    lstm_seeds: tuple[int, ...] | None = None,
    run_heldout_domains: bool = True,
) -> dict:
    datasets = tuple(dict.fromkeys(name.upper() for name in datasets))
    if not datasets or any(name not in DATASET_NAMES for name in datasets):
        raise ValueError("Select one or more valid subsets: FD001-FD004")
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    lstm_seeds = tuple(int(seed) for seed in (seeds[:1] if lstm_seeds is None else lstm_seeds))
    if not seeds or not set(lstm_seeds).issubset(seeds):
        raise ValueError("LSTM seeds must be included in the repeated seed list")
    config = config or ProjectConfig(dataset=datasets[0], max_epochs=20)
    output_dir = config.output_dir / "reliability"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    configuration = {
        "schema_version": 1,
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "datasets": datasets,
        "repeated_seeds": seeds,
        "lstm_seeds": lstm_seeds,
        "window_size": config.window_size,
        "rul_cap": config.rul_clip_value if config.clip_rul else None,
        "validation_protocol": "engine split; deterministic pseudo-censored validation endpoint per engine",
        "test_protocol": "official test labels are used only for final evaluation",
        "augmentation": {
            "copies_per_training_window": 1,
            "channel_dropout_probability": 0.08,
            "gaussian_noise_fraction_of_train_feature_std": 0.05,
            "targets_changed": False,
        },
        "corruptions": {
            "individual_dropout": "each retained sensor independently replaced by its source-training median at 60% of observed points",
            "grouped_dropout_probabilities": {"mild": 0.25, "moderate": 0.6, "severe": 1.0},
            "sensor_groups": ["sensor_1..sensor_7", "sensor_8..sensor_14", "sensor_15..sensor_21"],
            "stuck_sensor_onset_cycles": {"mild": 180, "moderate": 120, "severe": 60},
            "drift_multiples_of_training_sensor_sd": {"mild": 0.5, "moderate": 1.0, "severe": 2.0},
            "drift_ramp_cycles": 200,
            "noise_multiples_of_training_sensor_sd": {"mild": 0.1, "moderate": 0.25, "severe": 0.5},
            "median_imputed_missing_block_cycles": {"mild": 10, "moderate": 25, "severe": 50},
            "reduced_sampling_strides": {"mild": 2, "moderate": 3, "severe": 4},
            "combined": ["severe dropout plus moderate sensor noise", "severe drift plus severe missing block"],
            "all_corruptions_preserve_available_targets": True,
        },
        "abstention": {
            "health_score": "window median outside training median +/- 3 training IQR-derived scales",
            "requested_abstention_fractions": ABSTENTION_FRACTIONS,
            "thresholds_fit_on": "validation engine pseudo-censored windows",
            "flagged_fraction_summary_threshold": 0.2,
        },
        "domain_protocol": "leave one FD subset out; fit, preprocessing, and calibration use other subsets only",
        "domain_note": "FD001-FD004 are related C-MAPSS simulator subsets, not independent real fleets.",
        "sensor_corruption_seed": 2542,
    }
    (output_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2, default=list) + "\n", encoding="utf-8"
    )

    (
        repeated_rows,
        terminal_frames,
        stress_rows,
        stress_frames,
        online_rows,
        online_frames,
    ) = _run_in_domain(datasets, config, seeds, set(lstm_seeds))
    heldout_rows: list[dict] = []
    heldout_frames: list[pd.DataFrame] = []
    heldout_stress_rows: list[dict] = []
    heldout_stress_frames: list[pd.DataFrame] = []
    if run_heldout_domains and len(datasets) > 1:
        (
            heldout_rows,
            heldout_frames,
            heldout_stress_rows,
            heldout_stress_frames,
        ) = _run_heldout_domains(
            datasets,
            config,
            seeds[0],
            train_lstm=seeds[0] in set(lstm_seeds),
        )

    repeated = pd.DataFrame(repeated_rows)
    terminals = pd.concat(terminal_frames, ignore_index=True) if terminal_frames else pd.DataFrame()
    stress = pd.DataFrame(stress_rows)
    stress_predictions = pd.concat(stress_frames, ignore_index=True) if stress_frames else pd.DataFrame()
    online = pd.DataFrame(online_rows)
    online_predictions = pd.concat(online_frames, ignore_index=True) if online_frames else pd.DataFrame()
    heldout = pd.DataFrame(heldout_rows)
    heldout_predictions = pd.concat(heldout_frames, ignore_index=True) if heldout_frames else pd.DataFrame()
    heldout_stress = pd.DataFrame(heldout_stress_rows)
    heldout_stress_predictions = (
        pd.concat(heldout_stress_frames, ignore_index=True)
        if heldout_stress_frames
        else pd.DataFrame()
    )

    repeated.to_csv(metrics_dir / "repeated_seed_metrics.csv", index=False)
    terminal_repeats = repeated[repeated["protocol"] == "in_domain_terminal_engine"]
    if not terminal_repeats.empty:
        repeated_summary = terminal_repeats.groupby(["dataset", "model"], as_index=False).agg(
            seed_count=("seed", "nunique"),
            terminal_mae_mean=("terminal_engine_mae", "mean"),
            terminal_mae_sd=("terminal_engine_mae", "std"),
            terminal_rmse_mean=("terminal_engine_rmse", "mean"),
            terminal_rmse_sd=("terminal_engine_rmse", "std"),
            phm08_sum_mean=("phm08_score_sum", "mean"),
            phm08_sum_sd=("phm08_score_sum", "std"),
            interval_coverage_mean=("interval_coverage", "mean"),
            interval_coverage_sd=("interval_coverage", "std"),
        )
        repeated_summary.to_csv(metrics_dir / "repeated_seed_summary.csv", index=False)
    else:
        repeated_summary = pd.DataFrame()
    terminals.to_csv(predictions_dir / "terminal_engine_predictions.csv", index=False)
    stress.to_csv(metrics_dir / "sensor_stress_metrics.csv", index=False)
    stress_predictions.to_csv(predictions_dir / "sensor_stress_terminal_predictions.csv", index=False)
    online.to_csv(metrics_dir / "online_stage_metrics.csv", index=False)
    online_predictions.to_csv(predictions_dir / "online_stage_predictions.csv", index=False)
    heldout.to_csv(metrics_dir / "leave_one_subset_out_metrics.csv", index=False)
    heldout_predictions.to_csv(predictions_dir / "leave_one_subset_out_predictions.csv", index=False)
    heldout_stress.to_csv(metrics_dir / "leave_one_subset_out_stress_metrics.csv", index=False)
    heldout_stress_predictions.to_csv(
        predictions_dir / "leave_one_subset_out_stress_predictions.csv", index=False
    )
    heldout_comparison = _compare_existing_protocols(heldout, PROJECT_ROOT / "web" / "public" / "results" / "benchmark.json")
    heldout_comparison.to_csv(metrics_dir / "heldout_protocol_comparison.csv", index=False)
    risk = stress.loc[stress.get("record_type", pd.Series(index=stress.index, dtype=object)) == "risk_coverage"].copy()
    risk.to_csv(metrics_dir / "risk_coverage.csv", index=False)
    _write_plots(stress, risk, output_dir)

    summary = {
        **configuration,
        "outputs": {
            "repeated_metrics": str(metrics_dir / "repeated_seed_metrics.csv"),
            "repeated_seed_summary": str(metrics_dir / "repeated_seed_summary.csv"),
            "terminal_predictions": str(predictions_dir / "terminal_engine_predictions.csv"),
            "sensor_stress_metrics": str(metrics_dir / "sensor_stress_metrics.csv"),
            "sensor_stress_predictions": str(predictions_dir / "sensor_stress_terminal_predictions.csv"),
            "risk_coverage": str(metrics_dir / "risk_coverage.csv"),
            "online_stage_metrics": str(metrics_dir / "online_stage_metrics.csv"),
            "heldout_domain_metrics": str(metrics_dir / "leave_one_subset_out_metrics.csv"),
            "heldout_domain_predictions": str(predictions_dir / "leave_one_subset_out_predictions.csv"),
            "heldout_protocol_comparison": str(metrics_dir / "heldout_protocol_comparison.csv"),
            "figures": [str(path) for path in (output_dir / "figures").glob("*.png")],
        },
        "limitations": [
            "Split-conformal intervals are calibrated on validation engines and are descriptive under shift, not guarantees.",
            "Sensor corruptions are controlled synthetic stressors, not an empirical sensor-failure distribution.",
            "C-MAPSS subsets are related simulator domains, not independent fleets or field telemetry.",
            "N-CMAPSS is not run by this CLI; data access and format are documented separately.",
        ],
    }
    (metrics_dir / "reliability_study.json").write_text(
        json.dumps(summary, indent=2, default=list) + "\n", encoding="utf-8"
    )
    logger.info("Reliability study outputs written beneath %s", output_dir)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run terminal RUL reliability and robustness evaluations")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--lstm-seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-heldout-domains", action="store_true")
    parser.add_argument("--without-lstm", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    lstm_seeds = () if args.without_lstm else tuple(args.lstm_seeds or [args.seeds[0]])
    result = run_reliability_study(
        tuple(args.datasets),
        ProjectConfig(dataset=args.datasets[0], max_epochs=args.epochs),
        seeds=tuple(args.seeds),
        lstm_seeds=lstm_seeds,
        run_heldout_domains=not args.skip_heldout_domains,
    )
    print(json.dumps(result, indent=2, default=list))
