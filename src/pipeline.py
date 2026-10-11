from __future__ import annotations

import json
import logging
import os
import random
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_model import WindowRidgeBaseline
from src.config import DATASET_NAMES, ProjectConfig
from src.data_cleaner import DataCleaner
from src.data_loader import DataLoader
from src.data_splitter import DataSplitter
from src.data_validator import DataValidator
from src.evaluator import Evaluator
from src.normalization import Normalizer
from src.research_analysis import run_sensor_screening, summarize_engine_robustness
from src.segmentation_analysis import run_fractional_life_analysis
from src.sequence_generator import SequenceGenerator
from src.target_construction import TargetConstructor
from src.trainer import Trainer
from src.uncertainty import SplitConformalRUL
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    random.seed(seed)
    np.random.seed(seed)


def _save_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, default=str) + "\n", encoding="utf-8")


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "tensorflow", "matplotlib", "seaborn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _keras_predict(model, values: np.ndarray, batch_size: int) -> np.ndarray:
    return np.asarray(model.predict(values, batch_size=batch_size, verbose=0), dtype=np.float32).ravel()


def _write_report(path: Path, result: dict) -> None:
    model_rows = [
        f"| {name} | {values['mae']:.2f} | {values['rmse']:.2f} | {values['phm08_score']:.1f} |"
        for name, values in result["test_metrics"].items()
    ]
    report = [
        f"# {result['dataset']} experiment",
        "",
        f"Run ID: `{result['run_id']}`  ",
        f"Seed: `{result['seed']}`  ",
        f"Selected on validation MAE: **{result['selected_model']}**  ",
        "",
        "## Dataset",
        "",
        f"Training engines: {result['dataset_summary']['train_engines']}; "
        f"validation engines: {result['dataset_summary']['validation_engines']}; "
        f"test engines: {result['dataset_summary']['test_engines']}; "
        f"retained sensor channels: {result['dataset_summary']['retained_sensors']}.",
        "",
        "## Official test performance",
        "",
        "| Model | MAE (cycles) | RMSE (cycles) | PHM08 score (lower is better) |",
        "|---|---:|---:|---:|",
        *model_rows,
        "",
        "## Prediction intervals",
        "",
        f"Validation-calibrated conformal interval: {result['uncertainty']['picp']:.1%} "
        f"observed coverage at {result['uncertainty']['nominal_coverage']:.0%} nominal coverage; "
        f"mean width {result['uncertainty']['mpi_width_cycles']:.1f} cycles; "
        f"CWC {result['uncertainty']['cwc_cycles']:.1f} cycles.",
        "",
        "The research papers use proprietary in-service EHM data and predict TGT. "
        "This run uses NASA C-MAPSS simulated trajectories and predicts RUL; the "
        "reported procedures are adaptations, not a reproduction of their published results.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(report), encoding="utf-8")


def run_pipeline(
    dataset_name: str = "FD001",
    config: ProjectConfig | None = None,
    *,
    train_lstm: bool = True,
) -> dict:
    dataset_name = dataset_name.upper()
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"dataset_name must be one of {', '.join(DATASET_NAMES)}")
    config = config or ProjectConfig(dataset=dataset_name)
    if config.dataset != dataset_name:
        raise ValueError("config.dataset must match dataset_name")
    _set_seeds(config.random_seed)
    config.ensure_output_dirs()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    dataset_output = config.output_dir / dataset_name
    figures_dir = dataset_output / "figures"
    metrics_dir = dataset_output / "metrics"
    predictions_dir = dataset_output / "predictions"
    model_dir = config.model_dir / dataset_name
    for directory in (figures_dir, metrics_dir, predictions_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(config)
    raw_train = loader.load_train_data()
    raw_test = loader.load_test_data()
    final_rul = loader.load_rul_data()
    for split, frame in (("train", raw_train), ("test", raw_test)):
        valid, issues = DataValidator.validate(frame, dataset_name)
        if not valid:
            raise ValueError(f"Invalid {dataset_name} {split} data: {'; '.join(issues)}")
    if raw_test["unit"].nunique() != len(final_rul):
        raise ValueError("The test trajectories and RUL label file have different engine counts")

    train_labeled = TargetConstructor.compute_training_rul(
        raw_train,
        config.rul_clip_value if config.clip_rul else None,
    )
    test_labeled = TargetConstructor.compute_test_rul(
        raw_test,
        final_rul,
        config.rul_clip_value if config.clip_rul else None,
    )
    train_split, val_split = DataSplitter(config).split_engines(train_labeled)

    cleaner = DataCleaner(config)
    train_split, removed_sensors = cleaner.remove_constant_sensors(train_split)
    val_split = val_split.drop(columns=removed_sensors, errors="ignore")
    test_labeled = test_labeled.drop(columns=removed_sensors, errors="ignore")
    sensor_cols = [column for column in train_split if column.startswith("sensor_")]
    feature_cols = [
        column for column in train_split.columns
        if column not in {"unit", "cycle", "RUL"}
    ]
    if not feature_cols:
        raise ValueError(f"No predictor columns remain for {dataset_name}")

    normalizer = Normalizer(config).fit(train_split, feature_cols)
    train_scaled = normalizer.transform(train_split, feature_cols)
    val_scaled = normalizer.transform(val_split, feature_cols)
    test_scaled = normalizer.transform(test_labeled, feature_cols)
    normalizer.save(model_dir / "scaler.joblib")

    generator = SequenceGenerator(config)
    X_train, y_train, train_units, train_cycles = generator.generate_with_metadata(
        train_scaled,
        feature_cols,
    )
    X_val, y_val, val_units, val_cycles = generator.generate_with_metadata(val_scaled, feature_cols)
    X_test, y_test, test_units, test_cycles = generator.generate_with_metadata(test_scaled, feature_cols)
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError(
            f"{dataset_name} produced an empty sequence split; reduce window_size "
            "or adjust the engine-level validation split"
        )

    train_lifetimes = {
        int(unit): float(cycles)
        for unit, cycles in train_split.groupby("unit", sort=False)["cycle"].max().items()
    }
    test_observed_lifetimes = raw_test.groupby("unit", sort=False)["cycle"].max().to_dict()
    test_units_in_file = raw_test["unit"].drop_duplicates().tolist()
    test_lifetimes = {
        int(unit): float(test_observed_lifetimes[unit] + final_rul.iloc[index])
        for index, unit in enumerate(test_units_in_file)
    }
    fractional_life_analysis = run_fractional_life_analysis(
        X_train,
        y_train,
        train_units,
        train_cycles,
        X_test,
        y_test,
        test_units,
        test_cycles,
        train_lifetimes,
        test_lifetimes,
        metrics_dir / "fractional_life",
    )

    baseline = WindowRidgeBaseline().fit(X_train, y_train)
    predictions_val = {"ridge": baseline.predict(X_val)}
    predictions_test = {"ridge": baseline.predict(X_test)}
    model_paths = {"ridge": model_dir / "ridge_baseline.joblib"}
    baseline.save(model_paths["ridge"])
    history_data = None

    if train_lstm:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(config.random_seed)
        lstm_model = __import__("src.lstm_model", fromlist=["LSTMModel"]).LSTMModel(
            config,
            input_shape=(config.window_size, len(feature_cols)),
        )
        history = Trainer(config).train(lstm_model, X_train, y_train, X_val, y_val)
        predictions_val["lstm"] = _keras_predict(lstm_model.model, X_val, config.batch_size)
        predictions_test["lstm"] = _keras_predict(lstm_model.model, X_test, config.batch_size)
        model_paths["lstm"] = model_dir / "lstm_model.keras"
        lstm_model.save(str(model_paths["lstm"]))
        history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
        _save_json(metrics_dir / "training_history.json", history_data)
        Visualizer(config).plot_training_history(history, figures_dir / "training_history.png")
        del tf

    validation_metrics = {
        name: Evaluator.compute_metrics(y_val, values)
        for name, values in predictions_val.items()
    }
    test_metrics = {
        name: Evaluator.compute_metrics(y_test, values)
        for name, values in predictions_test.items()
    }
    selected_model = min(validation_metrics, key=lambda name: validation_metrics[name]["mae"])
    intervals = SplitConformalRUL(nominal_coverage=0.95).fit(y_val, predictions_val[selected_model])
    lower, upper = intervals.predict_interval(
        predictions_test[selected_model],
        max_rul=config.rul_clip_value if config.clip_rul else None,
    )
    uncertainty_metrics = intervals.metrics(y_test, lower, upper)

    selected_predictions = predictions_test[selected_model]
    prediction_frame = pd.DataFrame(
        {
            "unit": test_units,
            "cycle": test_cycles,
            "rul_true": y_test,
            "rul_predicted": selected_predictions,
            "interval_lower": lower,
            "interval_upper": upper,
        }
    )
    for name, values in predictions_test.items():
        prediction_frame[f"rul_{name}"] = values
    prediction_frame.to_csv(predictions_dir / "test_predictions.csv", index=False)

    per_engine = Evaluator.per_engine_metrics(y_test, selected_predictions, test_units)
    per_engine.insert(0, "dataset", dataset_name)
    per_engine.insert(1, "selected_model", selected_model)
    per_engine.to_csv(metrics_dir / "per_engine_metrics.csv", index=False)
    robustness_summary = summarize_engine_robustness(per_engine, metric="mae")
    _save_json(metrics_dir / "engine_robustness.json", robustness_summary)

    metric_rows = []
    for name in predictions_test:
        metric_rows.append({"dataset": dataset_name, "model": name, **test_metrics[name]})
    pd.DataFrame(metric_rows).to_csv(metrics_dir / "model_metrics.csv", index=False)
    sensor_screening = run_sensor_screening(
        train_split,
        sensor_cols,
        metrics_dir / "sensor_screening",
    )
    _save_json(metrics_dir / "uncertainty.json", uncertainty_metrics)
    _save_json(model_dir / "feature_columns.json", {"features": feature_cols, "removed_sensors": removed_sensors})

    visualizer = Visualizer(config)
    visualizer.plot_rul_distribution(train_split, figures_dir / "rul_distribution.png")
    if sensor_cols:
        visualizer.plot_sensor_correlation(train_split, sensor_cols, figures_dir / "sensor_correlation.png")
        visualizer.plot_sensor_trajectories(
            train_split,
            int(train_split["unit"].iloc[0]),
            sensor_cols[:5],
            figures_dir / "sensor_trajectories.png",
        )
    visualizer.plot_predictions_vs_actual(y_test, selected_predictions, figures_dir / "predicted_vs_actual.png")
    visualizer.plot_error_distribution(y_test, selected_predictions, figures_dir / "residual_distribution.png")
    visualizer.plot_model_comparison(test_metrics, figures_dir / "model_comparison.png")
    visualizer.plot_prediction_intervals(
        y_test,
        selected_predictions,
        lower,
        upper,
        figures_dir / "prediction_intervals.png",
    )
    first_unit = int(test_units[0])
    mask = test_units == first_unit
    visualizer.plot_engine_trajectory(
        y_test[mask],
        selected_predictions[mask],
        first_unit,
        figures_dir / f"engine_{first_unit}_trajectory.png",
    )

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": dataset_name,
        "seed": config.random_seed,
        "selected_model": selected_model,
        "dataset_summary": {
            "train_rows": int(len(raw_train)),
            "test_rows": int(len(raw_test)),
            "train_engines": int(train_split["unit"].nunique()),
            "validation_engines": int(val_split["unit"].nunique()),
            "test_engines": int(test_labeled["unit"].nunique()),
            "retained_sensors": int(len(sensor_cols)),
            "removed_sensors": removed_sensors,
            "window_size": config.window_size,
            "rul_cap": config.rul_clip_value if config.clip_rul else None,
        },
        "feature_columns": feature_cols,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "uncertainty": uncertainty_metrics,
        "fractional_life_analysis": fractional_life_analysis,
        "sensor_screening": sensor_screening,
        "engine_robustness": robustness_summary,
        "model_paths": {name: str(path) for name, path in model_paths.items()},
        "figures": {
            path.stem: str(path.relative_to(config.output_dir))
            for path in figures_dir.glob("*.png")
        },
        "package_versions": _package_versions(),
        "research_note": (
            "Methods are adaptations to NASA C-MAPSS RUL labels; the supplied papers use "
            "proprietary in-service engine-health data and TGT targets."
        ),
    }
    _save_json(metrics_dir / "results.json", result)
    _write_report(dataset_output / "experiment_report.md", result)
    logger.info(
        "%s complete: selected %s on validation MAE %.2f; test MAE %.2f",
        dataset_name,
        selected_model,
        validation_metrics[selected_model]["mae"],
        test_metrics[selected_model]["mae"],
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate a C-MAPSS RUL model")
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="FD001")
    parser.add_argument("--baseline-only", action="store_true", help="Run the Ridge benchmark without TensorFlow")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    config = ProjectConfig(dataset=args.dataset, max_epochs=args.epochs)
    run_pipeline(args.dataset, config, train_lstm=not args.baseline_only)
