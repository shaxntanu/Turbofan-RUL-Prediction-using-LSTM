from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_model import WindowRidgeBaseline
from src.config import DATASET_NAMES, ProjectConfig
from src.data_loader import DataLoader
from src.data_splitter import DataSplitter
from src.data_validator import DataValidator
from src.evaluator import Evaluator
from src.normalization import Normalizer
from src.sequence_generator import SequenceGenerator
from src.target_construction import TargetConstructor
from src.trainer import Trainer
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)


def _load_domains(
    datasets: tuple[str, ...], config: ProjectConfig
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    training: dict[str, pd.DataFrame] = {}
    testing: dict[str, pd.DataFrame] = {}
    feature_columns: list[str] | None = None

    for dataset in datasets:
        domain_config = replace(config, dataset=dataset)
        loader = DataLoader(domain_config)
        raw_train = loader.load_train_data()
        raw_test = loader.load_test_data()
        for split, frame in (("train", raw_train), ("test", raw_test)):
            valid, issues = DataValidator.validate(frame, dataset)
            if not valid:
                raise ValueError(f"Invalid {dataset} {split} data: {'; '.join(issues)}")

        final_rul = loader.load_rul_data()
        if raw_test["unit"].nunique() != len(final_rul):
            raise ValueError(f"{dataset} test trajectories and RUL labels do not align")
        training[dataset] = TargetConstructor.compute_training_rul(
            raw_train,
            config.rul_clip_value if config.clip_rul else None,
        )
        testing[dataset] = TargetConstructor.compute_test_rul(
            raw_test,
            final_rul,
            config.rul_clip_value if config.clip_rul else None,
        )
        current_features = [
            column for column in training[dataset]
            if column not in {"unit", "cycle", "RUL"}
        ]
        if feature_columns is None:
            feature_columns = current_features
        elif current_features != feature_columns:
            raise ValueError(f"{dataset} does not have the common C-MAPSS feature schema")

    return training, testing, feature_columns or []


def _pool_engine_ids(frames: list[pd.DataFrame]) -> pd.DataFrame:
    pooled = []
    offset = 0
    for frame in frames:
        item = frame.copy()
        item["unit"] = item["unit"].astype(int) + offset
        pooled.append(item)
        offset += int(frame["unit"].max()) + 1
    return pd.concat(pooled, ignore_index=True)


def _fit_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    config: ProjectConfig,
    *,
    train_lstm: bool,
) -> dict[str, dict]:
    train_frame, validation_frame = DataSplitter(config).split_engines(frame)
    normalizer = Normalizer(config).fit(train_frame, feature_columns)
    train_scaled = normalizer.transform(train_frame, feature_columns)
    validation_scaled = normalizer.transform(validation_frame, feature_columns)
    generator = SequenceGenerator(config)
    X_train, y_train = generator.generate(train_scaled, feature_columns)
    X_validation, y_validation = generator.generate(validation_scaled, feature_columns)
    if not len(X_train) or not len(X_validation):
        raise ValueError("Transfer split produced an empty sequence set; reduce window_size")

    fitted: dict[str, dict] = {
        "ridge": {
            "model": WindowRidgeBaseline().fit(X_train, y_train),
            "normalizer": normalizer,
            "feature_columns": feature_columns,
        }
    }
    if train_lstm:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(config.random_seed)
        lstm_config = replace(config, lstm_units=[10], dense_units=10)
        from src.lstm_model import LSTMModel

        model = LSTMModel(lstm_config, input_shape=(config.window_size, len(feature_columns)))
        Trainer(lstm_config).train(
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
        fitted["lstm_10"] = {
            "model": model,
            "normalizer": normalizer,
            "feature_columns": feature_columns,
        }
    return fitted


def _predict(bundle: dict, frame: pd.DataFrame, config: ProjectConfig) -> tuple[np.ndarray, np.ndarray]:
    scaled = bundle["normalizer"].transform(frame, bundle["feature_columns"])
    sequences, targets, units, _ = SequenceGenerator(config).generate_with_metadata(
        scaled,
        bundle["feature_columns"],
    )
    model = bundle["model"]
    if isinstance(model, WindowRidgeBaseline):
        predictions = model.predict(sequences)
    else:
        predictions = np.asarray(
            model.model.predict(sequences, batch_size=config.batch_size, verbose=0),
            dtype=np.float32,
        ).ravel()
    upper = config.rul_clip_value if config.clip_rul else None
    predictions = np.clip(predictions, 0.0, upper) if upper is not None else np.maximum(predictions, 0.0)
    return predictions, targets


def _save_model(bundle: dict, path: Path) -> str:
    model = bundle["model"]
    if isinstance(model, WindowRidgeBaseline):
        model.save(path.with_suffix(".joblib"))
        return str(path.with_suffix(".joblib"))
    model_path = path.with_suffix(".keras")
    model.save(str(model_path))
    return str(model_path)


def run_transfer_analysis(
    datasets: tuple[str, ...] = DATASET_NAMES,
    config: ProjectConfig | None = None,
    *,
    train_lstm: bool = False,
) -> dict:
    """Evaluate source-specific and pooled models across C-MAPSS subsets.

    FD001-FD004 are related simulated subsets, not independent real-world fleets.
    Scaling is fitted only on the source training engines for each model.
    """
    datasets = tuple(dict.fromkeys(name.upper() for name in datasets))
    if len(datasets) < 2 or any(name not in DATASET_NAMES for name in datasets):
        raise ValueError("Select at least two valid C-MAPSS subsets")
    config = config or ProjectConfig(dataset=datasets[0])
    training, testing, feature_columns = _load_domains(datasets, config)
    pooled_training = _pool_engine_ids([training[name] for name in datasets])

    model_sources: dict[str, dict[str, dict]] = {}
    for source in datasets:
        source_config = replace(config, dataset=source)
        model_sources[source] = _fit_models(
            training[source],
            feature_columns,
            source_config,
            train_lstm=train_lstm,
        )
    pooled_config = replace(config, dataset=datasets[0])
    model_sources["pooled"] = _fit_models(
        pooled_training,
        feature_columns,
        pooled_config,
        train_lstm=train_lstm,
    )

    output_dir = config.output_dir / "transfer"
    model_dir = config.model_dir / "transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    saved_models = {}
    for source, bundles in model_sources.items():
        saved_models[source] = {}
        for model_name, bundle in bundles.items():
            saved_models[source][model_name] = _save_model(
                bundle,
                model_dir / f"{source}_{model_name}",
            )
            for target in datasets:
                predictions, truth = _predict(bundle, testing[target], config)
                metrics = Evaluator.compute_metrics(truth, predictions)
                rows.append(
                    {
                        "source_dataset": source,
                        "target_dataset": target,
                        "model": model_name,
                        **metrics,
                    }
                )
                logger.info(
                    "%s -> %s (%s): MAE %.2f cycles",
                    source,
                    target,
                    model_name,
                    metrics["mae"],
                )

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "transfer_metrics.csv", index=False)
    Visualizer(config).plot_transfer_matrix(
        results.loc[results["model"] == "ridge"],
        output_dir / "transfer_mae_ridge.png",
        metric="mae",
    )
    if train_lstm:
        Visualizer(config).plot_transfer_matrix(
            results.loc[results["model"] == "lstm_10"],
            output_dir / "transfer_mae_lstm10.png",
            metric="mae",
        )

    result = {
        "schema_version": 1,
        "run_id": time.strftime("%Y%m%d-%H%M%S"),
        "datasets": list(datasets),
        "models": sorted(results["model"].unique().tolist()),
        "feature_columns": feature_columns,
        "validation_protocol": "engine-level holdout within source training trajectories",
        "evaluation_protocol": "official test trajectories and RUL labels for each target subset",
        "scaling_protocol": "fit on source training engines only; no target-domain refit",
        "domain_caveat": (
            "FD001-FD004 are simulated C-MAPSS operating-condition and fault-mode subsets, "
            "not distinct real-world engine fleets. Transfer scores are a proxy-domain study."
        ),
        "lstm_architecture": "single 10-unit LSTM with 10-unit dense layer" if train_lstm else None,
        "metrics_path": str(output_dir / "transfer_metrics.csv"),
        "figures": {
            "ridge_mae": str(output_dir / "transfer_mae_ridge.png"),
            **({"lstm10_mae": str(output_dir / "transfer_mae_lstm10.png")} if train_lstm else {}),
        },
        "model_paths": saved_models,
    }
    (output_dir / "transfer_results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run cross-subset C-MAPSS transfer analysis")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--with-lstm", action="store_true", help="Also train the 10-unit LSTM variant")
    args = parser.parse_args()
    config = ProjectConfig(dataset=args.datasets[0], max_epochs=args.epochs)
    run_transfer_analysis(tuple(args.datasets), config, train_lstm=args.with_lstm)
