import pandas as pd
import numpy as np
import logging
import joblib
import os
from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.target_construction import TargetConstructor
from src.data_cleaner import DataCleaner
from src.normalization import Normalizer
from src.sequence_generator import SequenceGenerator
from src.data_splitter import DataSplitter
from src.lstm_model import LSTMModel
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(dataset_name="FD001"):
    # 1. Configuration
    config = ProjectConfig()
    config.dataset = dataset_name
    logger.info(f"Starting RUL Prediction Pipeline for {config.dataset}")

    # 2. Data Ingestion
    loader = DataLoader(config)
    train_df = loader.load_train_data()
    test_df = loader.load_test_data()
    final_ruls = loader.load_rul_data()

    # 3. Validation
    validator = DataValidator()
    is_valid, issues = validator.validate(train_df, config.dataset)
    if not is_valid:
        logger.warning(f"Training data validation issues: {issues}")

    # 4. Target Construction
    constructor = TargetConstructor()
    train_df = constructor.compute_training_rul(train_df)
    test_df = constructor.compute_test_rul(test_df, final_ruls)

    # 5. Cleaning
    cleaner = DataCleaner(config)
    train_df, constant_sensors = cleaner.remove_constant_sensors(train_df)
    test_df = test_df.drop(columns=constant_sensors)

    # 6. Splitting (at engine level)
    splitter = DataSplitter(config)
    train_split_df, val_split_df = splitter.split_engines(train_df)

    # 7. Normalization
    feature_cols = [col for col in train_df.columns if col not in ['unit', 'cycle', 'RUL']]
    normalizer = Normalizer(config)
    normalizer.fit(train_split_df, feature_cols)

    train_split_scaled = normalizer.transform(train_split_df, feature_cols)
    val_split_scaled = normalizer.transform(val_split_df, feature_cols)
    test_scaled = normalizer.transform(test_df, feature_cols)

    # Save dataset-specific scaler
    scaler_path = f"{config.model_dir}/scaler_{dataset_name}.pkl"
    normalizer.save(scaler_path)

    # 8. Sequence Generation
    seq_gen = SequenceGenerator(config)
    X_train, y_train = seq_gen.generate(train_split_scaled, feature_cols)
    X_val, y_val = seq_gen.generate(val_split_scaled, feature_cols)
    X_test, y_test = seq_gen.generate(test_scaled, feature_cols)

    logger.info(f"Sequences generated for {dataset_name}. X_train: {X_train.shape}")

    # 9. Model Training
    model_wrapper = LSTMModel(config, input_shape=(config.window_size, len(feature_cols)))
    trainer = Trainer(config)
    history = trainer.train(model_wrapper, X_train, y_train, X_val, y_val)

    # 10. Evaluation
    evaluator = Evaluator()
    y_pred = model_wrapper.model.predict(X_test)
    metrics = evaluator.compute_metrics(y_test, y_pred)

    logger.info(f"Dataset {dataset_name} Test Metrics: {metrics}")

    # 11. Visualization
    viz = Visualizer(config)
    dataset_fig_dir = f"{config.output_dir}/figures/{dataset_name}"
    os.makedirs(dataset_fig_dir, exist_ok=True)

    viz.plot_training_history(history, f"{dataset_fig_dir}/loss_curves.png")
    viz.plot_predictions_vs_actual(y_test, y_pred, f"{dataset_fig_dir}/predicted_vs_actual.png")

    # Save Final Model for this dataset
    model_path = f"{config.model_dir}/final_lstm_model_{dataset_name}.h5"
    model_wrapper.save(model_path)

    return metrics

if __name__ == "__main__":
    # Run for the baseline dataset by default
    run_pipeline("FD001")
