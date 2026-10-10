import pandas as pd
import numpy as np
import logging
import joblib
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Configuration
    config = ProjectConfig()
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

    normalizer.save(f"{config.model_dir}/scaler.pkl")

    # 8. Sequence Generation
    seq_gen = SequenceGenerator(config)
    X_train, y_train = seq_gen.generate(train_split_scaled, feature_cols)
    X_val, y_val = seq_gen.generate(val_split_scaled, feature_cols)
    X_test, y_test = seq_gen.generate(test_scaled, feature_cols)

    logger.info(f"Sequences generated. X_train shape: {X_train.shape}, X_val shape: {X_val.shape}, X_test shape: {X_test.shape}")

    # 9. Model Training
    model_wrapper = LSTMModel(config, input_shape=(config.window_size, len(feature_cols)))
    trainer = Trainer(config)
    history = trainer.train(model_wrapper, X_train, y_train, X_val, y_val)

    # 10. Evaluation
    evaluator = Evaluator()
    y_pred = model_wrapper.model.predict(X_test)
    metrics = evaluator.compute_metrics(y_test, y_pred)

    logger.info(f"Test Metrics: {metrics}")

    # Save Final Model
    model_wrapper.save(f"{config.model_dir}/final_lstm_model.h5")

if __name__ == "__main__":
    main()
