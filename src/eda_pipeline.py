import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.target_construction import TargetConstructor
from src.data_cleaner import DataCleaner
from src.eda import EDA

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_eda_pipeline():
    config = ProjectConfig()
    loader = DataLoader(config)
    validator = DataValidator()
    constructor = TargetConstructor()
    cleaner = DataCleaner(config)
    eda_tool = EDA(config)

    # 1. Load and Validate
    train_df = loader.load_train_data()
    test_df = loader.load_test_data()
    final_ruls = loader.load_rul_data()

    is_valid, issues = validator.validate(train_df, config.dataset)
    if not is_valid:
        logger.warning(f"Training data validation issues: {issues}")

    # 2. Compute RUL
    train_df = constructor.compute_training_rul(train_df)
    test_df = constructor.compute_test_rul(test_df, final_ruls)

    # 3. Clean Constant Sensors
    train_df, constant_sensors = cleaner.remove_constant_sensors(train_df)
    test_df = test_df.drop(columns=constant_sensors)
    logger.info(f"Constant sensors removed: {constant_sensors}")

    # 4. Exploratory Analysis
    # Plot a few engine trajectories
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    for unit in [1, 10, 50]:
        save_path = f"outputs/figures/eda/unit_{unit}_trajectories.png"
        eda_tool.plot_sensor_trajectories(train_df, unit, sensor_cols[:5], save_path)

    # Plot RUL Distribution
    eda_tool.plot_rul_distribution(train_df, "outputs/figures/eda/rul_distribution.png")

    # Save processed data for later steps
    train_df.to_csv("data/processed/train_FD001_processed.csv", index=False)
    test_df.to_csv("data/processed/test_FD001_processed.csv", index=False)
    logger.info("EDA pipeline complete. Processed data saved.")

if __name__ == "__main__":
    run_eda_pipeline()
