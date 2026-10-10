import pandas as pd
import numpy as np
import os
import logging
from src.config import ProjectConfig

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.columns = ['unit', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + \
                       [f'sensor_{i}' for i in range(1, 22)]

    def load_train_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.config.raw_data_dir, f"train_{self.config.dataset}.txt")
        logger.info(f"Loading training data from {file_path}")
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=self.columns)
            return df
        except FileNotFoundError:
            logger.error(f"Training file not found: {file_path}")
            raise

    def load_test_data(self) -> pd.DataFrame:
        file_path = os.path.join(self.config.raw_data_dir, f"test_{self.config.dataset}.txt")
        logger.info(f"Loading test data from {file_path}")
        try:
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=self.columns)
            return df
        except FileNotFoundError:
            logger.error(f"Test file not found: {file_path}")
            raise

    def load_rul_data(self) -> pd.Series:
        file_path = os.path.join(self.config.raw_data_dir, f"RUL_{self.config.dataset}.txt")
        logger.info(f"Loading RUL ground truth from {file_path}")
        try:
            # RUL files are usually a single column of values
            rul = pd.read_csv(file_path, sep='\s+', header=None).iloc[:, 0]
            return rul
        except FileNotFoundError:
            logger.error(f"RUL file not found: {file_path}")
            raise
