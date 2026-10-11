from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.columns = [
            "unit",
            "cycle",
            "setting_1",
            "setting_2",
            "setting_3",
            *[f"sensor_{i}" for i in range(1, 22)],
        ]

    def _load_measurements(self, split: str) -> pd.DataFrame:
        path = Path(self.config.raw_data_dir) / f"{split}_{self.config.dataset}.txt"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {split} data for {self.config.dataset}: {path}. "
                "Set CMAPSS_DATA_DIR to the directory containing the C-MAPSS files."
            )

        frame = pd.read_csv(path, sep=r"\s+", header=None, names=self.columns)
        if frame.shape[1] != len(self.columns):
            raise ValueError(f"{path} has {frame.shape[1]} columns; expected 26")
        logger.info("Loaded %s rows from %s", len(frame), path)
        return frame

    def load_train_data(self) -> pd.DataFrame:
        return self._load_measurements("train")

    def load_test_data(self) -> pd.DataFrame:
        return self._load_measurements("test")

    def load_rul_data(self) -> pd.Series:
        path = Path(self.config.raw_data_dir) / f"RUL_{self.config.dataset}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing test RUL labels for {self.config.dataset}: {path}")
        values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
        return values.astype(float)
