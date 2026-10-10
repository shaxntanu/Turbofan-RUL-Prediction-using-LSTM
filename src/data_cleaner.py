import pandas as pd
import numpy as np
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def remove_constant_sensors(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Identifies sensors with variance below threshold and removes them.
        """
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        sensor_vars = df[sensor_cols].var()

        constant_sensors = sensor_vars[sensor_vars < self.config.constant_sensor_threshold].index.tolist()

        if constant_sensors:
            logger.info(f"Removing constant sensors: {constant_sensors}")
            df = df.drop(columns=constant_sensors)

        return df, constant_sensors
