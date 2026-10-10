import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class EDA:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def plot_sensor_trajectories(self, df: pd.DataFrame, unit: int, sensors: List[str], save_path: str = None):
        """
        Plots the trajectories of specified sensors for a given engine unit.
        """
        unit_df = df[df['unit'] == unit]
        plt.figure(figsize=(12, 8))

        for sensor in sensors:
            plt.plot(unit_df['cycle'], unit_df[sensor], label=sensor)

        plt.title(f"Sensor Trajectories for Engine Unit {unit}")
        plt.xlabel("Cycle")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved sensor trajectory plot to {save_path}")
        plt.close()

    def plot_rul_distribution(self, df: pd.DataFrame, save_path: str = None):
        """
        Plots the distribution of RUL values.
        """
        if 'RUL' not in df.columns:
            logger.error("RUL column missing for distribution plot")
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(df['RUL'], kde=True, bins=30)
        plt.title("Distribution of Remaining Useful Life (RUL)")
        plt.xlabel("RUL (Cycles)")
        plt.ylabel("Frequency")

        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved RUL distribution plot to {save_path}")
        plt.close()
