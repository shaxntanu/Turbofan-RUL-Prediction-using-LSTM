import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def plot_training_history(self, history, save_path: str):
        """Plots training and validation loss/mae."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE')
        ax1.legend()

        # MAE
        ax2.plot(history.history['mae'], label='Train MAE')
        ax2.plot(history.history['val_mae'], label='Val MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()

        plt.savefig(save_path)
        plt.close()
        logger.info(f"Training history saved to {save_path}")

    def plot_predictions_vs_actual(self, y_true, y_pred, save_path: str):
        """Scatter plot of predicted vs actual RUL."""
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.title('Predicted vs Actual RUL')
        plt.xlabel('Actual RUL')
        plt.ylabel('Predicted RUL')
        plt.grid(True)

        plt.savefig(save_path)
        plt.close()
        logger.info(f"Predictions plot saved to {save_path}")

    def plot_engine_trajectory(self, true_rul, pred_rul, unit: int, save_path: str):
        """Plots the RUL prediction trajectory for a single engine."""
        plt.figure(figsize=(10, 6))
        plt.plot(true_rul, label='Actual RUL', color='blue')
        plt.plot(pred_rul, label='Predicted RUL', color='red', linestyle='--')
        plt.title(f"RUL Prediction Trajectory - Engine Unit {unit}")
        plt.xlabel("Window Index")
        plt.ylabel("RUL")
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path)
        plt.close()
        logger.info(f"Trajectory plot saved to {save_path}")
