from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ProjectConfig

PALETTE = {"teal": "#138a8a", "blue": "#4169a1", "coral": "#dd6b55", "gold": "#d4a12a"}


class Visualizer:
    def __init__(self, config: ProjectConfig):
        self.config = config
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)

    @staticmethod
    def _save(fig, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path

    def plot_training_history(self, history, save_path: str | Path) -> Path:
        values = history.history if hasattr(history, "history") else history
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for key, label, color in (("loss", "Train", PALETTE["blue"]), ("val_loss", "Validation", PALETTE["coral"])):
            if key in values:
                axes[0].plot(values[key], label=label, color=color, linewidth=2)
        for key, label, color in (("mae", "Train", PALETTE["blue"]), ("val_mae", "Validation", PALETTE["coral"])):
            if key in values:
                axes[1].plot(values[key], label=label, color=color, linewidth=2)
        axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Mean squared error")
        axes[1].set(title="Mean absolute error", xlabel="Epoch", ylabel="RUL error (cycles)")
        for axis in axes:
            axis.legend(frameon=False)
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_predictions_vs_actual(self, y_true, y_pred, save_path: str | Path) -> Path:
        truth = np.asarray(y_true, dtype=float).ravel()
        pred = np.asarray(y_pred, dtype=float).ravel()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(truth, pred, s=13, alpha=0.35, color=PALETTE["teal"], edgecolors="none")
        low = float(min(truth.min(), pred.min()))
        high = float(max(truth.max(), pred.max()))
        ax.plot([low, high], [low, high], linestyle="--", color=PALETTE["coral"], linewidth=1.8)
        ax.set(title="Predicted versus observed RUL", xlabel="Observed RUL (cycles)", ylabel="Predicted RUL (cycles)")
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_error_distribution(self, y_true, y_pred, save_path: str | Path) -> Path:
        errors = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.histplot(errors, bins=40, kde=True, color=PALETTE["blue"], ax=ax)
        ax.axvline(0, color=PALETTE["coral"], linestyle="--", linewidth=1.7)
        ax.axvline(errors.mean(), color=PALETTE["gold"], linewidth=1.7, label=f"Mean = {errors.mean():.1f}")
        ax.set(title="RUL residual distribution", xlabel="Observed - predicted RUL (cycles)", ylabel="Window count")
        ax.legend(frameon=False)
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_engine_trajectory(self, true_rul, pred_rul, unit: int, save_path: str | Path) -> Path:
        truth = np.asarray(true_rul, dtype=float).ravel()
        pred = np.asarray(pred_rul, dtype=float).ravel()
        fig, ax = plt.subplots(figsize=(8, 4.3))
        ax.plot(truth, label="Observed", color=PALETTE["blue"], linewidth=2)
        ax.plot(pred, label="Predicted", color=PALETTE["coral"], linewidth=2)
        ax.set(title=f"Engine {unit}: remaining useful life", xlabel="Window endpoint", ylabel="RUL (cycles)")
        ax.legend(frameon=False)
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_sensor_trajectories(
        self,
        frame: pd.DataFrame,
        unit: int,
        sensors: list[str],
        save_path: str | Path,
    ) -> Path:
        data = frame.loc[frame["unit"] == unit].sort_values("cycle")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for sensor in sensors:
            if sensor in data:
                ax.plot(data["cycle"], data[sensor], label=sensor, linewidth=1.5)
        ax.set(title=f"Engine {unit}: sensor history", xlabel="Cycle", ylabel="Sensor reading")
        ax.legend(ncol=2, frameon=False)
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_rul_distribution(self, frame: pd.DataFrame, save_path: str | Path) -> Path:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        sns.histplot(frame["RUL"], bins=45, color=PALETTE["teal"], ax=ax)
        ax.set(title=f"{self.config.dataset}: RUL target distribution", xlabel="RUL (cycles)", ylabel="Cycle count")
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_sensor_correlation(self, frame: pd.DataFrame, columns: list[str], save_path: str | Path) -> Path:
        corr = frame[columns].corr()
        size = max(7, min(12, len(columns) * 0.42))
        fig, ax = plt.subplots(figsize=(size, size - 1))
        sns.heatmap(corr, cmap="vlag", center=0, square=True, ax=ax, cbar_kws={"shrink": 0.72})
        ax.set(title=f"{self.config.dataset}: sensor correlation")
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_model_comparison(self, metrics: dict[str, dict], save_path: str | Path) -> Path:
        rows = [{"Model": name, "MAE": values["mae"], "RMSE": values["rmse"]} for name, values in metrics.items()]
        frame = pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric", value_name="Cycles")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        palette = {"ridge": PALETTE["teal"], "lstm": PALETTE["coral"]}
        sns.barplot(data=frame, x="Metric", y="Cycles", hue="Model", palette=palette, ax=ax)
        ax.set(title=f"{self.config.dataset}: model comparison", ylabel="Error (cycles)")
        ax.legend(title=None, frameon=False)
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_fractional_life_metrics(self, results: pd.DataFrame, save_path: str | Path) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        palette = {
            "shared_full_life": PALETTE["blue"],
            "phase_specialized": PALETTE["coral"],
        }
        sns.barplot(
            data=results,
            x="life_stage",
            y="mae",
            hue="model_scope",
            order=["early", "middle", "late"],
            hue_order=["shared_full_life", "phase_specialized"],
            palette=palette,
            ax=ax,
        )
        ax.set(
            title=f"{self.config.dataset}: RUL error by fractional life stage",
            xlabel="Estimated engine life stage",
            ylabel="Mean absolute error (cycles)",
        )
        ax.legend(title=None, frameon=False)
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_transfer_matrix(self, results: pd.DataFrame, save_path: str | Path, metric: str = "mae") -> Path:
        matrix = results.pivot(index="source_dataset", columns="target_dataset", values=metric)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="crest", linewidths=0.6, ax=ax, cbar_kws={"label": f"{metric.upper()} (cycles)"})
        ax.set(title=f"Cross-subset transfer: {metric.upper()}", xlabel="Evaluation subset", ylabel="Training subset")
        fig.tight_layout()
        return self._save(fig, save_path)

    def plot_prediction_intervals(
        self,
        y_true,
        y_pred,
        lower,
        upper,
        save_path: str | Path,
        max_points: int = 450,
    ) -> Path:
        arrays = [np.asarray(values, dtype=float).ravel() for values in (y_true, y_pred, lower, upper)]
        order = np.argsort(arrays[0])
        if len(order) > max_points:
            order = order[np.linspace(0, len(order) - 1, max_points, dtype=int)]
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.fill_between(x, arrays[2][order], arrays[3][order], color=PALETTE["teal"], alpha=0.22, label="95% conformal interval")
        ax.plot(x, arrays[0][order], color=PALETTE["blue"], linewidth=1.5, label="Observed")
        ax.plot(x, arrays[1][order], color=PALETTE["coral"], linewidth=1.4, label="Predicted")
        ax.set(title=f"{self.config.dataset}: calibrated RUL intervals", xlabel="Test windows sorted by observed RUL", ylabel="RUL (cycles)")
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        return self._save(fig, save_path)
