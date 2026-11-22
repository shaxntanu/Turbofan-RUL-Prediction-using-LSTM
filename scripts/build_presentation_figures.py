from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ProjectConfig
from src.evaluator import Evaluator
from src.visualizer import Visualizer


ROOT = Path(__file__).resolve().parents[1]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
PRESENTATION_FIGURES = ROOT / "web" / "public" / "presentation" / "figures"


def _load_and_validate_predictions(snapshot: dict) -> dict[str, pd.DataFrame]:
    frames = {}
    for subset in SUBSETS:
        output = ROOT / "outputs" / subset
        prediction_path = output / "predictions" / "test_predictions.csv"
        result_path = output / "metrics" / "results.json"
        if not prediction_path.is_file() or not result_path.is_file():
            raise FileNotFoundError(
                f"Missing {subset} pipeline outputs. Run the documented analysis first, "
                f"including {prediction_path.relative_to(ROOT)}."
            )

        frame = pd.read_csv(prediction_path)
        required = {"rul_true", "rul_predicted"}
        if not required.issubset(frame.columns) or frame.empty:
            raise ValueError(f"{prediction_path} must contain non-empty RUL truth and prediction columns")
        if frame[["rul_true", "rul_predicted"]].isna().any().any():
            raise ValueError(f"{prediction_path} contains missing RUL values")

        local_result = json.loads(result_path.read_text(encoding="utf-8"))
        committed_result = snapshot["datasets"][subset]
        if local_result.get("selected_model") != committed_result["selected_model"]:
            raise ValueError(f"{subset} selected model does not match the committed benchmark snapshot")

        metrics = Evaluator.compute_metrics(frame["rul_true"], frame["rul_predicted"])
        expected = committed_result["test_metrics"]
        for name in ("mae", "rmse"):
            if abs(metrics[name] - expected[name]) > 0.02:
                raise ValueError(
                    f"{subset} {name.upper()} from local predictions does not match the committed snapshot: "
                    f"{metrics[name]:.4f} versus {expected[name]:.4f}"
                )
        frames[subset] = frame
    return frames


def build_figures() -> list[Path]:
    snapshot_path = ROOT / "web" / "public" / "results" / "benchmark.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    frames = _load_and_validate_predictions(snapshot)
    PRESENTATION_FIGURES.mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer(ProjectConfig(dataset="FD001"))
    outputs = []
    with plt.rc_context(
        {
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    ):
        fd001 = frames["FD001"]
        scatter_path = PRESENTATION_FIGURES / "fd001-predicted-vs-actual-landscape.png"
        outputs.append(
            visualizer.plot_predictions_vs_actual(
                fd001["rul_true"],
                fd001["rul_predicted"],
                scatter_path,
                figsize=(12, 6.8),
                axis_limits=(0, 130),
                dpi=130,
            )
        )

        history_path = ROOT / "outputs" / "FD001" / "metrics" / "training_history.json"
        if not history_path.is_file():
            raise FileNotFoundError(f"Missing FD001 training history: {history_path.relative_to(ROOT)}")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history_figure = PRESENTATION_FIGURES / "fd001-training-history.png"
        outputs.append(
            visualizer.plot_training_history(history, history_figure, figsize=(11, 4.2), dpi=130)
        )

        error_figure = PRESENTATION_FIGURES / "fd001-residual-distribution.png"
        outputs.append(
            visualizer.plot_error_distribution(
                fd001["rul_true"],
                fd001["rul_predicted"],
                error_figure,
                figsize=(8, 4.8),
                dpi=130,
            )
        )

        train_path = ROOT / "Ref" / "CMAPSSData" / "train_FD001.txt"
        columns = ["unit", "cycle", "setting_1", "setting_2", "setting_3"]
        columns.extend(f"sensor_{index}" for index in range(1, 22))
        train = pd.read_csv(train_path, sep=r"\s+", header=None, names=columns)
        local_result_path = ROOT / "outputs" / "FD001" / "metrics" / "results.json"
        local_result = json.loads(local_result_path.read_text(encoding="utf-8"))
        sensor_columns = [
            column
            for column in local_result["feature_columns"]
            if column.startswith("sensor_")
        ]
        correlation_figure = PRESENTATION_FIGURES / "fd001-sensor-correlation.png"
        outputs.append(
            visualizer.plot_sensor_correlation(
                train,
                sensor_columns,
                correlation_figure,
                figsize=(8, 7),
                dpi=130,
            )
        )
    return outputs


if __name__ == "__main__":
    for generated in build_figures():
        print(f"wrote {generated.relative_to(ROOT)}")
