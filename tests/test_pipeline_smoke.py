from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import pipeline
from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.data_splitter import DataSplitter


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "Ref" / "CMAPSSData"


def _first_engine_cycles(frame: pd.DataFrame, engine_count: int, cycle_count: int) -> pd.DataFrame:
    selected = frame.loc[frame["unit"] <= engine_count]
    return selected.groupby("unit", group_keys=False, sort=False).head(cycle_count).copy()


def test_ridge_pipeline_smoke_writes_artifacts_without_engine_leakage(tmp_path, monkeypatch):
    observed_split = {}

    class BoundedDataLoader(DataLoader):
        def load_train_data(self):
            return _first_engine_cycles(super().load_train_data(), engine_count=8, cycle_count=60)

        def load_test_data(self):
            return _first_engine_cycles(super().load_test_data(), engine_count=4, cycle_count=45)

        def load_rul_data(self):
            return super().load_rul_data().iloc[:4].copy()

    class RecordingDataSplitter(DataSplitter):
        def split_engines(self, frame):
            train, validation = super().split_engines(frame)
            train_units = set(train["unit"].unique())
            validation_units = set(validation["unit"].unique())
            assert train_units.isdisjoint(validation_units)
            observed_split["train"] = train_units
            observed_split["validation"] = validation_units
            return train, validation

    monkeypatch.setattr(pipeline, "DataLoader", BoundedDataLoader)
    monkeypatch.setattr(pipeline, "DataSplitter", RecordingDataSplitter)
    config = ProjectConfig(
        dataset="FD001",
        raw_data_dir=DATA_DIRECTORY,
        output_dir=tmp_path / "outputs",
        model_dir=tmp_path / "models",
        train_ratio=0.75,
    )

    result = pipeline.run_pipeline("FD001", config, train_lstm=False)

    assert observed_split["train"]
    assert observed_split["validation"]
    assert observed_split["train"].isdisjoint(observed_split["validation"])
    assert result["selected_model"] == "ridge"
    assert result["dataset_summary"]["train_engines"] == len(observed_split["train"])
    assert result["dataset_summary"]["validation_engines"] == len(observed_split["validation"])

    dataset_output = config.output_dir / "FD001"
    model_output = config.model_dir / "FD001"
    result_path = dataset_output / "metrics" / "results.json"
    predictions_path = dataset_output / "predictions" / "test_predictions.csv"
    predictions = pd.read_csv(predictions_path)

    assert json.loads(result_path.read_text(encoding="utf-8"))["selected_model"] == "ridge"
    assert (model_output / "ridge_baseline.joblib").is_file()
    assert (model_output / "scaler.joblib").is_file()
    assert (model_output / "feature_columns.json").is_file()
    assert (dataset_output / "experiment_report.md").is_file()
    assert (dataset_output / "figures" / "prediction_intervals.png").is_file()
    assert set(predictions["unit"].unique()) == {1, 2, 3, 4}
    assert predictions.groupby("unit")["cycle"].min().ge(config.window_size).all()
    assert predictions[["rul_true", "rul_predicted", "interval_lower", "interval_upper"]].notna().all().all()
