from __future__ import annotations

import json
from contextlib import nullcontext

import pandas as pd

from src.baseline_model import WindowRidgeBaseline
from src.config import DATASET_NAMES, ProjectConfig
from src import streamlit_dashboard
from src.data_loader import DataLoader
from src.normalization import Normalizer
from src.sequence_generator import SequenceGenerator
from src.streamlit_dashboard import (
    _aggregate_maintenance_scenario,
    _build_uploaded_forecast,
    _combined_model_rows,
    _combined_subset_rows,
    _maintenance_capacity_curve,
    _read_maintenance_scenarios,
    _read_dashboard_bundle,
    _read_upload,
    _reliability_snapshot,
)


def _bundle() -> dict:
    return {
        "datasets": {
            name: {
                "selected_model": "ridge",
                "test_metrics": {"n_samples": 10, "mae": index + 1.0, "rmse": index + 2.0},
                "model_metrics": {"ridge": {"mae": index + 1.0, "rmse": index + 2.0}},
                "validation_metrics": {"ridge": {"mae": index + 0.5}},
                "dataset_summary": {"test_engines": index + 100},
                "uncertainty": {"picp": 0.9},
            }
            for index, name in enumerate(DATASET_NAMES)
        }
    }


def test_all_subset_tables_cover_each_subset_and_candidate():
    bundle = _bundle()

    subset_rows = _combined_subset_rows(bundle)
    model_rows = _combined_model_rows(bundle)

    assert subset_rows["Subset"].tolist() == list(DATASET_NAMES)
    assert subset_rows["MAE (cycles)"].mean() == 2.5
    assert model_rows["Selected on validation"].all()
    assert model_rows["Subset"].tolist() == list(DATASET_NAMES)


def test_method_animation_renders_the_bundled_mp4(monkeypatch):
    rendered_videos = []
    monkeypatch.setattr(streamlit_dashboard.st, "expander", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(streamlit_dashboard.st, "video", rendered_videos.append)
    monkeypatch.setattr(streamlit_dashboard.st, "caption", lambda *args, **kwargs: None)

    streamlit_dashboard._render_method_video()

    assert rendered_videos == [str(streamlit_dashboard.METHOD_VIDEO_PATH)]


def test_reliability_view_falls_back_to_committed_tables_when_outputs_are_absent(tmp_path):
    study = {"run_id": "published-run", "datasets": list(DATASET_NAMES)}
    rows = [{"dataset": "FD001", "model": "ridge_clean", "terminal_mae_mean": 12.5}]
    bundle = {
        "reliability": {
            "status": "available",
            "study": study,
            "tables": {"repeated_seed_summary": rows},
        }
    }

    loaded_study, tables = _reliability_snapshot(tmp_path / "metrics", bundle)

    assert loaded_study == study
    assert tables["repeated_seed_summary"] == rows


def test_dashboard_bundle_merges_subset_reliability_snapshots(tmp_path, monkeypatch):
    core_path = tmp_path / "benchmark.json"
    core_path.write_text(json.dumps({"datasets": {"FD001": {}}}), encoding="utf-8")
    reliability_dir = tmp_path / "reliability"
    reliability_dir.mkdir()
    study = {"run_id": "split-study", "datasets": ["FD001", "FD002"]}
    for dataset in ("FD001", "FD002"):
        (reliability_dir / f"{dataset.lower()}.json").write_text(
            json.dumps(
                {
                    "status": "available",
                    "study": study,
                    "tables": {
                        "repeated_seed_summary": [
                            {"dataset": dataset, "model": "ridge_clean"}
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(streamlit_dashboard, "RESULTS_BUNDLE_PATH", core_path)
    monkeypatch.setattr(streamlit_dashboard, "RELIABILITY_BUNDLE_DIRECTORY", reliability_dir)
    _read_dashboard_bundle.clear()

    try:
        bundle = _read_dashboard_bundle()
    finally:
        _read_dashboard_bundle.clear()

    assert bundle["reliability"]["status"] == "available"
    assert bundle["reliability"]["study"] == study
    assert bundle["reliability"]["tables"]["repeated_seed_summary"] == [
        {"dataset": "FD001", "model": "ridge_clean"},
        {"dataset": "FD002", "model": "ridge_clean"},
    ]


def test_maintenance_scenarios_reconcile_for_each_subset_and_all_subsets():
    scenarios = _read_maintenance_scenarios()

    assert set(scenarios["dataset"]) == set(DATASET_NAMES)
    assert len(scenarios) == 320
    for dataset in (*DATASET_NAMES, "All subsets"):
        result = _aggregate_maintenance_scenario(
            scenarios, dataset, 30, 20, "point_estimate"
        )
        assert result is not None
        assert result["scheduled_engines"] <= result["slot_limit"]
        assert result["urgent_serviced"] + result["missed_urgent"] == result["urgent_engine_count"]
        assert result["urgent_serviced"] + result["unnecessary_service"] == result["scheduled_engines"]

    combined = _aggregate_maintenance_scenario(
        scenarios, "All subsets", 30, 20, "point_estimate"
    )
    assert combined["engine_count"] == 690
    assert combined["slot_limit"] == 137


def test_maintenance_interval_policy_and_capacity_frontier_are_consistent():
    scenarios = _read_maintenance_scenarios()
    point = _aggregate_maintenance_scenario(
        scenarios, "FD001", 30, 20, "point_estimate"
    )
    interval = _aggregate_maintenance_scenario(
        scenarios, "FD001", 30, 20, "conservative_interval"
    )
    assert interval["eligible_engines"] >= point["eligible_engines"]

    curve = _maintenance_capacity_curve(
        scenarios, "FD001", 30, "conservative_interval"
    )
    recall = curve["urgent_recall"].dropna()
    assert recall.is_monotonic_increasing


def test_uploaded_ridge_forecast_parses_labels_and_bounds_prediction_intervals(tmp_path):
    config = ProjectConfig(dataset="FD001", window_size=30, output_dir=tmp_path / "outputs")
    raw_columns = DataLoader(config).columns
    uploaded_frame = pd.DataFrame(
        {
            "unit": [7] * 40,
            "cycle": list(range(1, 41)),
            "setting_1": [0.1] * 40,
            "setting_2": [0.2] * 40,
            "setting_3": [0.3] * 40,
            **{
                f"sensor_{sensor}": [float(cycle * sensor) for cycle in range(1, 41)]
                for sensor in range(1, 22)
            },
            "RUL": [125.0 - cycle for cycle in range(1, 41)],
        }
    )[raw_columns + ["RUL"]]
    csv_bytes = uploaded_frame.to_csv(index=False).encode("utf-8")

    class UploadedFile:
        name = "history.csv"

        def getvalue(self):
            return csv_bytes

    uploaded = _read_upload(UploadedFile())
    feature_columns = [column for column in raw_columns if column not in {"unit", "cycle"}]
    training = uploaded.iloc[:35].copy()
    training["RUL"] = 250.0 - training["cycle"]
    normalizer = Normalizer(config).fit(training, feature_columns)
    scaled = normalizer.transform(training, feature_columns)
    sequences, targets, _, _ = SequenceGenerator(config).generate_with_metadata(
        scaled,
        feature_columns,
        target_col="RUL",
    )
    model = WindowRidgeBaseline().fit(sequences, targets)
    model_dir = tmp_path / "FD001"
    model_dir.mkdir()
    normalizer.save(model_dir / "scaler.joblib")
    model.save(model_dir / "ridge_baseline.joblib")
    (model_dir / "feature_columns.json").write_text(
        json.dumps({"features": feature_columns}),
        encoding="utf-8",
    )

    forecast, observed_metrics = _build_uploaded_forecast(
        uploaded,
        config,
        model_dir,
        {"uncertainty_by_model": {"ridge": {"interval_radius_cycles": 12.0}}},
    )

    assert "rul" in uploaded
    assert forecast is not None
    assert observed_metrics is not None
    assert len(forecast) == 11
    assert forecast["cycle"].tolist() == list(range(30, 41))
    assert forecast["unit"].unique().tolist() == [7]
    assert forecast["rul_observed"].tolist() == [125.0 - cycle for cycle in range(30, 41)]
    assert forecast["rul_predicted"].between(0, config.rul_clip_value).all()
    assert (forecast["interval_lower"] <= forecast["rul_predicted"]).all()
    assert (forecast["rul_predicted"] <= forecast["interval_upper"]).all()
    assert (forecast["interval_upper"] <= config.rul_clip_value).all()
