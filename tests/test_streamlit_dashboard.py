from __future__ import annotations

from contextlib import nullcontext

from src.config import DATASET_NAMES
from src import streamlit_dashboard
from src.streamlit_dashboard import (
    _aggregate_maintenance_scenario,
    _combined_model_rows,
    _combined_subset_rows,
    _maintenance_capacity_curve,
    _read_maintenance_scenarios,
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
