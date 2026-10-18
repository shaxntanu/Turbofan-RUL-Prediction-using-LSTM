from __future__ import annotations

from contextlib import nullcontext

from src.config import DATASET_NAMES
from src import streamlit_dashboard
from src.streamlit_dashboard import (
    _combined_model_rows,
    _combined_subset_rows,
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
