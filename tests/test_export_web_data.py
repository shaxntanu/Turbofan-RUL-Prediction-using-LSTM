import json

import pandas as pd

from scripts import export_web_data


def test_reliability_export_keeps_aggregate_tables_portable(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    manifest = {
        "run_id": "run-1",
        "datasets": ["FD001"],
        "repeated_seeds": [42, 43, 44],
        "outputs": {"terminal_predictions": "C:/private/local/path.csv"},
    }
    (metrics_dir / "reliability_study.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [{"dataset": "FD001", "model": "ridge_clean", "terminal_mae_mean": 12.5}]
    ).to_csv(metrics_dir / "repeated_seed_summary.csv", index=False)
    (metrics_dir / "risk_coverage.csv").write_text("", encoding="utf-8")
    monkeypatch.setattr(export_web_data, "RELIABILITY_DIR", tmp_path)

    payload = export_web_data._read_reliability_results()

    assert payload["status"] == "available"
    assert payload["study"]["repeated_seeds"] == [42, 43, 44]
    assert "outputs" not in payload["study"]
    assert payload["tables"]["repeated_seed_summary"][0]["terminal_mae_mean"] == 12.5
    assert "risk_coverage" not in payload["tables"]


def test_web_export_limits_risk_curve_to_chart_fields(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "reliability_study.json").write_text(
        json.dumps({"run_id": "run-2"}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "dataset": "FD001",
                "model": "ridge_clean",
                "scenario": "clean",
                "record_type": "risk_coverage",
                "requested_abstention_fraction": 0.2,
                "coverage_fraction": 0.8,
                "retained_mae": 10.0,
                "terminal_engine_mae": 11.0,
            },
            {
                "dataset": "FD001",
                "model": "ridge_clean",
                "scenario": "clean",
                "record_type": "summary",
                "requested_abstention_fraction": 0.0,
                "coverage_fraction": 1.0,
                "retained_mae": 11.0,
            },
        ]
    ).to_csv(metrics_dir / "risk_coverage.csv", index=False)
    monkeypatch.setattr(export_web_data, "RELIABILITY_DIR", tmp_path)

    payload = export_web_data._read_reliability_results()

    assert payload["tables"]["risk_coverage"] == [
        {
            "dataset": "FD001",
            "model": "ridge_clean",
            "scenario": "clean",
            "requested_abstention_fraction": 0.2,
            "coverage_fraction": 0.8,
            "retained_mae": 10.0,
        }
    ]
