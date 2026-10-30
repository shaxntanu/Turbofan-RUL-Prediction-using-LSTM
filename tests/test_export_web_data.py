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


def test_export_keeps_core_bundle_small_and_partitions_reliability(tmp_path, monkeypatch):
    outputs = tmp_path / "outputs"
    metrics_dir = outputs / "FD001" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "results.json").write_text(
        json.dumps(
            {
                "run_id": "run-core",
                "selected_model": "ridge",
                "test_metrics": {"ridge": {"mae": 12.0}},
                "validation_metrics": {"ridge": {"mae": 11.0}},
                "uncertainty": {"picp": 0.95},
                "dataset_summary": {"test_engines": 100},
                "sensor_screening": {},
                "engine_robustness": {},
                "fractional_life_analysis": {"metrics": []},
                "figures": {},
            }
        ),
        encoding="utf-8",
    )
    reliability = {
        "status": "available",
        "study": {"run_id": "study-1"},
        "tables": {
            "risk_coverage": [
                {"dataset": "FD001", "retained_mae": 10.0},
                {"dataset": "FD002", "retained_mae": 20.0},
            ]
        },
    }
    destination = tmp_path / "web" / "results" / "benchmark.json"
    reliability_destination = destination.parent / "reliability"
    monkeypatch.setattr(export_web_data, "OUTPUTS", outputs)
    monkeypatch.setattr(export_web_data, "DESTINATION", destination)
    monkeypatch.setattr(export_web_data, "RELIABILITY_DESTINATION", reliability_destination)
    monkeypatch.setattr(export_web_data, "CPP_BENCHMARK", tmp_path / "missing-cpp.json")
    monkeypatch.setattr(export_web_data, "_read_reliability_results", lambda: reliability)

    export_web_data.export()

    core = json.loads(destination.read_text(encoding="utf-8"))
    assert set(core["datasets"]) == {"FD001"}
    assert "reliability" not in core
    fd001 = json.loads((reliability_destination / "fd001.json").read_text(encoding="utf-8"))
    fd002 = json.loads((reliability_destination / "fd002.json").read_text(encoding="utf-8"))
    assert fd001["tables"]["risk_coverage"] == [
        {"dataset": "FD001", "retained_mae": 10.0}
    ]
    assert fd002["tables"]["risk_coverage"] == [
        {"dataset": "FD002", "retained_mae": 20.0}
    ]
