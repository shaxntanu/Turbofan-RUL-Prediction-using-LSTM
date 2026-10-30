from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DESTINATION = ROOT / "web" / "public" / "results" / "benchmark.json"
RELIABILITY_DESTINATION = DESTINATION.parent / "reliability"
CPP_BENCHMARK = OUTPUTS / "benchmarks" / "cpp_engine_benchmark.json"
RELIABILITY_DIR = OUTPUTS / "reliability"
DATASETS = ("FD001", "FD002", "FD003", "FD004")
FIGURE_NAMES = ("predicted_vs_actual", "fractional_life_metrics", "prediction_intervals")
RELIABILITY_TABLES = {
    "repeated_seed_summary": "repeated_seed_summary.csv",
    "sensor_stress_summary": "sensor_stress_summary.csv",
    "risk_coverage": "risk_coverage.csv",
    "history_protocol_metrics": "history_protocol_metrics.csv",
    "online_onset_metrics": "online_onset_metrics.csv",
    "rul_cap_sensitivity": "rul_cap_sensitivity.csv",
    "leave_one_subset_out_metrics": "leave_one_subset_out_metrics.csv",
}
RELIABILITY_COLUMNS = {
    "risk_coverage": [
        "dataset", "model", "scenario", "requested_abstention_fraction",
        "coverage_fraction", "retained_mae",
    ],
    "history_protocol_metrics": [
        "dataset", "model", "protocol", "model_window_cycles", "observed_history_cycles",
        "terminal_engine_mae", "terminal_engine_rmse", "interval_coverage",
    ],
    "online_onset_metrics": [
        "dataset", "model", "protocol", "mae", "rmse", "window_f1",
        "engine_detection_rate", "median_detection_delay_cycles",
    ],
    "rul_cap_sensitivity": [
        "dataset", "rul_cap", "validation_overlapping_mae", "terminal_engine_mae",
        "terminal_engine_rmse", "terminal_phm08_sum",
    ],
    "leave_one_subset_out_metrics": [
        "dataset", "model", "source_subsets", "terminal_engine_mae",
        "terminal_engine_rmse", "phm08_score_sum", "target_training_used",
    ],
}


def _read_reliability_results() -> dict | None:
    metrics_dir = RELIABILITY_DIR / "metrics"
    manifest_path = metrics_dir / "reliability_study.json"
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables = {}
    for key, filename in RELIABILITY_TABLES.items():
        path = metrics_dir / filename
        if path.is_file():
            try:
                frame = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            if key == "risk_coverage":
                frame = frame.loc[frame["record_type"] == "risk_coverage"]
            columns = RELIABILITY_COLUMNS.get(key)
            if columns:
                frame = frame[[column for column in columns if column in frame.columns]]
            tables[key] = json.loads(frame.to_json(orient="records"))

    study_fields = (
        "run_id",
        "datasets",
        "repeated_seeds",
        "lstm_seeds",
        "extended_model_comparisons",
        "short_history_protocols",
        "online_onset_protocol",
        "window_size",
        "rul_cap",
        "rul_cap_sensitivity",
        "validation_protocol",
        "test_protocol",
        "domain_protocol",
        "domain_note",
        "limitations",
    )
    return {
        "status": "available",
        "study": {key: manifest[key] for key in study_fields if key in manifest},
        "tables": tables,
    }


def _reliability_snapshot_for_dataset(reliability: dict | None, dataset: str) -> dict:
    if reliability is None:
        return {"status": "not_exported", "study": {}, "tables": {}}

    tables = {
        name: [row for row in rows if row.get("dataset") == dataset]
        for name, rows in reliability.get("tables", {}).items()
    }
    return {
        "status": reliability.get("status", "available"),
        "study": reliability.get("study", {}),
        "tables": tables,
    }


def _write_reliability_snapshots(reliability: dict | None) -> list[Path]:
    RELIABILITY_DESTINATION.mkdir(parents=True, exist_ok=True)
    destinations = []
    for dataset in DATASETS:
        path = RELIABILITY_DESTINATION / f"{dataset.lower()}.json"
        snapshot = _reliability_snapshot_for_dataset(reliability, dataset)
        path.write_text(
            json.dumps(snapshot, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        destinations.append(path)
    return destinations


def export() -> Path:
    datasets = {}
    for name in DATASETS:
        result_path = OUTPUTS / name / "metrics" / "results.json"
        if not result_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        selected = result["selected_model"]
        datasets[name] = {
            "run_id": result["run_id"],
            "selected_model": selected,
            "test_metrics": result["test_metrics"][selected],
            "model_metrics": result["test_metrics"],
            "validation_metrics": result["validation_metrics"],
            "uncertainty": result.get("uncertainty_by_model", {}).get(
                selected,
                result["uncertainty"],
            ),
            "dataset_summary": result["dataset_summary"],
            "sensor_screening": result["sensor_screening"],
            "engine_robustness": result["engine_robustness"],
            "life_stage_metrics": result.get("fractional_life_analysis", {}).get("metrics", []),
            "figures": {
                key: f"/results/{name.lower()}-{key.replace('_', '-')}.png"
                for key in FIGURE_NAMES
                if key in result.get("figures", {})
            },
        }

    transfer_path = OUTPUTS / "transfer" / "transfer_metrics.csv"
    transfer_rows = []
    if transfer_path.is_file():
        frame = pd.read_csv(transfer_path)
        columns = ["source_dataset", "target_dataset", "model", "mae", "rmse"]
        if "n_samples" in frame.columns:
            columns.append("n_samples")
        transfer_rows = frame[columns].to_dict("records")

    cpp_benchmark = (
        json.loads(CPP_BENCHMARK.read_text(encoding="utf-8"))
        if CPP_BENCHMARK.is_file()
        else {
            "status": "not_executed",
            "reason": "Run python -m src.run_cpp_benchmark to generate native kernel timings.",
            "datasets": [],
        }
    )

    payload = {
        "schema_version": 1,
        "protocol": {
            "target": "RUL in cycles",
            "label_cap": 125,
            "window_size": 30,
            "selection": "minimum validation MAE",
            "domain_note": (
                "The supplied papers predict turbine gas temperature on proprietary fleet data. "
                "This repository evaluates method adaptations on simulated NASA C-MAPSS RUL labels."
            ),
        },
        "datasets": datasets,
        "transfer": {
            "rows": transfer_rows,
            "limitation": (
                "FD001-FD004 vary simulated operating conditions and fault modes; they are proxy domains, "
                "not separate real-world engine fleets."
            ),
        },
        "cpp_engine": cpp_benchmark,
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_reliability_snapshots(_read_reliability_results())
    return DESTINATION


if __name__ == "__main__":
    print(export())
