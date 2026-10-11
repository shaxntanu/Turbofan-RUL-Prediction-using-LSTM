from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DESTINATION = ROOT / "web" / "public" / "results" / "benchmark.json"
DATASETS = ("FD001", "FD002", "FD003", "FD004")
FIGURE_NAMES = ("predicted_vs_actual", "fractional_life_metrics", "prediction_intervals")


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
        transfer_rows = frame[["source_dataset", "target_dataset", "model", "mae", "rmse"]].to_dict("records")

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
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return DESTINATION


if __name__ == "__main__":
    print(export())
