from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("FD001", "FD002", "FD003", "FD004")


def _default_raw_data_dir() -> Path:
    configured = os.environ.get("CMAPSS_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    bundled = PROJECT_ROOT / "Ref" / "CMAPSSData"
    if bundled.exists():
        return bundled
    return PROJECT_ROOT / "data" / "raw"


@dataclass
class ProjectConfig:
    dataset: str = "FD001"
    raw_data_dir: Path = field(default_factory=_default_raw_data_dir)
    processed_data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")

    window_size: int = 30
    remove_constant_sensors: bool = True
    constant_sensor_threshold: float = 1e-6
    clip_rul: bool = True
    rul_clip_value: int = 125
    normalize: bool = True
    scaler_type: str = "minmax"

    lstm_units: list[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    dense_units: int = 16
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10

    train_ratio: float = 0.8
    random_seed: int = 42
    detrend_degree: int = 2

    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models")

    def __post_init__(self) -> None:
        self.dataset = self.dataset.upper()
        if self.dataset not in DATASET_NAMES:
            raise ValueError(f"dataset must be one of {', '.join(DATASET_NAMES)}")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0.0 < self.train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0 and 1")
        if self.scaler_type not in {"minmax", "standard"}:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")

        for name in ("raw_data_dir", "processed_data_dir", "output_dir", "model_dir"):
            setattr(self, name, Path(getattr(self, name)).expanduser())

    def ensure_output_dirs(self) -> None:
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
