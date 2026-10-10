from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class ProjectConfig:
    # Dataset
    dataset: str = "FD001"
    raw_data_dir: str = r"C:\\Users\\Shantanu\\Documents\\GitHub\\Turbofan-RUL-Prediction-using-LSTM\\Ref\\CMAPSSData"
    processed_data_dir: str = "data/processed"

    # Preprocessing
    window_size: int = 30
    remove_constant_sensors: bool = True
    constant_sensor_threshold: float = 1e-6

    # RUL
    clip_rul: bool = False
    rul_clip_value: int = 125

    # Normalization
    normalize: bool = True
    scaler_type: str = "minmax"

    # Model
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    dense_units: int = 16
    learning_rate: float = 0.001

    # Training
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10

    # Split
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # Reproducibility
    random_seed: int = 42

    # Paths
    output_dir: str = "outputs"
    model_dir: str = "models"

    def __post_init__(self):
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
