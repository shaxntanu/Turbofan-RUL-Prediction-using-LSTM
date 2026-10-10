import tensorflow as tf
from tensorflow.keras import layers, models
import logging
from src.config import ProjectConfig

logger = logging.getLogger(__name__)

class LSTMModel:
    def __init__(self, config: ProjectConfig, input_shape: tuple):
        self.config = config
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self) -> models.Sequential:
        model = models.Sequential()

        # LSTM Layer 1
        model.add(layers.LSTM(
            units=self.config.lstm_units[0] if hasattr(self.config, 'lstm_units') else 64,
            input_shape=self.input_shape,
            return_sequences=True
        ))
        model.add(layers.Dropout(self.config.dropout_rate if hasattr(self.config, 'dropout_rate') else 0.2))

        # LSTM Layer 2
        model.add(layers.LSTM(
            units=self.config.lstm_units[1] if hasattr(self.config, 'lstm_units') else 32,
            return_sequences=False
        ))
        model.add(layers.Dropout(self.config.dropout_rate if hasattr(self.config, 'dropout_rate') else 0.2))

        # Dense Layers
        model.add(layers.Dense(self.config.dense_units if hasattr(self.config, 'dense_units') else 16, activation='relu'))
        model.add(layers.Dense(1)) # Output RUL

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate if hasattr(self.config, 'learning_rate') else 0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def save(self, path: str):
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        self.model = models.load_model(path)
        logger.info(f"Model loaded from {path}")
