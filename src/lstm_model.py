from __future__ import annotations

import logging

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


class LSTMModel:
    def __init__(self, config: ProjectConfig, input_shape: tuple[int, int]):
        self.config = config
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError(
                "TensorFlow is required for LSTM training. Install the project "
                "dependencies in the Python 3.12 environment pinned by .python-version."
            ) from exc

        model = tf.keras.Sequential(name="cmapss_rul_lstm")
        model.add(tf.keras.layers.Input(shape=self.input_shape))
        for index, units in enumerate(self.config.lstm_units):
            model.add(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=index < len(self.config.lstm_units) - 1,
                )
            )
            model.add(tf.keras.layers.Dropout(self.config.dropout_rate))
        model.add(tf.keras.layers.Dense(self.config.dense_units, activation="relu"))
        model.add(tf.keras.layers.Dense(1, name="rul_cycles"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )
        logger.info("Built %s with input shape %s", model.name, self.input_shape)
        return model

    def save(self, path: str):
        self.model.save(path)
        logger.info("Saved model to %s", path)

    def load(self, path: str):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load this LSTM model") from exc
        self.model = tf.keras.models.load_model(path)
        return self
