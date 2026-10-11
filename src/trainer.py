from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.config import ProjectConfig

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def train(self, model, X_train, y_train, X_val, y_val):
        if not len(X_train) or not len(X_val):
            raise ValueError("Training and validation sequences must both be non-empty")
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to train the LSTM model") from exc

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_mae",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_mae",
                factor=0.5,
                patience=max(2, self.config.early_stopping_patience // 3),
                min_lr=1e-5,
                mode="min",
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]

        logger.info(
            "Training on %s sequences; validating on %s (batch=%d, max_epochs=%d)",
            f"{len(X_train):,}",
            f"{len(X_val):,}",
            self.config.batch_size,
            self.config.max_epochs,
        )
        history = model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            shuffle=True,
            verbose=2,
        )
        if not np.isfinite(history.history["loss"]).all():
            raise FloatingPointError("LSTM training produced a non-finite loss")
        return history
