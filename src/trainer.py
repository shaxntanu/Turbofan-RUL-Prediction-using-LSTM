import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from src.config import ProjectConfig
from src.lstm_model import LSTMModel

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def train(self, model: LSTMModel, X_train, y_train, X_val, y_val):
        # Early Stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=self.config.early_stopping_patience if hasattr(self.config, 'early_stopping_patience') else 10,
            restore_best_weights=True,
            mode='min'
        )

        # Checkpointing
        checkpoint_path = f"{self.config.model_dir}/best_model.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_mae',
            save_best_only=True,
            mode='min'
        )

        logger.info("Starting training...")
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.max_epochs if hasattr(self.config, 'max_epochs') else 100,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 64,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )

        return history
