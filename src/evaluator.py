import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    @staticmethod
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return {
            'mae': mae,
            'rmse': rmse
        }

    @staticmethod
    def analyze_errors(y_true, y_pred):
        errors = y_true - y_pred
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors))
        }
