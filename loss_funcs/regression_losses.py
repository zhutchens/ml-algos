import numpy as np

def residual_sum_of_squares(preds: np.ndarray, true_values: np.ndarray) -> np.ndarray:
    return np.sum(preds - true_values)

def mean_squared_error(preds: np.ndarray, true_values: np.ndarray):
    return residual_sum_of_squares(preds, true_values) / preds.shape[0]