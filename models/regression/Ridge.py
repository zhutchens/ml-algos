import numpy as np
from cost_functions import RidgeLoss, RSS
from optimization import gradient_descent
from models.regression import LinearRegression
from functools import partial

class Ridge(LinearRegression):
    def __init__(self, penalty: float):
        super().__init__()
        self.penalty = penalty

    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01):
        if x.ndim == 1:
            raise ValueError(f'X cannot be 1 dimension. reshape with .reshape(-1, 1)')
        
        self.coefficents = np.zeros(x.shape[1])
        self.intercept = np.zeros(1)

        cost_func = partial(RidgeLoss, loss_func = RSS, penalty = self.penalty, params = self.get_parameters())
        gradient_descent(self, cost_func, x, y, lr)
