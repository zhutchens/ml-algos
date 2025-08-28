import numpy as np
from cost_functions import MSE, ElasticNetLoss
from optimization import gradient_descent
from models.regression import LinearRegression
from functools import partial

class ElasticNet(LinearRegression):
    def __init__(self, L1_penalty: float, L2_penalty: float):
        super().__init__()
        self.L1_penalty = L1_penalty
        self.L2_penalty = L2_penalty

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01):
        if X.ndim == 1:
            raise ValueError(f'X cannot be 1 dimension. reshape with .reshape(-1, 1)')
        
        self.a = np.zeros(X.shape[1])
        self.intercept = np.zeros(1)

        # cost_func = partial(ElasticNetLoss, x = X, y_true = y, loss_func = MSE, L1_penalty = self.L1_penalty, L2_penalty = self.L2_penalty)
        cost_func = partial(ElasticNetLoss, y_true )
        params = gradient_descent(self, cost_func, lr)

        self.a = params[:-1]
        self.intercept = [params[-1]]