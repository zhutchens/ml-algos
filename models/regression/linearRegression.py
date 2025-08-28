import numpy as np
from cost_functions import RSS
from optimization import gradient_descent
from models import Model

class LinearRegression(Model):
    def __init__(self):
        self.coefficents = None
        self.intercept = None


    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01):
        if x.ndim == 1:
            raise ValueError(f'X cannot be 1 dimension. reshape with .reshape(-1, 1)')
        
        self.coefficents = np.zeros(x.shape[1])
        self.intercept = np.zeros(1)

        gradient_descent(self, RSS, x, y, lr)


    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.coefficents) + self.intercept


    def get_parameters(self) -> np.ndarray:
        return np.append(self.coefficents, self.intercept)
    

    def update_parameters(self, params: np.ndarray) -> None:
        self.coefficents = params[:-1]
        self.intercept = params[-1]