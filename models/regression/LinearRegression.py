from models.BaseModel import Model
import numpy as np

class LinearRegression(Model):
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        # loss function for linear regression is nice and has a closed-form solution, so using that over optimization
        # add feature of 1's to get bias 
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1)
        self.weights = np.linalg.inv(np.dot(X.transpose(), X)) @ np.dot(X.transpose(), y)

        self.bias = self.weights[-1]
        self.weights = self.weights[:-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias