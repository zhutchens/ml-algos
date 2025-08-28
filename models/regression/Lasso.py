import numpy as np
from models.regression import LinearRegression

class Lasso(LinearRegression):
    def __init__(self, penalty: float):
        super().__init__()
        self.penalty = penalty

    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01, max_steps: int = 1000):
        '''
        Coordinate descent on the lasso objective function.
        Gradient descent cannot be used due to the absolute value term in L1 regularization
        '''
        if x.ndim == 1:
            raise ValueError(f'X cannot be 1 dimension. reshape with .reshape(-1, 1)')

        self.coefficents = np.zeros(x.shape[1])
        self.intercept = np.zeros(1)

        params = self.get_parameters()
        indices = np.arange(params.size)
        np.random.shuffle(indices)
        
        for _ in range(max_steps):
            for index in indices:
                # compute preds using parameters
                preds = self.predict(x)
                if index != params.size - 1:
                    # compute gradient of selected weight using soft-thresholding
                    weight_grad = -(1 / (x.shape[0])) * np.sum(x[:, index] * (y - preds))
                    params[index] = np.sign(params[index] - lr * weight_grad) * max(0, abs(params[index] - lr * weight_grad) - lr * self.penalty)
                else:
                    # compute gradient of bias term using partial derivative of RSS cost function
                    bias_grad = -(1 / (x.shape[0])) * np.sum(y - preds)
                    params[index] -= lr * bias_grad
                
                self.update_parameters(params)

