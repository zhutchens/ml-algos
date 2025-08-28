import numpy as np
from models import Model

# for functions of single variables, e.g., f(x)
def compute_derivative(func, x: float, h: float = 1e-7) -> float:
    # central finite difference 
    return (func(x + h) - func(x - h)) / (2 * h)


# for multivariable functions, e.g., f(x, y)
def compute_gradient(cost_func, param_list: np.ndarray, h: float = 1e-7) -> np.ndarray:
    gradient = np.zeros(len(param_list))

    for i in range(len(gradient)):
        params_plus = param_list.copy()
        params_minus = param_list.copy()

        params_plus[i] += h
        params_minus[i] -= h

        # use central finite difference to estimate partial derivatives
        gradient[i] = (cost_func(params_plus) - cost_func(params_minus)) / (2 * h)

    return gradient


def gradient_descent(estimator: Model, metric: callable, x: np.ndarray, y: np.ndarray, lr: float, max_steps: int = 1000, h: float = 1e-7) -> np.ndarray:
    params = estimator.get_parameters()  
    for _ in range(max_steps):
        # gradients = compute_gradient(cost_func, params, h)
        gradients = np.zeros(len(params))

        for i in range(len(gradients)):
            params_plus = params.copy()
            params_minus = params.copy()

            params_plus[i] += h
            estimator.update_parameters(params_plus)
            preds_plus = estimator.predict(x)

            params_minus[i] -= h
            estimator.update_parameters(params_minus)
            preds_minus = estimator.predict(x)

            # use central finite difference to estimate partial derivatives
            # gradients[i] = (cost_func(params_plus) - cost_func(params_minus)) / (2 * h)
            gradients[i] = (metric(y, preds_plus) - metric(y, preds_minus)) / (2 * h)


        for i in range(len(gradients)):
            params[i] -= lr * gradients[i]

        estimator.update_parameters(params)
        
    # return params


def adam(estimator: Model, cost_func, lr: float, max_steps: int = 1000, h: float = 1e-7) -> np.ndarray:
    pass


