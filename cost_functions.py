import numpy as np

# linear model loss functions
def RSS(y_true: np.ndarray, y_preds: np.ndarray) -> float:
    return np.sum((y_true - y_preds) ** 2)


def MSE(y_true: np.ndarray, y_preds: np.ndarray) -> float:
    return RSS(y_true, y_preds) / y_true.shape[0]


# def LassoLoss(params: np.ndarray, x: np.ndarray, y_true: np.ndarray, loss_func, lam: float) -> float:
#     return loss_func(params, x, y_true) + lam * np.sum(np.absolute(params[:-1]))

def LassoLoss(y_true: np.ndarray, y_preds: np.ndarray, loss_func: callable, penalty: float, params: np.ndarray) -> np.ndarray:
    return loss_func(y_true, y_preds) + penalty * np.sum(np.absolute(params[:-1]))


def RidgeLoss(y_true: np.ndarray, y_preds: np.ndarray, loss_func: callable, penalty: float, params: np.ndarray) -> float:
    return loss_func(y_true, y_preds) + penalty * np.sum(np.square(params[:-1]))


def ElasticNetLoss(y_true: np.ndarray, y_preds: np.ndarray, loss_func, penalty: float, l1_ratio: float, params: np.ndarray) -> float:
    # return loss_func(y_true, y_preds) + LassoLoss(y_true, y_preds, loss_func, penalty * l1_ratio, params) + \
    #     RidgeLoss(y_true, y_preds, loss_func, penalty * (1 - l1_ratio), params)
    return loss_func(y_true, y_preds) + ((penalty * l1_ratio) * np.sum(np.absolute(params[:-1]))) * ((penalty * (1 - l1_ratio)) * np.sum(np.square(params[:-1])))



# classification loss functions