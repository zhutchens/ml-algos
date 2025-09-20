from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
