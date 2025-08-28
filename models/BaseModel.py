from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_parameters(self) -> np.ndarray:
        pass

    @abstractmethod
    def update_parameters(self, params: np.ndarray) -> None:
        pass