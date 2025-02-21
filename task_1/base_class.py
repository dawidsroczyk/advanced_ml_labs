from abc import ABC, abstractmethod
import numpy as np

class aml_classifier(ABC):

    @abstractmethod
    def fit(X: np.array, y: np.array):
        pass
    
    @abstractmethod
    def predict_proba(X_test: np.array) -> np.array:
        pass
    
    @abstractmethod
    def predict(X_test: np.array) -> np.array:
        pass
    
    @abstractmethod
    def get_params() -> list:
        pass