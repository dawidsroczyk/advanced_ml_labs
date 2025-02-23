from base_class import AmlClassifier
import numpy as np
from common import empirical_covariance

class LDA(AmlClassifier):

    def __init__(self):
        self.W = None
        self.m0 = None
        self.m1 = None
        self.p0 = None
        self.p1 = None


    def fit(self, X: np.array, y: np.array) -> None:
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert X.shape[0] == y.shape[0]
        assert np.all((y == 0) | (y == 1))

        X_0 = X[y == 0]
        X_1 = X[~(y == 0)]

        n0 = X_0.shape[0]
        n1 = X_1.shape[0]
        S0 = empirical_covariance(X_0)
        S1 = empirical_covariance(X_1)
        W = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 -2)
        diag, U = np.linalg.eigh(W)
        D = np.diag(diag)

        m0 = X_0.mean(axis=0)
        m1 = X_1.mean(axis=0)

        p0 = n0 / (n0 + n1)
        p1 = 1 - p0

        self.W = W
        self.m0 = m0
        self.m1 = m1
        self.p0 = p0
        self.p1 = p1
    

    def calculate_classification_value(self, X_test: np.array) -> np.array:
        comp_1 = np.linalg.matmul(np.linalg.matmul(self.m0 - self.m1, np.linalg.inv(self.W)), X_test.T)
        comp_2 = np.linalg.matmul(np.linalg.matmul(self.m0 - self.m1, np.linalg.inv(self.W)), self.m0 + self.m1)
        comp_3 = np.log(self.p0 / self.p1)
        class_fun_value = comp_1 - 0.5 * comp_2 + comp_3
        return class_fun_value
    

    def predict(self, X_test: np.array) -> np.array:
        class_fun_values = self.calculate_classification_value(X_test)
        result = class_fun_values > 0
        result = (~result).astype(int)
        return result
    

    def predict_proba(self, X_test: np.array) -> np.array:
        class_fun_values = self.calculate_classification_value(X_test)
        proba = 1 / (np.exp(class_fun_values) + 1)
        return proba
    
    def get_params(self) -> list:
        return [self.W, self.m0, self.m1, self.p0, self.p1]