from base_class import AMLClassifier
from common import naive_covariance
import numpy as np


class BinaryNB(AMLClassifier):

    def __init__(self):
        self.S0 = None
        self.S1 = None
        self.m0 = None
        self.m1 = None
        self.p0 = None
        self.p1 = None

    
    def fit(self, X: np.array, y: np.array):
        assert len(X.shape) == 2
        assert len(y.shape) == 1
        assert X.shape[0] == y.shape[0]
        assert np.all((y == 0) | (y == 1))

        X_0 = X[y == 0]
        X_1 = X[~(y == 0)]

        n0 = X_0.shape[0]
        n1 = X_1.shape[0]
        p0 = n0 / (n0 + n1)
        p1 = 1 - p0
        m0 = X_0.mean(axis=0)
        m1 = X_1.mean(axis=0)
        S0 = naive_covariance(X_0)
        S1 = naive_covariance(X_1)

        self.S0 = S0
        self.S1 = S1
        self.m0 = m0
        self.m1 = m1
        self.p0 = p0
        self.p1 = p1
    

    def calculate_classification_value(self, X_test: np.array) -> np.array:
        S0 = self.S0
        S1 = self.S1
        m0 = self.m0
        m1 = self.m1
        p0 = self.p0
        p1 = self.p1

        comp1 = np.log(np.linalg.det(S0) / np.linalg.det(S1))
        comp2 = np.linalg.matmul(X_test, (np.linalg.matmul(np.linalg.inv(S1), m1) - np.linalg.matmul(np.linalg.inv(S0), m0)))
        comp3 = np.sum(X_test @ (np.linalg.inv(S1) - np.linalg.inv(S0)) * X_test, axis=1)
        comp4 = np.linalg.matmul(np.linalg.matmul(m1, np.linalg.inv(S1)), m1.T)
        comp5 = np.linalg.matmul(np.linalg.matmul(m0, np.linalg.inv(S0)), m0.T)
        comp6 = np.log(p1) - np.log(p0)
        class_fun_value = 0.5 * comp1 + comp2 - 0.5 * comp3 - 0.5 * comp4 + 0.5 * comp5 + comp6

        return class_fun_value
    
    
    def predict_proba(self, X_test: np.array) -> np.array:
        class_fun_values = self.calculate_classification_value(X_test)
        proba = 1 / (np.exp(-class_fun_values) + 1)
        return proba
    
    
    def predict(self, X_test: np.array) -> np.array:
        class_fun_values = self.calculate_classification_value(X_test)
        result = class_fun_values > 0
        result = result.astype(int)
        return result
    
    
    def get_params(self) -> list:
        result = [self.S0, self.S1, self.m0, self.m1, self.p0, self.p1]
        return result