from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self):
        self.trained = False

    @abstractmethod
    def predict(self):
        return NotImplementedError("predict() is not implemented")

    @abstractmethod
    def fit(self, X, y):
        return NotImplementedError("fit() is not implemented")

    @abstractmethod
    def save(self):
        return NotImplementedError("save() is not implemented")

    @abstractmethod
    def load(self):
        return NotImplementedError("load() is not implemented")


import GPy
import pickle

class GPR:
    """
    Gaussian Process Regression
    """

    def __init__(self, input_dim, kernel=None):
        self.kernel = kernel or GPy.kern.RBF(input_dim=input_dim)
        self.model = None

    def load(self, file):
        with open(file, "rb") as f:
            self.model = pickle.load(f)      

    def predict(self, X):
        return self.model.predict(X)