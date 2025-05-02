from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all models
    """

    def __init__(self):
        self.trained = False

    @abstractmethod
    def predict(self, X):
        pass

    def fit(self, X, y):
        raise NotImplementedError("This model type does not support training")

    def save(self):
        raise NotImplementedError("This model type does not support saving")

    def load(self):
        raise NotImplementedError("This model type does not support loading")


import GPy
import pickle


class GPR(Model):
    """
    Pre-trained Gaussian Process Regression model
    """

    def __init__(self):
        super().__init__()
        self.model = None

    def predict(self, X):
        return self.model.predict(X)

    def load(self, file):
        with open(file, "rb") as f:
            self.model = pickle.load(f)
            self.trained = True
