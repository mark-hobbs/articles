from abc import ABC, abstractmethod


class Model(ABC):

    @abstractmethod
    def __call__(self, X):
        """
        Evaluate the model on input X
        """
        pass

    @abstractmethod
    def get_input_size(self) -> int:
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        pass

    @abstractmethod
    def forward(self, X):
        """
        Alias for __call__, can override for clarity
        """
        pass

    def fitness(self):
        """
        Objective function
        """
        pass


import pypd

from utils import setup_problem, objective


class Beam(Model):

    def __init__(self):
        super().__init__()
        self.simulation = pypd.Simulation(n_time_steps=100000, damping=0)

    def __call__(self, x):
        model = setup_problem(x[0], x[1])
        self.simulation.run(model)
        return float(objective(model))

    def get_input_size(self) -> int:
        """
        alpha and k
        """
        return 2

    def get_output_size(self) -> int:
        return None

    def forward(self, x):
        model = setup_problem(x[0], x[1])
        self.simulation.run(model)

    def fitness(self):
        """
        Minimise the discrepancy between the experimental load-CMOD data
        and numerical predictions
        """
        pass
