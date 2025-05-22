from scipy.optimize import minimize
from model import Beam

beam = Beam()


def objective(x):
    return -beam(x)


x0 = [0.8, 200]
result = minimize(
    objective, x0, method="Nelder-Mead", options={"maxiter": 10, "disp": False}
)

print("Optimal input:", result.x)
print("Maximum fitness:", -result.fun)
