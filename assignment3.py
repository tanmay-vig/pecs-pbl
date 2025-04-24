import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

# Branin function (benchmark function for BO)
def branin(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    a = 1.0
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    return (a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s).reshape(-1, 1)

# Acquisition function: Expected Improvement
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.min(Y_sample)
    
    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

# Propose next sampling point
def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    dim = bounds.shape[0]
    min_val = 1e20
    min_x = None

    def min_obj(X):
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x
    return min_x.reshape(1, -1)

# Bayesian Optimization loop using initial points from CSV
def bayesian_optimization(n_iters, sample_loss, bounds, csv_path):
    # Load initial sample points from CSV
    df = pd.read_csv(csv_path)
    X_sample = df[["x1", "x2"]].values
    Y_sample = df[["branin_value"]].values

    kernel = Matern(length_scale=1.0)
    gpr = GaussianProcessRegressor(kernel=kernel)

    for i in range(n_iters):
        gpr.fit(X_sample, Y_sample)
        x_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)
        y_next = sample_loss(x_next)

        X_sample = np.vstack((X_sample, x_next))
        Y_sample = np.vstack((Y_sample, y_next))

        print(f"Iteration {i+1}, Best Value: {np.min(Y_sample)}")

    return X_sample, Y_sample

# Define bounds for the Branin function
bounds = np.array([[-5, 10], [0, 15]])

# Run the optimizer using 50 sample points from CSV
csv_path = "branin_sample_points_50.csv"  # Update if the file is in a different location
X_observed, Y_observed = bayesian_optimization(
    n_iters=20,
    sample_loss=branin,
    bounds=bounds,
    csv_path=csv_path
)

# Plot observed points
plt.scatter(X_observed[:, 0], X_observed[:, 1], c=Y_observed.ravel(), cmap='viridis')
plt.colorbar(label='Branin Value')
plt.title("Sampled Points in Bayesian Optimization")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
