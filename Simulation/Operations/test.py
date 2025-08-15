import numpy as np
from pyoptsparse import Optimization, SNOPT


# Define objective function
def objfunc(x):
    return x[0, 0] ** 2 + x[1, 1] ** 2  # Example objective function


# Create optimization problem
opt_prob = Optimization("Example", objfunc)

# Define variable bounds
n_rows = 2
n_cols = 3
x_init = np.zeros((n_rows, n_cols))  # Initial guess for variables
lb = np.full((n_rows, n_cols), -5)  # Lower bounds for variables
ub = np.full((n_rows, n_cols), 5)  # Upper bounds for variables

# Add variables to optimization problem
for i in range(n_rows):
    for j in range(n_cols):
        var_name = f"x_{i}_{j}"
        opt_prob.addVar(
            var_name, "c", lower=lb[i, j], upper=ub[i, j], value=x_init[i, j]
        )

# Set optimizer options
opt_prob.addObj("f")
opt_prob.addConGroup("g", 1, lower=0.0, upper=0.0)

# Create optimizer
opt = SNOPT()

# Solve the optimization problem
sol = opt(opt_prob)
print(sol)
