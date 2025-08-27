# First party modules
# from pyoptsparse import SLSQP, Optimization

from resources.configuration.configuration import Configuration

power_degree = Configuration.instance().degree_of_power_in_price_function


from scipy.optimize import minimize
import numpy as np

def ev_decision_making(
    p_0=0.2, alpha=0.1, p_p=1 / 60, max_power=50,
    beta_0=0.6, beta_1=0.2, D=10, T=300, power_degree=1
):
    """
    Solve the EV decision-making optimization using scipy.optimize.minimize.
    """

    # Objective function
    def objective(x):
        energy = x[0]  # x[0] = total energy charged
        duration = x[1]  # x[1] = charging duration in minutes
        power = energy / duration * 60  # convert to kW

        return (
            energy * (p_0 + alpha * (power ** power_degree)) +
            p_p * duration +
            beta_0 * (D - energy) ** 2 +
            beta_1 * ((T - duration) / 60) ** 2
        )

    # Constraint: average charging power <= max_power
    def power_constraint(x):
        energy = x[0]
        duration = x[1]
        return max_power - (energy / duration * 60)

    # Initial guess (can be improved)
    x0 = [1, 15]

    # Bounds: energy between 0 and D, duration between 15 and T
    bounds = [(0, D), (15, T)]

    # Constraint dictionary for SLSQP
    constraints = [
        {'type': 'ineq', 'fun': power_constraint}  # <= max_power
    ]

    # Solve the optimization problem
    result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')

    if result.success:
        print("Optimal solution found:", result.x)
    else:
        print("Optimization failed:", result.message)

    return result.x

# For this model you need to install pyoptsparse

# def ev_decision_making(
#     p_0=0.2, alpha=0.1, p_p=1 / 60, max_power=50, beta_0=0.6, beta_1=0.2, D=10, T=300
# ):
#     # rst begin objfunc
#     def objfunc(xdict):
#         x = xdict["xvars"]
#         funcs = {}
#
#         funcs["obj"] = (
#             x[0] * (p_0 + alpha * (x[0] / x[1] * 60) ** power_degree)
#             + p_p * x[1]
#             + beta_0 * (D - x[0]) ** 2
#             + beta_1 * ((T - x[1]) / 60) ** 2
#         )
#         conval = [0] * 1
#         conval[0] = x[0] / x[1] * 60
#         funcs["con"] = conval
#         fail = False
#
#         return funcs, fail
#
#     # rst begin optProb
#     # Optimization Object
#     optProb = Optimization("TP037 Constraint Problem", objfunc)
#
#     # rst begin addVar
#     # Design Variables
#     optProb.addVarGroup("xvars", 2, "c", lower=[0, 15], upper=[D, T], value=1)
#
#     # rst begin addCon
#     # Constraints
#     optProb.addConGroup("con", 1, lower=0, upper=max_power)
#
#     # rst begin addObj
#     # Objective
#     optProb.addObj("obj")
#
#     # rst begin print
#     # Check optimization problem
#     # print(optProb)
#
#     # rst begin OPT
#     # Optimizer
#     optOptions = {"IPRINT": -1}
#     opt = SLSQP(options=optOptions)
#
#     # rst begin solve
#     # Solve
#     sol = opt(optProb, sens="FD")
#
#     # rst begin check
#     # Check Solution
#     # print(sol)
#
#     return sol.xStar