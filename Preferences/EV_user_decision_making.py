# First party modules
from pyoptsparse import SLSQP, Optimization

from Environment.helper.configuration.configuration import Configuration

power_degree = Configuration.instance().degree_of_power_in_price_function


def ev_decision_making(
    p_0=0.2, alpha=0.1, p_p=1 / 60, max_power=50, beta_0=0.6, beta_1=0.2, D=10, T=300
):
    # rst begin objfunc
    def objfunc(xdict):
        x = xdict["xvars"]
        funcs = {}

        funcs["obj"] = (
            x[0] * (p_0 + alpha * (x[0] / x[1] * 60) ** power_degree)
            + p_p * x[1]
            + beta_0 * (D - x[0]) ** 2
            + beta_1 * ((T - x[1]) / 60) ** 2
        )
        conval = [0] * 1
        conval[0] = x[0] / x[1] * 60
        funcs["con"] = conval
        fail = False

        return funcs, fail

    # rst begin optProb
    # Optimization Object
    optProb = Optimization("TP037 Constraint Problem", objfunc)

    # rst begin addVar
    # Design Variables
    optProb.addVarGroup("xvars", 2, "c", lower=[0, 15], upper=[D, T], value=1)

    # rst begin addCon
    # Constraints
    optProb.addConGroup("con", 1, lower=0, upper=max_power)

    # rst begin addObj
    # Objective
    optProb.addObj("obj")

    # rst begin print
    # Check optimization problem
    # print(optProb)

    # rst begin OPT
    # Optimizer
    optOptions = {"IPRINT": -1}
    opt = SLSQP(options=optOptions)

    # rst begin solve
    # Solve
    sol = opt(optProb, sens="FD")

    # rst begin check
    # Check Solution
    # print(sol)

    return sol.xStar
