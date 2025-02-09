# # First party modules
from pyoptsparse import SLSQP, Optimization, pyOpt_objective
import numpy as np

from Environment.helper.configuration.configuration import Configuration


# def nonlinear_pricing(vehicles_list,electricity_cost, PV, base_load, sim_time, p_0=np.zeros(24) + 0.2, smart_charging=False):
#     # rst begin objfunc
#     vehicle_range = range(len(vehicles_list))
#     peak_charge_penalty = Configuration.instance().peak_cost
#     beta = [vehicle.utility_beta for vehicle in vehicles_list]
#     D = [vehicle.energy_requested for vehicle in vehicles_list]
#     T = np.zeros(len(vehicles_list))
#     A = np.zeros(len(vehicles_list))
#     De = np.zeros(len(vehicles_list))
#     delta_time = 60
#     BIG_T = 24
#     time_range = range(BIG_T)
#     AB = {} #np.full((len(vehicles_list), len(time_range)), 1)
#     U = {}
#     C = {}
#     print(PV)
#     print(base_load)
#     for t in time_range:
#         hour = int((t % 24))
#         C[t] = electricity_cost[hour]
#     for i in vehicle_range:
#         # SOC[i.id] = i.energy_charged
#         A[i] = int(vehicles_list[i].arrival_period/delta_time)
#         beta[i] = vehicles_list[i].utility_beta
#         D[i] = vehicles_list[i].energy_requested
#         De[i] = int(vehicles_list[i].departure_period/delta_time)
#         T[i] = int(vehicles_list[i].park_duration/60)
#
#         for t in time_range:
#             if A[i] <= t <= De[i]:
#                 U[i, t] = 1
#             else:
#                 U[i, t] = 0
#         for t in time_range:
#             if A[i] == t:
#                 AB[i, t] = 1
#             else:
#                 AB[i, t] = 0
#
#     # T = [vehicle.energy_requested for vehicle in vehicles_list]
#     def objfunc(xdict):
#         x = np.full((len(vehicle_range), len(time_range)), 0)
#         if Configuration.instance().capacity_pricing:
#             alpha = xdict['alpha']
#         else:
#             alpha = np.zeros(BIG_T) + 0
#         if Configuration.instance().dynamic_fix_term_pricing:
#             p_0 = xdict['p_0']
#         else:
#             p_0 = np.zeros(BIG_T) + Configuration.instance().instance().price_parameters[0]
#         if smart_charging:
#             y = xdict['y'].reshape(len(vehicle_range), len(time_range))
#             p_star = xdict['p_star']
#         for i in vehicle_range:
#             for t in time_range:
#                 x[i, t] = ((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t]) / (2 * (beta[i] * T[i] + alpha[t]))
#                 # if x[i, t] != 0:
#                 #     print(np.sum(x))
#         funcs = {}
#
#         energy_charged = 0
#         funcs["obj"] = 0
#
#
#         for t in time_range:
#             if smart_charging:
#                 funcs["obj"] -= electricity_cost[t] * PV[t]
#             for i in vehicle_range:
#
#                 power = x[i, t]/T[i]
#
#                 # funcs["obj"] -= (p_0[t] -0.15 + alpha[t] * power) * x[i, t]
#                                  # -np.sum(np.array(electricity_cost)*y[i, :]))
#                 if not smart_charging:
#                     if Configuration.instance().dynamic_fix_term_pricing and not Configuration.instance().capacity_pricing:
#                         funcs["obj"] -= ((p_0[t] - 0.15) * ((2*beta[i] * D[i] - p_0[t])*AB[i, t]/(2*beta[i])))
#                         (2*beta[i] * D[i] - p_0[t])*AB[i, t]/(2*beta[i])
#                     else:
#                         funcs["obj"] -= ((p_0[t] - 0.15 + alpha[t] * (((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t])
#                                                                       / (2 * (beta[i] * T[i] + alpha[t]))) / T[i]) * (
#                                                      ((2 * beta[i] * D[i] - p_0[t])
#                                                       * T[i] * AB[i, t]) / (2 * (beta[i] * T[i] + alpha[t]))))
#
#                 else:
#
#                     funcs["obj"] -= ((p_0[t] + alpha[t] * (((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t])
#                                                                   / (2 * (beta[i] * T[i] + alpha[t]))) / T[i]) * (
#                                                  ((2 * beta[i] * D[i] - p_0[t])
#                                                   * T[i] * AB[i, t]) / (2 * (beta[i] * T[i] + alpha[t])))
#                                      - electricity_cost[t] * y[i,t] * U[i, t])
#
#                 energy_charged += (((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t]) / (2 * (beta[i] * T[i] + alpha[t])))
#         if smart_charging:
#             funcs["obj"] -= - p_star * peak_charge_penalty
#         conval = [0] * BIG_T
#         for t in time_range:
#             conval[t] = alpha[t] + p_0[t]
#         funcs["con"] = conval
#         if smart_charging:
#             conval = [0] * BIG_T * len(vehicle_range)
#             for i in vehicle_range:
#                 for t in time_range:
#                     conval[i*BIG_T+t] = y[i,t]-U[i,t]*50
#             funcs["con1"] = conval
#             conval = [0] * len(vehicle_range)
#             for i in vehicle_range:
#                 conval[i] = np.sum(y[i,:]) - ((2 * beta[i] * D[i] - p_0[int(A[i])]) * T[i]) / (2 * (beta[i] * T[i] + alpha[int(A[i])]))
#             funcs["con2"] = conval
#             conval = [0] * BIG_T
#             for t in time_range:
#                 conval[t] = np.sum(y[:, t]) + base_load[t] - PV[t] - p_star
#             funcs["con3"] = conval
#         fail = False
#         print(energy_charged, funcs["obj"])
#
#         return funcs, fail
#
#     # rst begin optProb
#     # Optimization Object
#     optProb = Optimization("TP037 Constraint Problem", objfunc)
#
#     # rst begin addVar
#     # Design Variables
#     optProb.addVarGroup("alpha", BIG_T, "c", lower=np.zeros(BIG_T), upper=np.zeros(BIG_T) + 1, value=1)
#     if Configuration.instance().dynamic_fix_term_pricing:
#         optProb.addVarGroup("p_0", BIG_T, "c", lower=np.zeros(BIG_T), upper=np.zeros(BIG_T) + 2, value=1)
#     if smart_charging:
#         optProb.addVarGroup("y", BIG_T*len(vehicles_list), "c", lower=np.zeros(BIG_T*len(vehicles_list)),
#                             upper=np.zeros(BIG_T*len(vehicles_list)) + 50, value=1)
#         optProb.addVar(name="p_star", varType="c", lower=0, upper=1200, value=1)
#
#     # rst begin addCon
#     # Constraints
#     optProb.addConGroup("con", BIG_T, lower=0)
#     if smart_charging:
#         optProb.addConGroup("con1", BIG_T*len(vehicles_list), upper=0)
#         optProb.addConGroup("con2", len(vehicles_list), lower=0)
#         optProb.addConGroup("con3", BIG_T, upper=0)
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
#     print(sol.objectives['obj'])
#     print(sol)
#     return (sol.xStar)


def nonlinear_pricing(
    vehicles_list,
    electricity_cost,
    PV,
    base_load,
    sim_time,
    p_0=np.zeros(24) + 0.2,
    smart_charging=True,
):
    # rst begin objfunc
    vehicle_range = range(len(vehicles_list))
    peak_charge_penalty = Configuration.instance().peak_cost
    beta = [vehicle.utility_beta for vehicle in vehicles_list]
    D = [vehicle.energy_requested for vehicle in vehicles_list]
    T = np.zeros(len(vehicles_list))
    A = np.zeros(len(vehicles_list))
    De = np.zeros(len(vehicles_list))
    delta_time = 60
    BIG_T = 24
    time_range = range(BIG_T)
    AB = {}  # np.full((len(vehicles_list), len(time_range)), 1)
    U = {}
    C = {}
    print(PV)
    print(base_load)
    for t in time_range:
        hour = int((t % 24))
        C[t] = electricity_cost[hour]
    for i in vehicle_range:
        # SOC[i.id] = i.energy_charged
        A[i] = int(vehicles_list[i].arrival_period / delta_time)
        beta[i] = vehicles_list[i].utility_beta
        D[i] = vehicles_list[i].energy_requested
        De[i] = int(vehicles_list[i].departure_period / delta_time)
        T[i] = int(vehicles_list[i].park_duration / 60)

        for t in time_range:
            if A[i] <= t <= De[i]:
                U[i, t] = 1
            else:
                U[i, t] = 0
        for t in time_range:
            if A[i] == t:
                AB[i, t] = 1
            else:
                AB[i, t] = 0

    # T = [vehicle.energy_requested for vehicle in vehicles_list]
    def objfunc(xdict):
        x = np.full((len(vehicle_range), len(time_range)), 0)
        if Configuration.instance().capacity_pricing:
            alpha = xdict["alpha"]
        else:
            alpha = np.zeros(BIG_T) + 0
        if Configuration.instance().dynamic_fix_term_pricing:
            p_0 = xdict["p_0"]
        else:
            p_0 = (
                np.zeros(BIG_T)
                + Configuration.instance().instance().price_parameters[0]
            )
        if smart_charging:
            # y = xdict['y'].reshape(len(vehicle_range), len(time_range))
            p_star = xdict["p_star"]
            # pv_usage = xdict['pv_usage']
        # for i in vehicle_range:
        #     for t in time_range:
        #         x[i, t] = ((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t]) / (2 * (beta[i] * T[i] + alpha[t]))
        # if x[i, t] != 0:
        #     print(np.sum(x))
        funcs = {}

        energy_charged = 0
        funcs["obj"] = 0

        for t in time_range:
            # if smart_charging:
            #     funcs["obj"] -= pv_usage[t] * electricity_cost[t]
            for i in vehicle_range:

                # power = x[i, t]/T[i]

                # funcs["obj"] -= (p_0[t] -0.15 + alpha[t] * power) * x[i, t]
                # -np.sum(np.array(electricity_cost)*y[i, :]))
                if not smart_charging:
                    if (
                        Configuration.instance().dynamic_fix_term_pricing
                        and not Configuration.instance().capacity_pricing
                    ):
                        funcs["obj"] -= (p_0[t] - 0.15) * (
                            (2 * beta[i] * D[i] - p_0[t]) * AB[i, t] / (2 * beta[i])
                        )
                        (2 * beta[i] * D[i] - p_0[t]) * AB[i, t] / (2 * beta[i])
                    else:
                        funcs["obj"] -= (
                            p_0[t]
                            - 0.15
                            + alpha[t]
                            * (
                                ((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t])
                                / (2 * (beta[i] * T[i] + alpha[t]))
                            )
                            / T[i]
                        ) * (
                            ((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t])
                            / (2 * (beta[i] * T[i] + alpha[t]))
                        )

                else:

                    funcs["obj"] -= (
                        p_0[t]
                        + alpha[t]
                        * (
                            ((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t])
                            / (2 * (beta[i] * T[i] + alpha[t]))
                        )
                        / T[i]
                    ) * (
                        ((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t])
                        / (2 * (beta[i] * T[i] + alpha[t]))
                    ) - electricity_cost[
                        t
                    ] * (
                        (
                            (
                                ((2 * beta[i] * D[i] - p_0[int(A[i])]) * T[i])
                                / (2 * (beta[i] * T[i] + alpha[int(A[i])]))
                            )
                            / T[i]
                            * U[i, t]
                        )
                    )

                energy_charged += ((2 * beta[i] * D[i] - p_0[t]) * T[i] * AB[i, t]) / (
                    2 * (beta[i] * T[i] + alpha[t])
                )
        if smart_charging:
            funcs["obj"] -= -p_star * peak_charge_penalty
        if not smart_charging:
            conval = [0] * BIG_T
            for t in time_range:
                conval[t] = alpha[t] + p_0[t]
            funcs["con"] = conval
        if smart_charging:
            # conval = [0] * BIG_T * len(vehicle_range)
            # for i in vehicle_range:
            #     for t in time_range:
            #         conval[i*BIG_T+t] = y[i,t]-U[i,t]*50
            # funcs["con1"] = conval
            # conval = [0] * len(vehicle_range)
            # for i in vehicle_range:
            #     conval[i] = np.sum(y[i,:]) - ((2 * beta[i] * D[i] - p_0[int(A[i])]) * T[i]) / (2 * (beta[i] * T[i] + alpha[int(A[i])]))
            # funcs["con2"] = conval
            conval = [0] * BIG_T
            for t in time_range:
                total_charging = 0
                for i in vehicle_range:
                    total_charging += (
                        ((2 * beta[i] * D[i] - p_0[int(A[i])]) * T[i])
                        * U[i, t]
                        / (2 * (beta[i] * T[i] + alpha[int(A[i])]))
                    ) / T[i]
                conval[t] = (
                    total_charging
                    + base_load[t]
                    - PV[t]
                    - p_star
                    - Configuration.instance().peak_threshold
                )
            funcs["con3"] = conval

            # conval = [0] * BIG_T
            # for t in time_range:
            #     total_charging = 0
            #     for i in vehicle_range:
            #         total_charging += (((2 * beta[i] * D[i] - p_0[int(A[i])])
            #                             * T[i]) * U[i, t] / (2 * (beta[i] * T[i] + alpha[int(A[i])]))) / T[i]
            #     conval[t] = total_charging + base_load[t] - pv_usage[t]
            # funcs["con4"] = conval
            #
            # conval = [0] * BIG_T
            # for t in time_range:
            #     conval[t] =  PV[t] - pv_usage[t]
            # funcs["con5"] = conval
        fail = False
        if smart_charging:
            print(energy_charged, funcs["obj"], p_star)

        return funcs, fail

    # rst begin optProb
    # Optimization Object
    optProb = Optimization("TP037 Constraint Problem", objfunc)

    # rst begin addVar
    # Design Variables
    optProb.addVarGroup(
        "alpha",
        BIG_T,
        "c",
        lower=np.zeros(BIG_T),
        upper=np.zeros(BIG_T) + 0.5,
        value=0.2,
    )
    if Configuration.instance().dynamic_fix_term_pricing:
        optProb.addVarGroup(
            "p_0",
            BIG_T,
            "c",
            lower=np.zeros(BIG_T),
            upper=np.zeros(BIG_T) + 1,
            value=0.3,
        )
    if smart_charging:
        # optProb.addVarGroup("y", BIG_T*len(vehicles_list), "c", lower=np.zeros(BIG_T*len(vehicles_list)),
        #                     upper=np.zeros(BIG_T*len(vehicles_list)) + 50, value=1)
        optProb.addVar(name="p_star", varType="c", lower=0, upper=1200, value=20)
        # optProb.addVarGroup("pv_usage", BIG_T, "c", lower=np.zeros(BIG_T), upper=np.zeros(BIG_T) + 1000, value=0)

    # rst begin addCon
    # Constraints
    if not smart_charging:
        optProb.addConGroup("con", BIG_T, lower=0)
    if smart_charging:
        # optProb.addConGroup("con1", BIG_T*len(vehicles_list), upper=0)
        # optProb.addConGroup("con2", len(vehicles_list), lower=0)
        optProb.addConGroup("con3", BIG_T, upper=0)
        # optProb.addConGroup("con4", BIG_T, lower=0)
        # optProb.addConGroup("con5", BIG_T, lower=0)

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
    print(sol.objectives["obj"])
    # print(sol)
    return sol.xStar


# print(ev_decision_making())
# import numpy as np
# from scipy.optimize import minimize, rosen
# def rosen_func(x):
#     return sum(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**(-1))
# x0 = np.array([1.3,0.7, 0.8, 1.9, 1.2])
# res = minimize(rosen, x0, method='SLSQP', options={'xatol':1e-8, 'disp': True,'full_output': True,})
# print(res)
