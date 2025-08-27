import random
from docplex.mp.model import Model
from datetime import datetime
import pandas as pd

from Resource.data_facility_4 import data_preparation
from log import lg
import numpy as np


def planning(facility, date):
    start_time = datetime.now()
    random.seed(2)
    mdl = Model("Charging-cluster_Management")
    mdl.parameters.timelimit = 7200 * 2
    # mdl.parameters.threads = 32
    # mdl.parameters.simplex.tolerances.optimality = 1e-1
    # mdl.parameters.feasopt.tolerance = 1e-1
    # mdl.parameters.mip.strategy.lbheur = 1
    # mdl.parameters.optimalitytarget = 2
    # mdl.parameters.benders.strategy = 3
    # mdl.parameters.mip.strategy.bbinterval = 1
    # mdl.parameters.mip.tolerances.absmipgap = 0.2
    mdl.parameters.mip.tolerances.mipgap = 0.05
    # mdl.parameters.mip.limits.nodes = 100
    # Sets
    time = 1
    dfs, loads = data_preparation(facility, date)
    space_range = range(300)
    connector_range = range(1)
    vehicle_range = range(dfs["EntryHour"].count())
    lg.error(f'Number of Events={dfs["EntryHour"].count()}')
    time_range = range(5, 23)
    # Parameters
    S = 2000
    N = 6
    C_plug = 250 / 365 * 0.5
    C_EVSE = 4500 / 365 * 0.5
    C_grid = 240 / 365 * 0.5
    P_EVSE = 22 * time
    P_grid = loads.max().values[0] * 1.5
    n_s = 1
    l_star = loads.max().values[0]
    T_p = 15.48 / 30
    T_e = {}
    if date in ["2019-06-03", "2019-06-04", "2019-06-05", "2019-06-06", "2019-06-07"]:
        T = [
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            23,
            23,
            23,
            23,
            23,
            23,
            8,
            8,
        ]
    else:
        """T = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
        , 23, 23, 23, 23, 23, 23, 8, 8]"""
        T = [
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            8,
            5.5,
            5.5,
            5.5,
            5.5,
            5.5,
            5.5,
            5.5,
            23,
            23,
            23,
            23,
            23,
            23,
            8,
            8,
        ]

    l = {}
    for t in time_range:
        T_e[t] = T[t] / 100
        # T_e[t] = 0.25
        # l[t] = loads.iloc[t, 0]
        l[t] = loads.iloc[t, 0]
    c = {}
    for k in space_range:
        c[k] = 1 + (k / 10000)
    cc = {}
    for i in connector_range:
        cc[i] = 1 + (i / 10000)
    price = {}

    for j in vehicle_range:
        for t in time_range:
            price[j, t] = 1  # + ((j + t) / 10000)

    e_d = {}
    A = {}
    U = {}
    D = {}
    for j in vehicle_range:
        A[j] = int(dfs.iloc[j]["EntryHour"] / time)
        D[j] = int(dfs.iloc[j]["ExitHour"] / time)
        e_d[j] = float(dfs.iloc[j]["final_kWhRequested"])
        for t in time_range:
            if A[j] <= t <= D[j]:
                U[j, t] = 1
            else:
                U[j, t] = 0

    # Variables
    x = mdl.binary_var_dict(space_range, name="x")
    # h = mdl.continuous_var_cube(space_range, vehicle_range, time_range, lb=0, name=f'h{j}')
    w = mdl.binary_var_matrix(space_range, vehicle_range, name="w")
    e = mdl.continuous_var_matrix(vehicle_range, time_range, lb=0, name="e")
    p_plus = mdl.continuous_var(lb=0, name="p_plus")
    p_star = mdl.continuous_var(lb=0, name="p_star")

    # Constraints

    mdl.add_constraint(mdl.sum(x[k] for k in space_range) <= S, "C1")

    for j in vehicle_range:
        mdl.add_constraint(
            mdl.sum(e[j, t] for t in range(A[j], D[j] + 1)) >= n_s * e_d[j], "C4"
        )
    """mdl.add_constraint(mdl.sum(e[j, t] for j in vehicle_range for t in range(A[j], D[j] + 1))
                       >= mdl.sum(e_d[j] for j in vehicle_range), 'C5')
    for j in vehicle_range:
        mdl.add_constraint(mdl.sum(e[j, t] for t in range(A[j], D[j] + 1)) <= e_d[j], 'C4')"""
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(e[j, t] for j in vehicle_range) + l[t] <= P_grid + p_plus, "C6"
        )

    for t in time_range:
        mdl.add_constraint(
            mdl.sum(w[k, j] * U[j, t] for k in space_range for j in vehicle_range)
            <= mdl.sum(x[k] for k in space_range),
            "C10",
        )
    for j in vehicle_range:
        mdl.add_constraint(mdl.sum(w[k, j] for k in space_range) <= 1, "C11")

    for j in vehicle_range:
        for t in range(A[j], D[j] + 1):
            mdl.add_constraint(
                e[j, t] <= mdl.sum(w[k, j] for k in space_range) * P_EVSE, "C15"
            )

    for t in time_range:
        mdl.add_constraint(
            mdl.sum(e[j, t] for j in vehicle_range) + l[t] - l_star <= p_star, "C17"
        )

    c1 = mdl.sum(C_EVSE * c[k] * x[k] for k in space_range) + C_grid * p_plus
    c2 = (
        mdl.sum((T_e[t]) * e[j, t] for j in vehicle_range for t in time_range)
        + T_p * p_star
    )
    mdl.minimize(c1 + c2)
    lg.error("Start_time: {}".format(start_time))
    mdl.print_information()

    assert mdl.solve(), "!!! Solve of the model fails"
    end_time = datetime.now()
    lg.error("Duration: {}".format(end_time - start_time))
    NoC = mdl.sum(x[k] for k in space_range)

    lg.error("c1: {}".format(c1.solution_value))
    lg.error("c2: {}".format(c2.solution_value))
    lg.error("NoC: {}".format(NoC.solution_value))

    # lg.error("it did not solved")
    mdl.report()
    lg.error(f"facility = {facility}, date = {date}")
    lg.error(f"objective_function = {mdl.objective_value}")
    lg.error(f"p_plus = {p_plus.solution_value}")
    lg.error(f"p_star = {p_star.solution_value}")

    v_index = pd.MultiIndex.from_product(
        [vehicle_range, time_range], names=["vehicle", "time"]
    )
    v_results = pd.DataFrame(-np.random.rand(len(v_index), 4), index=v_index)
    v_results.columns = ["Energy", "Arrival", "Departure", "Occupation"]
    for j in vehicle_range:
        for t in time_range:
            # v_results.loc[(k, j, t), 'Connection'] = w[k, j].solution_value
            v_results.loc[(j, t), "Energy"] = e[j, t].solution_value
            v_results.loc[(j, t), "Arrival"] = A[j]
            v_results.loc[(j, t), "Departure"] = D[j]
            v_results.loc[(j, t), "Occupation"] = U[j, t]
    v_results.to_csv(f"vehicles_{facility}_{date}.csv")
    end_time = datetime.now()
    lg.error("Duration: {}".format(end_time - start_time))
    results = pd.read_csv("results_plugs.csv")

    results.loc[
        (results["Plugs"] == 1) & (results["Date"] == date), "Installation_costs"
    ] = c1.solution_value
    results.loc[
        (results["Plugs"] == 1) & (results["Date"] == date), "Operations_costs"
    ] = c2.solution_value
    results.loc[(results["Plugs"] == 1) & (results["Date"] == date), "NoC"] = (
        NoC.solution_value
    )
    results.loc[(results["Plugs"] == 1) & (results["Date"] == date), "NoP"] = (
        NoC.solution_value
    )
    results.loc[(results["Plugs"] == 1) & (results["Date"] == date), "P_plus"] = (
        p_plus.solution_value
    )
    results.loc[(results["Plugs"] == 1) & (results["Date"] == date), "P_star"] = (
        p_star.solution_value
    )
    results.loc[(results["Plugs"] == 1) & (results["Date"] == date), "costs"] = (
        mdl.objective_value
    )
    results.loc[(results["Plugs"] == 1) & (results["Date"] == date), "Event"] = dfs[
        "EntryHour"
    ].count()
    results.to_csv("results_plugs.csv")


dates = [
    "2019-06-03",
    "2019-06-04",
    "2019-06-05",
    "2019-06-06",
    "2019-06-07",
    "2019-10-21",
    "2019-10-22",
    "2019-10-23",
    "2019-10-24",
    "2019-10-25",
]


for i in dates:
    planning(facility="Facility_3", date=i)
