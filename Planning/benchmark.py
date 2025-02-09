import random
from docplex.mp.model import Model
from datetime import datetime
import pandas as pd

from data import data_preparation
from log import lg
import numpy as np


def planning(facility, date, adoption):
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
    # mdl.parameters.mip.tolerances.mipgap = 0.05
    # mdl.parameters.mip.limits.nodes = 100
    # Sets
    time = 1
    dfs, loads = data_preparation(facility, date)
    charging_stations = int(250 * adoption)
    Event = dfs["EntryHour"].count()
    dfs = dfs.sample(n=int(Event * adoption))
    occupation = pd.DataFrame(range(24))
    for i in range(24):
        occupation.iloc[i][0] = dfs[(dfs["EntryHour"] <= i) & (dfs["ExitHour"] >= i)][
            "EntryHour"
        ].count()
    lg.error(f"peak = {occupation[0].max()}")
    space_range = range(charging_stations)
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
    w = mdl.binary_var_dict(vehicle_range, name="w")
    ll = mdl.continuous_var_dict(vehicle_range, lb=0, name="ll")
    e = mdl.continuous_var_matrix(vehicle_range, time_range, lb=0, name="e")
    p_plus = mdl.continuous_var(lb=0, name="p_plus")
    p_star = mdl.continuous_var(lb=0, name="p_star")

    # Constraints

    for j in vehicle_range:
        mdl.add_constraint(
            e_d[j] - mdl.sum(e[j, t] for t in range(A[j], D[j] + 1)) <= ll[j], "C4"
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
            mdl.sum(w[j] * U[j, t] for j in vehicle_range) <= charging_stations, "C10"
        )
    """for j in vehicle_range:
        mdl.add_constraint(mdl.sum(w[k, j] for k in space_range) <= 1, 'C11')"""

    for j in vehicle_range:
        for t in range(A[j], D[j] + 1):
            mdl.add_constraint(e[j, t] <= w[j] * P_EVSE, "C15")

    for t in time_range:
        mdl.add_constraint(
            mdl.sum(e[j, t] for j in vehicle_range) + l[t] - l_star <= p_star, "C17"
        )

    c1 = C_grid * p_plus + charging_stations * C_EVSE
    c2 = (
        mdl.sum((T_e[t]) * e[j, t] for j in vehicle_range for t in time_range)
        + T_p * p_star
    )
    loss = mdl.sum(ll[j] for j in vehicle_range)
    mdl.minimize(c1 + c2 + loss)
    lg.error("Start_time: {}".format(start_time))
    mdl.print_information()
    assert mdl.solve(), "!!! Solve of the model fails"
    end_time = datetime.now()
    lg.error("Duration: {}".format(end_time - start_time))
    lg.error("c1: {}".format(c1.solution_value))
    lg.error("c2: {}".format(c2.solution_value))
    lg.error("loss: {}".format(loss.solution_value))
    lg.error("loss: {}".format(mdl.sum(w[j] for j in vehicle_range).solution_value))

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
    results = pd.read_csv("results_benchmark.csv")

    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date),
        "Installation_costs",
    ] = c1.solution_value
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date),
        "Operations_costs",
    ] = c2.solution_value
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date), "NoC"
    ] = charging_stations
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date), "NoP"
    ] = charging_stations
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date), "P_plus"
    ] = p_plus.solution_value
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date), "P_star"
    ] = p_star.solution_value
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date), "costs"
    ] = mdl.objective_value
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date), "Event"
    ] = dfs["EntryHour"].count()
    results.loc[
        (results["Adoption"] == adoption) & (results["Date"] == date), "Loss"
    ] = loss.solution_value
    results.to_csv("results_benchmark.csv")


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
    planning(facility="Facility_3", date=i, adoption=0.10)
