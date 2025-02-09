import random
from docplex.mp.model import Model
from datetime import datetime
import pandas as pd
from data import data_preparation
from log import lg
import numpy as np


def planning(facility, date, plug=4, adoption=1, service_level=1):
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
    mdl.parameters.mip.tolerances.mipgap = 0.01
    # mdl.parameters.mip.limits.nodes = 100

    # Sets
    time = 1
    dfs, loads, pv_profile = data_preparation(facility, date)
    Event = dfs["EntryHour"].count()
    year_range = range(4)
    df = {}
    vehicle_range = {}
    for s in year_range:
        df[s] = dfs.sample(n=int(Event * adoption * 0.25 * (s + 1)))
        lg.error(f'Number of Events={df[s]["EntryHour"].count()}')
        vehicle_range[s] = range(df[s]["EntryHour"].count())
    space_range = range(50)
    connector_range = range(plug)
    time_range = range(5, 23)

    # Parameters
    S = 200
    N = 6
    (
        C_plug,
        C_EVSE,
        C_grid,
        C_PV,
        C_Battery,
        P_EVSE,
        P_grid,
        n_s,
        l_star,
        T_p,
        T_e,
        l,
    ) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    for s in year_range:
        C_plug[s] = 250 / 365 / 5
        C_EVSE[s] = 4500 / 365 / 5
        C_grid[s] = 240 / 365 / 10
        C_PV[s] = 2000 / 365 / 10
        C_Battery[s] = 20 / 365 / 5
        P_EVSE[s] = 22 * time
        P_grid[s] = loads.max().values[0] * 1.5
        n_s[s] = service_level
        l_star[s] = loads.max().values[0]
        T_p[s] = 15.48 / 30
        if date in [
            "2019-06-03",
            "2019-06-04",
            "2019-06-05",
            "2019-06-06",
            "2019-06-07",
        ]:
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
        for t in time_range:
            T_e[s, t] = T[t] / 100
            # T_e[t] = 0.25
            # l[t] = loads.iloc[t, 0]
            l[s, t] = loads.iloc[t, 0]
    c = {}
    for k in space_range:
        c[k] = 1 + (k / 10000)
    cc = {}
    for i in connector_range:
        cc[i] = 1 + (i / 10000)

    e_d = {}
    A = {}
    U = {}
    D = {}
    for s in year_range:
        for j in vehicle_range[s]:
            A[j, s] = int(dfs.iloc[j]["EntryHour"] / time)
            D[j, s] = int(dfs.iloc[j]["ExitHour"] / time)
            e_d[j, s] = float(dfs.iloc[j]["final_kWhRequested"])
            for t in time_range:
                if A[j, s] <= t <= D[j, s]:
                    U[j, s, t] = 1
                else:
                    U[j, s, t] = 0
    PV_rate = pv_profile

    # Variables
    x = mdl.binary_var_matrix(space_range, year_range, name="x")
    y = mdl.integer_var_matrix(space_range, year_range, lb=0, ub=plug, name="y")
    h = mdl.continuous_var_dict(
        (k, j, s, t)
        for k in space_range
        for s in year_range
        for j in vehicle_range[s]
        for t in time_range
    )
    w = mdl.continuous_var_dict(
        (k, j, s) for k in space_range for s in year_range for j in vehicle_range[s]
    )
    p_plus = mdl.continuous_var_dict(year_range, lb=0, name="p_plus")
    p_star = mdl.continuous_var_dict(year_range, lb=0, name="p_star")
    pv = mdl.integer_var_dict(year_range, lb=0, name="pv")
    b_size = mdl.integer_var_dict(year_range, lb=0, name="b_size")
    b_direction = mdl.binary_var_matrix(year_range, time_range, name="b_direction")
    b_charge = mdl.continuous_var_matrix(
        year_range, time_range, lb=0, ub=20, name="b_charge"
    )
    b_discharge = mdl.continuous_var_matrix(
        year_range, time_range, lb=0, ub=20, name="b_discharge"
    )
    SoC = mdl.continuous_var_matrix(year_range, time_range, lb=0, name="SoC")
    # Constraints
    e = {}
    for s in year_range:
        for t in time_range:
            e[s, t] = (
                mdl.sum(
                    h[k, j, s, t]
                    for k in space_range
                    for j in vehicle_range[s]
                    for t in time_range
                )
                + b_charge[s, t]
                - b_discharge[s, t]
                - mdl.sum(pv[v] * PV_rate[t] for v in range(0, s + 1))
            )
    mdl.add_constraint(
        mdl.sum(x[k, s] for k in space_range for s in year_range) <= S, "C1"
    )
    for k in space_range:
        for s in year_range:
            mdl.add_constraint(
                mdl.sum(y[k, v] for v in range(0, s + 1))
                <= N * mdl.sum(x[k, v] for v in range(0, s + 1)),
                "C3",
            )
    mdl.add_constraint(mdl.sum(pv[s] for s in year_range) <= 500 * 0.1666, "C3")
    mdl.add_constraint(mdl.sum(b_size[s] for s in year_range) <= 200, "C3")
    for s in year_range:
        for j in vehicle_range[s]:
            mdl.add_constraint(
                mdl.sum(
                    h[k, j, s, t]
                    for k in space_range
                    for t in range(A[j, s], D[j, s] + 1)
                )
                >= n_s[s] * e_d[j, s],
                "C4",
            )

    for s in year_range:
        mdl.add_constraint(SoC[s, 5] <= 0)
        mdl.add_constraint(b_charge[s, 5] <= b_direction[s, 5] * 50)
        mdl.add_constraint(b_discharge[s, 5] <= (1 - b_direction[s, 5]) * 50)
        mdl.add_constraint(0 >= b_discharge[s, 5])
        mdl.add_constraint(
            b_charge[s, t] <= mdl.sum(b_size[v] for v in range(0, s + 1)) - 0
        )
        for t in range(6, 23):
            mdl.add_constraint(
                SoC[s, t] == SoC[s, t - 1] + b_charge[s, t] - b_discharge[s, t]
            )
            mdl.add_constraint(b_charge[s, t] <= b_direction[s, t] * 50)
            mdl.add_constraint(b_discharge[s, t] <= (1 - b_direction[s, t]) * 50)
            mdl.add_constraint(SoC[s, t - 1] >= b_discharge[s, t])
            mdl.add_constraint(
                b_charge[s, t]
                <= mdl.sum(b_size[v] for v in range(0, s + 1)) - SoC[s, t - 1]
            )

    for t in time_range:
        for s in year_range:
            mdl.add_constraint(
                mdl.sum(h[k, j, s, t] for k in space_range for j in vehicle_range[s])
                + l[s, t]
                <= P_grid[s]
                + mdl.sum(p_plus[s] + pv[v] * PV_rate[t] for v in range(0, s + 1))
                - b_charge[s, t]
                + b_discharge[s, t],
                "C6",
            )

    for k in space_range:
        for t in time_range:
            for s in year_range:
                mdl.add_constraint(
                    mdl.sum(w[k, j, s] * U[j, s, t] for j in vehicle_range[s])
                    <= mdl.sum(y[k, v] for v in range(0, s + 1)),
                    "C10",
                )
    for s in year_range:
        for j in vehicle_range[s]:
            mdl.add_constraint(mdl.sum(w[k, j, s] for k in space_range) <= 1, "C11")

    for k in space_range:
        for s in year_range:
            for j in vehicle_range[s]:
                for t in range(A[j, s], D[j, s] + 1):
                    mdl.add_constraint(h[k, j, s, t] <= w[k, j, s] * P_EVSE[s], "C15")

    for k in space_range:
        for t in time_range:
            for s in year_range:
                mdl.add_constraint(
                    mdl.sum(h[k, j, s, t] for j in vehicle_range[s]) <= P_EVSE[s], "C16"
                )

    for t in time_range:
        for s in year_range:
            mdl.add_constraint(
                mdl.sum(h[k, j, s, t] for k in space_range for j in vehicle_range[s])
                + l[s, t]
                - mdl.sum(pv[v] for v in range(0, s + 1)) * PV_rate[t]
                + b_charge[s, t]
                - l_star[s]
                - b_discharge[s, t]
                <= p_star[s],
                "C17",
            )

    c1 = mdl.sum(
        C_EVSE[s] * c[k] * x[k, s]
        + C_plug[s] * y[k, s]
        + C_grid[s] * p_plus[s]
        + C_PV[s] * pv[s]
        + C_Battery[s] * b_size[s]
        for k in space_range
        for s in year_range
    )
    c2 = mdl.sum(
        (T_e[s, t]) * e[s, t] + T_p[s] * p_star[s]
        for t in time_range
        for s in year_range
    )
    mdl.minimize(c1 + c2)
    lg.error("Start_time: {}".format(start_time))
    mdl.print_information()
    """import warnings
    warnings.filterwarnings("error")"""

    try:
        assert mdl.solve(), "!!! Solve of the model fails"
        end_time = datetime.now()
        lg.error("Duration: {}".format(end_time - start_time))
        lg.error("c1: {}".format(c1.solution_value))
        lg.error("c2: {}".format(c2.solution_value))
        NoC, NoP = {}, {}
        for s in year_range:
            NoC[s] = mdl.sum(x[k, s] for k in space_range)
            NoP[s] = mdl.sum(y[k, s] for k in space_range)
            lg.error(f"NoC_{s}: {NoC[s].solution_value}")
            lg.error(f"NoP_{s}: {NoP[s].solution_value}")
            lg.error(f"battery_installed_{s}: {b_size[s].solution_value}")
            lg.error(f"pv_installed_{s}: {pv[s].solution_value}")
            lg.error(f"p_plus_{s}: {p_plus[s].solution_value}")
            lg.error(f"p_star_{s}: {p_star[s].solution_value}")
            for t in time_range:
                if b_charge[s, t].solution_value != 0:
                    lg.error(f"b_charge_{s,t} = {b_charge[s,t].solution_value}")
                if b_discharge[s, t].solution_value != 0:
                    lg.error(f"b_discharge_{s,t} = {b_discharge[s,t].solution_value}")
                if SoC[s, t].solution_value != 0:
                    lg.error(f"SoC_{s,t} = {SoC[s,t].solution_value}")
        mdl.report()
        # lg.error("it did not solved")
        lg.error(f"facility = {facility}, date = {date}")
        lg.error(f"objective_function = {mdl.objective_value}")

        """for k in space_range:
            if x[k].solution_value != 0:
                lg.error(f'x_{k} = {x[k].solution_value}')

            for k in space_range:
                if y[ k].solution_value != 0:
                    lg.error(f'y_{k} = {y [k].solution_value}')
    
        for j in vehicle_range:
            for k in space_range:
                for t in time_range:
                    if h[k, j, t].solution_value != 0:
                        lg.error(f'w_{k, j} = {w[k, j].solution_value}, '
                                 f'h_{k, j, t} = {h[k, j, t].solution_value}, '
                                 f'A_{j} = {A[j]}, D_{j} = {D[j]}, '
                                 f'e_{j} = {e_d[j]}')"""
        """for k in time_range:
            if b_charge[k].solution_value != 0:
                lg.error(f'b_charge_{k} = {b_charge[k].solution_value}')
            if b_discharge[k].solution_value != 0:
                lg.error(f'b_discharge_{k} = {b_discharge[k].solution_value}')
            if SoC[k].solution_value != 0:
                lg.error(f'SoC_{k} = {SoC[k].solution_value}')"""

        """v_index = pd.MultiIndex.from_product([space_range, vehicle_range, range(24)],
                                             names=['charger', 'vehicle', 'time'])
        v_results = pd.DataFrame(-np.random.rand(len(v_index), 5), index=v_index)
        v_results.columns = ['Energy', 'Connection', 'Arrival', 'Departure', 'Occupation']
        for j in vehicle_range:
            for k in space_range:
                for t in time_range:
                    v_results.loc[(k, j, t), 'Connection'] = w[k, j].solution_value
                    v_results.loc[(k, j, t), 'Energy'] = h[k, j, t].solution_value
                    v_results.loc[(k, j, t), 'Arrival'] = A[j]
                    v_results.loc[(k, j, t), 'Departure'] = D[j]
                    v_results.loc[(k, j, t), 'Occupation'] = w[k, j].solution_value * U[j, t]
        v_results.to_csv(f'vehicles_{facility}_{date}.csv')
        CS_index = pd.MultiIndex.from_product([space_range, connector_range],
                                              names=['charger', 'plug'])
        CS_results = pd.DataFrame(-np.random.rand(len(CS_index), 2), index=CS_index)
        CS_results.columns = ['CS', 'Connector']
        for k in space_range:
            CS_results.loc[(k, i), 'CS'] = x[k].solution_value
            CS_results.loc[(k, i), 'Connector'] = y[k].solution_value
        CS_results.to_csv(f'CS_{facility}_{date}.csv')

        results = pd.read_csv('results.csv')

        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'Installation_costs'] = c1.solution_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'Operations_costs'] = c2.solution_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'NoC'] = NoC.solution_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'NoP'] = NoP.solution_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'P_plus'] = p_plus.solution_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'P_star'] = p_star.solution_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'costs'] = mdl.objective_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                    results['Date'] == date), 'Event'] = dfs["EntryHour"].count()
        results.loc[(results['Facility'] == int(facility[-1])) & (
                results['Date'] == date), 'PV'] = pv.solution_value
        results.loc[(results['Facility'] == int(facility[-1])) & (
                results['Date'] == date), 'Battery'] = b_size.solution_value
        results.to_csv('results.csv')"""

    except:
        lg.error("it did not solve")
        lg.error(f"facility = {facility}, date = {date}")
        end_time = datetime.now()
        lg.error("Duration: {}".format(end_time - start_time))
        return


# dates = ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07',
#          '2019-10-21', '2019-10-22', '2019-10-23', '2019-10-24', '2019-10-25']
# dates = ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07']
dates = ["2019-06-03"]
for i in dates:
    planning(facility="Facility_1", date=i, plug=4, adoption=1, service_level=1)
