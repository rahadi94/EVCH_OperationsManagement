import random
from docplex.mp.model import Model
from datetime import datetime
import pandas as pd
from data import data_preparation

from Planning.log import lg
import numpy as np


def planning(facility, date, plug=4, adoption=1, service_level=0.5):
    start_time = datetime.now()
    random.seed(2)
    mdl = Model("Charging-cluster_Management")
    mdl.parameters.timelimit = 7200 * 100
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
    t_0 = 7
    t_end = 22
    dfs, loads, pv_profile = data_preparation(facility, date, t_0=t_0, size=200)
    Event = dfs["EntryHour"].count()
    year_range = range(10)
    df = {}
    vehicle_range = {}
    EV_ADOPTION = [
        5,
        5,
        8,
        8,
        12,
        12,
        18,
        18,
        27,
        27,
        37,
        37,
        42,
        42,
        49,
        49,
        56,
        56,
        65,
        65,
    ]
    PV_PRICES = [
        2125,
        2125,
        2041,
        2041,
        1960,
        1960,
        1882,
        1882,
        1808,
        1808,
        1736,
        1736,
        1668,
        1668,
        1601,
        1601,
        1538,
        1538,
        1477,
        1477,
    ]
    BATTERY_PRICES = [
        575,
        575,
        507,
        507,
        441,
        441,
        391,
        391,
        352,
        352,
        321,
        321,
        295,
        295,
        273,
        273,
        255,
        255,
        239,
        239,
    ]
    CHARGER_PRICES = {}
    PLUG_PRICES = {}
    PLUG_PRICES["slow"] = 250
    PLUG_PRICES["fast"] = 2500
    CHARGER_PRICES["slow_one"] = [
        4500,
        4500,
        4322,
        4322,
        4151,
        4151,
        3986,
        3986,
        3828,
        3828,
        3577,
        3677,
        3531,
        3531,
        3391,
        3391,
        3257,
        3257,
        3128,
        3128,
    ]
    CHARGER_PRICES["slow_two"] = [
        x + PLUG_PRICES["slow"] for x in CHARGER_PRICES["slow_one"]
    ]
    CHARGER_PRICES["slow_four"] = [
        x + PLUG_PRICES["slow"] * 3 for x in CHARGER_PRICES["slow_one"]
    ]
    CHARGER_PRICES["fast_one"] = [
        50000,
        50000,
        49000,
        49000,
        47060,
        47060,
        45196,
        45196,
        43406,
        43406,
        41687,
        41687,
        40037,
        40037,
        38451,
        38451,
        36928,
        36928,
        35466,
        35466,
    ]
    CHARGER_PRICES["fast_two"] = [
        x + PLUG_PRICES["fast"] for x in CHARGER_PRICES["fast_one"]
    ]
    CHARGER_PRICES["fast_four"] = [
        x + PLUG_PRICES["fast"] for x in CHARGER_PRICES["fast_one"]
    ]
    GRID_PRICES = [
        250,
        250,
        276,
        276,
        304,
        304,
        335,
        335,
        369,
        369,
        407,
        407,
        449,
        449,
        495,
        495,
        546,
        546,
        602,
        602,
    ]
    TR_PRICES = [x * 200 for x in GRID_PRICES]
    for s in year_range:
        df[s] = dfs.sample(
            n=int(
                Event * (EV_ADOPTION[s * 2] / 100 + EV_ADOPTION[s * 2 + 1] / 100) / 2
            ),
            random_state=42,
        )
        lg.error(f'Number of Events={df[s]["EntryHour"].count()}')
        vehicle_range[s] = range(df[s]["EntryHour"].count())
    space_range = range(40)
    connector_range = [1, 2, 4]
    time_range = range(t_0, t_end)
    vv_range = {}
    for s in year_range:
        for t in time_range:
            vv_range[s, t] = []

    # Parameters
    S = 500
    (
        C_plug,
        C_EVSE_fast,
        C_EVSE_standard,
        C_grid,
        C_grid_fixed,
        C_PV,
        C_Battery,
        P_EVSE_standard,
        P_EVSE_fast,
        P_grid,
        n_s,
        l_star,
        T_p,
        T_e,
        l,
    ) = ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    for s in year_range:
        C_EVSE_fast[s, 1] = (
            (CHARGER_PRICES["fast_one"][s * 2] + CHARGER_PRICES["fast_one"][s * 2 + 1])
            / 2
            / 365
            / 5
        )
        C_EVSE_fast[s, 2] = (
            (CHARGER_PRICES["fast_two"][s * 2] + CHARGER_PRICES["fast_two"][s * 2 + 1])
            / 2
            / 365
            / 5
        )
        C_EVSE_fast[s, 4] = (
            (
                CHARGER_PRICES["fast_four"][s * 2]
                + CHARGER_PRICES["fast_four"][s * 2 + 1]
            )
            / 2
            / 365
            / 5
        )
        C_EVSE_standard[s, 1] = (
            (CHARGER_PRICES["slow_one"][s * 2] + CHARGER_PRICES["slow_one"][s * 2 + 1])
            / 2
            / 365
            / 5
        )
        C_EVSE_standard[s, 2] = (
            (CHARGER_PRICES["slow_two"][s * 2] + CHARGER_PRICES["slow_two"][s * 2 + 1])
            / 2
            / 365
            / 5
        )
        C_EVSE_standard[s, 4] = (
            (
                CHARGER_PRICES["slow_four"][s * 2]
                + CHARGER_PRICES["slow_four"][s * 2 + 1]
            )
            / 2
            / 365
            / 5
        )
        C_grid[s] = 0.000  # (GRID_PRICES[s*2]+GRID_PRICES[s*2+1]) / 2 / 365 / 20
        C_grid_fixed[s] = (TR_PRICES[s * 2] + GRID_PRICES[s * 2 + 1]) / 2 / 365 / 20
        C_PV[s] = (PV_PRICES[s * 2] + PV_PRICES[s * 2 + 1]) / 2 / 365 / 20
        C_Battery[s] = (BATTERY_PRICES[s * 2] + BATTERY_PRICES[s * 2 + 1]) / 2 / 365 / 5
        P_EVSE_standard[s] = 22 * time
        P_EVSE_fast[s] = 50 * time
        P_grid[s] = loads.max() * 1.2

        n_s[s] = service_level
        l_star[s] = loads.max()
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
        # T = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
        #         , 8, 8, 8, 8, 8, 8, 8, 8]

        for t in time_range:
            T_e[s, t] = T[t] / 100
            l[s, t] = loads.iloc[t * 60 - 1]
    C_MAINTENANCE = 0.05
    e_d = {}
    A = {}
    U = {}
    D = {}
    for s in year_range:
        for j in vehicle_range[s]:
            A[j, s] = max(int(t_0 / time), int(df[s].iloc[j]["EntryHour"] / time))
            D[j, s] = max(int(A[j, s] + 1), int(df[s].iloc[j]["ExitHour"] / time))
            if D[j, s] >= t_end:
                D[j, s] = t_end - 1
            e_d[j, s] = min(
                float(df[s].iloc[j]["final_kWhRequested"]),
                (D[j, s] - A[j, s]) * 50 * time,
            )
            for t in time_range:
                if A[j, s] <= t <= D[j, s]:
                    U[j, s, t] = 1
                    vv_range[s, t].append(j)
                else:
                    U[j, s, t] = 0
    PV_rate = pv_profile
    gamma_max = 0.95
    gamma_min = 0.05
    battery_efficiency = 0.94

    # Variables
    x_fast = mdl.binary_var_cube(
        space_range, year_range, connector_range, name="x_fast"
    )
    x_standard = mdl.binary_var_cube(
        space_range, year_range, connector_range, name="x_standard"
    )
    h = mdl.continuous_var_dict(
        (k, j, s, t)
        for k in space_range
        for s in year_range
        for j in vehicle_range[s]
        for t in range(A[j, s], D[j, s] + 1)
    )
    w = mdl.binary_var_dict(
        (k, j, s) for k in space_range for s in year_range for j in vehicle_range[s]
    )
    p_plus = mdl.continuous_var_dict(year_range, lb=0, name="p_plus")
    p_plus_binary = mdl.integer_var_dict(year_range, name="p_plus_binary")
    p_star = mdl.continuous_var_dict(year_range, lb=0, name="p_star")
    pv = mdl.integer_var_dict(year_range, lb=0, name="pv")
    b_size = mdl.integer_var_dict(year_range, lb=0, name="b_size")
    b_direction = mdl.binary_var_matrix(year_range, time_range, name="b_direction")
    b_charge = mdl.continuous_var_matrix(
        year_range, time_range, lb=0, ub=50 * time, name="b_charge"
    )
    b_discharge = mdl.continuous_var_matrix(
        year_range, time_range, lb=0, ub=50 * time, name="b_discharge"
    )
    SoC = mdl.continuous_var_matrix(year_range, time_range, lb=0, name="SoC")
    # Constraints
    e = {}
    for s in year_range:
        for t in time_range:
            # TODO: check it
            e[s, t] = (
                mdl.sum(
                    h[k, j, s, t] * U[j, s, t]
                    for k in space_range
                    for j in vv_range[s, t]
                )
                + l[s, t]
                + (b_charge[s, t] - b_discharge[s, t]) * battery_efficiency
                - mdl.sum(pv[v] * PV_rate[t] for v in range(0, s + 1))
            )
            mdl.add_constraint(e[s, t] >= 0, "C16")
    mdl.add_constraint(
        mdl.sum(
            x_fast[k, s, l] * l + x_standard[k, s, l] * l
            for k in space_range
            for s in year_range
            for l in connector_range
        )
        <= S,
        "C1",
    )
    mdl.add_constraint(mdl.sum(pv[s] for s in year_range) <= 100, "C3")
    mdl.add_constraint(mdl.sum(b_size[s] for s in year_range) <= 500, "C3")
    # TODO: change it to a constraint again
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
        mdl.add_constraint(SoC[s, t_0] == 0)
        mdl.add_constraint(b_charge[s, t_0] <= b_direction[s, t_0] * 50 * time)
        mdl.add_constraint(b_discharge[s, t_0] <= (1 - b_direction[s, t_0]) * 50 * time)
        mdl.add_constraint(0 >= b_discharge[s, t_0])
        mdl.add_constraint(
            b_charge[s, t_0]
            <= mdl.sum(b_size[v] for v in range(0, s + 1)) * gamma_max - 0
        )
        for t in range(t_0 + 1, t_end):
            mdl.add_constraint(
                SoC[s, t] == SoC[s, t - 1] + b_charge[s, t - 1] - b_discharge[s, t - 1]
            )
            mdl.add_constraint(b_charge[s, t] <= b_direction[s, t] * 50 * time)
            mdl.add_constraint(b_discharge[s, t] <= (1 - b_direction[s, t]) * 50 * time)
            mdl.add_constraint(SoC[s, t - 1] >= b_discharge[s, t])
            mdl.add_constraint(
                b_charge[s, t]
                <= mdl.sum(b_size[v] for v in range(0, s + 1)) * gamma_max
                - SoC[s, t - 1]
            )

    for s in year_range:
        mdl.add_constraint(
            p_plus[s] <= 200 * mdl.sum(p_plus_binary[v] for v in range(0, s + 1))
        )
        for t in time_range:
            mdl.add_constraint(e[s, t] <= P_grid[s] + p_plus[s], "C6")

    for k in space_range:
        for t in time_range:
            for s in year_range:
                mdl.add_constraint(
                    mdl.sum(w[k, j, s] * U[j, s, t] for j in vv_range[s, t])
                    <= mdl.sum(
                        x_fast[k, v, l] * l + x_standard[k, v, l] * l
                        for l in connector_range
                        for v in range(0, s + 1)
                    ),
                    "C10",
                )
    for s in year_range:
        for j in vehicle_range[s]:
            mdl.add_constraint(mdl.sum(w[k, j, s] for k in space_range) <= 1, "C11")

    for k in space_range:
        mdl.add_constraint(
            mdl.sum(
                x_standard[k, s, l] + x_fast[k, s, l]
                for l in connector_range
                for s in year_range
            )
            <= 1,
            "C15",
        )
        for s in year_range:
            for t in time_range:
                for j in vv_range[s, t]:
                    mdl.add_constraint(
                        h[k, j, s, t] <= w[k, j, s] * U[j, s, t] * 50, "C15"
                    )
                    mdl.add_constraint(
                        mdl.sum(h[k, j, s, t] for k in space_range) >= 0, "C16"
                    )

    for k in space_range:
        for t in time_range:
            for s in year_range:
                mdl.add_constraint(
                    mdl.sum(h[k, j, s, t] * U[j, s, t] for j in vv_range[s, t])
                    <= mdl.sum(
                        x_fast[k, v, l] * P_EVSE_fast[s]
                        for l in connector_range
                        for v in range(0, s + 1)
                    )
                    + mdl.sum(
                        x_standard[k, v, l] * P_EVSE_standard[s]
                        for l in connector_range
                        for v in range(0, s + 1)
                    ),
                    "C16",
                )
    # for s in year_range:
    #     for j in vehicle_range[s]:
    #         for t in range(A[j,s], D[j,s]+1):
    #             #TODO: check if h gets a negative value or not

    for t in time_range:
        for s in year_range:
            mdl.add_constraint(e[s, t] - l_star[s] <= p_star[s], "C17")
    c1 = {}
    for s in year_range:
        c1[s] = (
            mdl.sum(
                C_EVSE_standard[s, l] * x_standard[k, s, l]
                + C_EVSE_fast[s, l] * x_fast[k, s, l]
                for l in connector_range
                for k in space_range
            )
            + C_grid[s] * p_plus[s]
            + C_grid_fixed[s] * p_plus_binary[s]
            + C_PV[s] * pv[s]
            + C_Battery[s] * b_size[s]
        )
    c1_total = mdl.sum(
        c1[s] * (1 + C_MAINTENANCE * (year_range[-1] - s)) for s in year_range
    )
    c2 = mdl.sum(
        (T_e[s, t]) * e[s, t] + T_p[s] * p_star[s]
        for t in time_range
        for s in year_range
    )
    total_service_quality = (
        0  # mdl.sum(QoS[s,j] for s in year_range for j in vehicle_range[s]) * 1000000
    )
    mdl.minimize(c1_total + c2 + total_service_quality)
    lg.error("Start_time: {}".format(start_time))
    mdl.print_information()
    """import warnings
    warnings.filterwarnings("error")"""

    # try:
    assert mdl.solve(), "!!! Solve of the model fails"
    end_time = datetime.now()
    lg.error("Duration: {}".format(end_time - start_time))
    lg.error("c1: {}".format(c1_total.solution_value))
    lg.error("c2: {}".format(c2.solution_value))
    NoC_standard, NoC_fast, NoP = {}, {}, {}
    for s in year_range:
        for t in time_range:
            if SoC[s, t].solution_value != 0:
                lg.error(f"SoC{s, t}: {SoC[s, t].solution_value}")
        for l in connector_range:
            NoC_standard[s, l] = mdl.sum(x_standard[k, s, l] for k in space_range)
            NoC_fast[s, l] = mdl.sum(x_fast[k, s, l] for k in space_range)
            lg.error(f"NoC_standard_{s, l}: {NoC_standard[s, l].solution_value}")
            lg.error(f"NoC_fast_{s, l}: {NoC_fast[s, l].solution_value}")
        lg.error(f"battery_installed_{s}: {b_size[s].solution_value}")
        lg.error(f"pv_installed_{s}: {pv[s].solution_value}")
        lg.error(f"p_plus_{s}: {p_plus[s].solution_value}")
        lg.error(f"NoT_{s}: {p_plus_binary[s].solution_value}")
        lg.error(f"p_star_{s}: {p_star[s].solution_value}")
        # for t in time_range:
        #     if b_charge[s, t].solution_value != 0:
        #         lg.error(f'b_charge_{s, t} = {b_charge[s, t].solution_value}')
        #     if b_discharge[s, t].solution_value != 0:
        #         lg.error(f'b_discharge_{s, t} = {b_discharge[s, t].solution_value}')
        #     lg.error(f'SoC_{s, t} = {SoC[s, t].solution_value}')
    mdl.report()
    # lg.error("it did not solved")
    lg.error(f"facility = {facility}, date = {date}")
    lg.error(f"objective_function = {mdl.objective_value}")

    investment_index = pd.MultiIndex.from_product([year_range], names=["year"])
    investment_results = pd.DataFrame(
        -np.random.rand(len(investment_index), 10), index=investment_index
    )
    investment_results.columns = [
        "NoC_standard_one",
        "NoC_standard_two",
        "NoC_standard_four",
        "NoC_fast_one",
        "NoC_fast_two",
        "NoC_fast_four",
        "NoT",
        "p_plus",
        "battery",
        "PV",
    ]
    for s in year_range:
        investment_results.loc[(s), "NoC_standard_one"] = NoC_standard[
            s, 1
        ].solution_value
        investment_results.loc[(s), "NoC_standard_two"] = NoC_standard[
            s, 2
        ].solution_value
        investment_results.loc[(s), "NoC_standard_four"] = NoC_standard[
            s, 4
        ].solution_value
        investment_results.loc[(s), "NoC_fast_one"] = NoC_fast[s, 1].solution_value
        investment_results.loc[(s), "NoC_fast_two"] = NoC_fast[s, 2].solution_value
        investment_results.loc[(s), "NoC_fast_four"] = NoC_fast[s, 4].solution_value
        investment_results.loc[(s), "NoT"] = p_plus_binary[s].solution_value
        investment_results.loc[(s), "p_plus"] = p_plus[s].solution_value
        investment_results.loc[(s), "battery"] = b_size[s].solution_value
        investment_results.loc[(s), "PV"] = pv[s].solution_value
    investment_results.to_csv(f"investment_results_{facility}_{date}_TR_PRICES.csv")

    # results = pd.read_csv('results.csv')

    """results.loc[(results['Facility'] == int(facility[-1])) & (
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

    # except:
    #     lg.error('it did not solve')
    #     lg.error(f'facility = {facility}, date = {date}')
    #     end_time = datetime.now()
    #     lg.error('Duration: {}'.format(end_time - start_time))
    #     return


# dates = ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07',
#          '2019-10-21', '2019-10-22', '2019-10-23', '2019-10-24', '2019-10-25']
# dates = ['2019-06-03', '2019-06-04', '2019-06-05', '2019-06-06', '2019-06-07']
dates = ["2019-06-03"]
for i in dates:
    planning(facility="Facility_KoeBogen", date=i, plug=4, adoption=1, service_level=1)
