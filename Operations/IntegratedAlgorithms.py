from Environment.log import lg
from docplex.mp.model import Model
import numpy as np
from Environment.helper.configuration.configuration import Configuration


# TODO: myopic and multi-period optimal routing and charging (as implemented in ICIS paper but with specified lookahead window (e.g., 8 planning periods, which is 2h))


def perfect_info_charging_routing(
    vehicles,
    charging_stations,
    env,
    grid_capacity,
    electricity_cost,
    sim_time,
    baseload,
    generation=None,
    service_level=1,
    time_range=24,
):
    """
    It scales the charging power of all vehicles using multi-period optimization with perfect information
    :param service_level:
    :param sim_time: end of simulation
    :param electricity_cost: Time of use electricity tarrif
    :param grid_capacity:
    :param env: Simpy environment
    :param charging_stations: all charging_stations
    :param vehicles: all charging vehicles
    Objective function: maximize the charging power for low flexibility
    :return: charging power and routing for all vehicles
    """
    mdl = Model("perfect_info_scaling")
    vehicle_range = []
    delta_time = 60
    sim_duration = 5

    for i in vehicles:
        vehicle_range.append(i.id)
    CS_range = []
    sub_vehicles_charging = {}
    sub_vehicles = {}
    for j in charging_stations:
        CS_range.append(j.id)
        j.charging_vehicles = [x for x in j.connected_vehicles if x.mode == "Connected"]
        sub_vehicles_charging[j] = j.charging_vehicles
        sub_vehicles = j.connected_vehicles
    number_of_connectors = j.number_of_connectors

    time_range = range(
        max(round((sim_time - env.now) / delta_time), time_range * sim_duration)
    )
    l_star = Configuration.instance().peak_threshold
    # l_predicted = list(np.random.normal(100, 20, len(time_range)+1))
    c_p = Configuration.instance().peak_cost
    D = {}  # Energy demand
    De = {}  # Departure time
    A = {}
    c = {}
    for t in time_range:
        hour = int((t % 24))
        c[t] = electricity_cost[hour]
    U = {}
    for i in vehicles:
        D[i.id] = i.energy_requested
        # SOC[i.id] = i.energy_charged
        A[i.id] = int(min(i.arrival_period, sim_time) / delta_time)

        De[i.id] = int(min(i.departure_period, sim_time) / delta_time) - 1

        for t in time_range:
            if A[i.id] <= t <= De[i.id]:
                U[i.id, t] = 1
            else:
                U[i.id, t] = 0
    x = mdl.continuous_var_cube(
        keys1=vehicle_range, keys2=CS_range, keys3=time_range, lb=0, ub=50, name="x"
    )
    w = mdl.binary_var_matrix(keys1=vehicle_range, keys2=CS_range, name="w")
    loss = mdl.continuous_var_dict(vehicle_range, lb=0, name="loss")
    p_star = mdl.continuous_var(lb=0, name="p_star")
    # SoC = mdl.continuous_var_matrix(keys1=vehicle_range, keys2=time_range, lb=0, name='SoC')
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(x[i.id, j, t] for i in vehicles for j in CS_range)
            <= grid_capacity[t],
            "C1",
        )
    for j in charging_stations:
        for t in time_range:
            mdl.add_constraint(
                mdl.sum(w[i.id, j.id] * U[i.id, t] for i in vehicles)
                <= number_of_connectors,
                "C2",
            )
            mdl.add_constraint(
                mdl.sum(x[i.id, j.id, t] for i in vehicles) <= j.power, "C2"
            )
    for i in vehicles:
        mdl.add_constraint(mdl.sum(w[i.id, j] for j in CS_range) <= 1, "C3")
        for j in charging_stations:
            for t in time_range:
                mdl.add_constraint(
                    x[i.id, j.id, t] <= w[i.id, j.id] * U[i.id, t] * j.power, "C4"
                )
    for i in vehicles:
        mdl.add_constraint(
            loss[i.id]
            >= service_level * D[i.id]
            - mdl.sum(
                x[i.id, j.id, t] * delta_time / 60
                for t in time_range
                for j in charging_stations
            )
        )
        mdl.add_constraint(
            D[i.id]
            >= mdl.sum(
                x[i.id, j.id, t] * delta_time / 60
                for t in time_range
                for j in charging_stations
            )
        )
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(x[i.id, j, t] for i in vehicles for j in CS_range)
            - l_star
            + baseload[t]
            <= p_star,
            "C6",
        )
    # mdl.minimize(mdl.sum(x[i.id,j.id , t] * c[t] for i in vehicles for t in time_range for j in charging_stations) +
    #              mdl.sum(loss[i.id] * 1000000 for i in vehicles) + c_p * p_star)

    mdl.maximize(
        mdl.sum(
            mdl.sum(x[i.id, j.id, t] for i in vehicles for j in charging_stations)
            * delta_time
            / 60
            * (0.5 - c[t])
            for t in time_range
        )
        + mdl.sum(
            (generation[t] - baseload[t]) * delta_time / 60 * c[t] for t in time_range
        )
        - c_p * p_star
    )

    assert mdl.solve()
    # print(mdl.sum(
    #     x[i.id, j.id, t] * delta_time / 60 * (0.5 - c[t]) for i in vehicles for j in charging_stations for t in
    #     time_range).solution_value)
    for i in vehicles:
        for t in time_range:
            i.charge_schedule[t] = mdl.sum(
                x[i.id, j.id, t] for j in charging_stations
            ).solution_value
            # if i.charge_schedule[t] >0:
            # print(t,i.charge_schedule[t], D[i.id])
        # print(D[i.id],loss[i.id].solution_value)
        for j in charging_stations:
            if w[i.id, j.id].solution_value > 0:
                i.assigned_charger = j


def perfect_info_charging_routing_storage(
    vehicles,
    charging_stations,
    env,
    grid_capacity,
    electricity_cost,
    sim_time,
    baseload,
    storage,
    service_level=1,
    time_range=24 * 5,
):
    """
    It scales the charging power of all vehicles using multi-period optimization with perfect information
    :param service_level:
    :param sim_time: end of simulation
    :param electricity_cost: Time of use electricity tarrif
    :param grid_capacity:
    :param env: Simpy environment
    :param charging_stations: all charging_stations
    :param vehicles: all charging vehicles
    Objective function: maximize the charging power for low flexibility
    :return: charging power and routing for all vehicles
    """
    mdl = Model("perfect_info_scaling")
    vehicle_range = []
    delta_time = 60

    for i in vehicles:
        vehicle_range.append(i.id)
    CS_range = []
    sub_vehicles_charging = {}
    sub_vehicles = {}
    for j in charging_stations:
        CS_range.append(j.id)
        j.charging_vehicles = [x for x in j.connected_vehicles if x.mode == "Connected"]
        sub_vehicles_charging[j] = j.charging_vehicles
        sub_vehicles = j.connected_vehicles
    number_of_connectors = j.number_of_connectors

    time_range = range(max(round((sim_time - env.now) / delta_time), time_range))
    l_star = Configuration.instance().peak_threshold
    # l_predicted = list(np.random.normal(100, 20, len(time_range)+1))
    c_p = Configuration.instance().peak_cost
    D = {}  # Energy demand
    De = {}  # Departure time
    A = {}
    c = {}
    for t in time_range:
        for j in range(0, 24):
            if j * 60 <= env.now % 1440 <= (j + 1) * 60:
                hour = j
        c[t] = electricity_cost[hour]
    U = {}
    for i in vehicles:
        D[i.id] = i.energy_requested
        # SOC[i.id] = i.energy_charged
        A[i.id] = int(min(i.arrival_period, sim_time) / delta_time)
        De[i.id] = int(min(i.departure_period, sim_time) / delta_time) - 1
        for t in time_range:
            if A[i.id] <= t <= De[i.id]:
                U[i.id, t] = 1
            else:
                U[i.id, t] = 0

    x = mdl.continuous_var_cube(
        keys1=vehicle_range, keys2=CS_range, keys3=time_range, lb=0, ub=50, name="x"
    )
    w = mdl.binary_var_matrix(keys1=vehicle_range, keys2=CS_range, name="w")
    b_direction = mdl.binary_var_dict(time_range, name="b_direction")
    b_charge = mdl.continuous_var_dict(
        time_range, lb=0, ub=storage.kW_charge_peak, name="b_charge"
    )
    b_discharge = mdl.continuous_var_dict(
        time_range, lb=0, ub=storage.kW_discharge_peak, name="b_discharge"
    )
    SoC = mdl.continuous_var_dict(time_range, lb=0, name="SoC")
    loss = mdl.continuous_var_dict(vehicle_range, lb=0, name="loss")
    p_star = mdl.continuous_var(lb=0, name="p_star")
    # SoC = mdl.continuous_var_matrix(keys1=vehicle_range, keys2=time_range, lb=0, name='SoC')
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(x[i.id, j, t] for i in vehicles for j in CS_range)
            + b_charge[t]
            - b_discharge[t]
            <= grid_capacity[t],
            "C1",
        )
    for j in charging_stations:
        for t in time_range:
            mdl.add_constraint(
                mdl.sum(w[i.id, j.id] * U[i.id, t] for i in vehicles)
                <= number_of_connectors,
                "C2",
            )
            mdl.add_constraint(
                mdl.sum(x[i.id, j.id, t] for i in vehicles) <= j.power, "C2"
            )
    for i in vehicles:
        mdl.add_constraint(mdl.sum(w[i.id, j] for j in CS_range) <= 1, "C3")
        for j in charging_stations:
            for t in time_range:
                mdl.add_constraint(
                    x[i.id, j.id, t] <= w[i.id, j.id] * U[i.id, t] * j.power, "C4"
                )
    for i in vehicles:
        mdl.add_constraint(
            loss[i.id]
            >= service_level * D[i.id]
            - mdl.sum(
                x[i.id, j.id, t] * delta_time / 60
                for t in time_range
                for j in charging_stations
            )
        )
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(x[i.id, j, t] for i in vehicles for j in CS_range)
            - l_star
            + baseload[t]
            + b_charge[t]
            - b_discharge[t]
            <= p_star,
            "C6",
        )
    mdl.add_constraint(SoC[0] == 0)
    mdl.add_constraint(b_charge[0] <= b_direction[0] * storage.kW_charge_peak)
    mdl.add_constraint(
        b_discharge[0] <= (1 - b_direction[0]) * storage.kW_discharge_peak
    )
    mdl.add_constraint(0 >= b_discharge[0])
    mdl.add_constraint(b_charge[0] <= storage.kW_charge_peak)
    for t in range(1, len(time_range)):
        mdl.add_constraint(SoC[t] == SoC[t - 1] + b_charge[t - 1] - b_discharge[t - 1])
        mdl.add_constraint(b_charge[t] <= b_direction[t] * storage.kW_charge_peak)
        mdl.add_constraint(
            b_discharge[t] <= (1 - b_direction[t]) * storage.kW_discharge_peak
        )
        mdl.add_constraint(SoC[t - 1] >= b_discharge[t])
        mdl.add_constraint(b_charge[t] <= storage.kW_discharge_peak)
    mdl.minimize(
        mdl.sum(
            x[i.id, j.id, t] * c[t]
            for i in vehicles
            for t in time_range
            for j in charging_stations
        )
        + mdl.sum(loss[i.id] * 1000000 for i in vehicles)
        + mdl.sum(b_charge[t] * c[t] - b_discharge[t] * c[t] for t in time_range)
        + c_p * p_star
    )

    assert mdl.solve()
    # print(mdl.report())
    for i in vehicles:
        for t in time_range:
            i.charge_schedule[t] = mdl.sum(
                x[i.id, j.id, t] for j in charging_stations
            ).solution_value
            # if i.charge_schedule[t] >0:
            # print(t,i.charge_schedule[t], D[i.id])
        # print(D[i.id],loss[i.id].solution_value)
        for j in charging_stations:
            if w[i.id, j.id].solution_value > 0:
                i.assigned_charger = j
        for t in time_range:
            storage.charge_schedule[t] = SoC[t].solution_value


def perfect_info_pricing_charging_routing(
    vehicles,
    charging_stations,
    env,
    grid_capacity,
    electricity_cost,
    sim_time,
    baseload,
    generation=None,
    service_level=1,
    time_range=24,
):
    """
    It scales the charging power of all vehicles using multi-period optimization with perfect information
    :param service_level:
    :param sim_time: end of simulation
    :param electricity_cost: Time of use electricity tarrif
    :param grid_capacity:
    :param env: Simpy environment
    :param charging_stations: all charging_stations
    :param vehicles: all charging vehicles
    Objective function: maximize the charging power for low flexibility
    :return: charging power and routing for all vehicles
    """
    mdl = Model("perfect_info_scaling")
    vehicle_range = []
    delta_time = 60
    sim_duration = 1

    time_range = range(
        max(round((sim_time - env.now) / delta_time), time_range * sim_duration)
    )
    beta = {}

    for i in vehicles:
        vehicle_range.append(i.id)
        beta[i.id] = i.utility_beta

    l_star = Configuration.instance().peak_threshold
    # l_predicted = list(np.random.normal(100, 20, len(time_range)+1))
    P_O = 0.2
    c_p = Configuration.instance().peak_cost
    D = {}  # Energy demand
    De = {}  # Departure time
    A = {}
    A_binary = {}
    c = {}
    duration = {}
    for t in time_range:
        hour = int((t % 24))
        c[t] = electricity_cost[hour]
    U = {}
    for i in vehicles:
        D[i.id] = i.energy_requested
        # SOC[i.id] = i.energy_charged
        A[i.id] = int(min(i.arrival_period, sim_time) / delta_time)

        De[i.id] = int(min(i.departure_period, sim_time) / delta_time) - 1
        duration[i.id] = D[i.id] - A[i.id]

        for t in time_range:
            if A[i.id] <= t <= De[i.id]:
                U[i.id, t] = 1
            else:
                U[i.id, t] = 0
        for t in time_range:
            if A[i.id] == t:
                A_binary[i.id, t] = 1
            else:
                A_binary[i.id, t] = 0
    # x = mdl.continuous_var_matrix(keys1=vehicle_range, keys2=time_range, lb=0, ub=100, name='x')
    y = mdl.continuous_var_matrix(
        keys1=vehicle_range, keys2=time_range, lb=0, ub=100, name="y"
    )
    z = mdl.binary_var_dict(keys=vehicle_range, name="z")
    alpha = mdl.continuous_var_dict(keys=time_range, name="alpha")
    p_star = mdl.continuous_var(lb=0, name="p_star")
    # SoC = mdl.continuous_var_matrix(keys1=vehicle_range, keys2=time_range, lb=0, name='SoC')
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(y[i.id, t] for i in vehicles) <= grid_capacity[t], "C1"
        )
    for t in time_range:
        for i in vehicles:
            mdl.add_constraint(y[i.id, t] <= z[i.id] * U[i.id, t] * 50, "C2")
    for t in time_range:
        mdl.add_constraint(mdl.sum(z[i.id] for i in vehicles * U[i.id, t]) <= 100, "C2")

    x = {}
    for i in vehicles:
        for t in time_range:
            x[i.id, t] = (
                (P_O + 2 * beta[i.id] * D[i.id])
                * duration[i.id]
                / (2 * (beta[i.id] * duration[i.id] - alpha[t]))
            ) * A_binary[i.id]

    for i in vehicles:
        mdl.add_constraint(
            mdl.sum(y[i.id, t] for t in time_range)
            >= mdl.sum(x[i.id, t] for t in time_range),
            "C3",
        )
    e_grid = {}
    e_grid[t] = mdl.sum(y[i.id, t] for i in vehicles) * delta_time / 60
    for t in time_range:
        mdl.add_constraint(e_grid[t] - l_star <= p_star, "C6")

    costs = mdl.sum(e_grid[t] * c[t] for t in time_range) + c_p * p_star
    (
        mdl.maximize(
            mdl.sum(P_O + alpha[t] * x[i.id, t] / duration[i.id]) * x[i.id, t]
            for t in time_range
        )
        - costs
    )
    # mdl.maximize(mdl.sum(mdl.sum(
    #     x[i.id, j.id, t] for i in vehicles for j in charging_stations) * delta_time / 60 * (0.5 - c[t]) for t in
    #                      time_range) +
    #              mdl.sum(
    #                  (generation[t] - baseload[t]) * delta_time / 60 * c[t] for t in time_range) - c_p * p_star)
    mdl.print_information()
    assert mdl.solve()
    for i in vehicles:
        for t in time_range:
            i.charge_schedule[t] = mdl.sum(
                x[i.id, j.id, t] for j in charging_stations
            ).solution_value
        # for j in charging_stations:
        #     if w[i.id, j.id].solution_value > 0:
        #         i.assigned_charger = j
