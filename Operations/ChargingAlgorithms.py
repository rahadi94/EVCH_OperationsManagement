import numpy as np
from docplex.mp.model import Model

from Environment.helper.configuration.configuration import Configuration
from Environment.log import lg


# Uncontrolled charging (ignores all infrastructure)
# Equal sharing/scaling based algorithms: (current implementation)
# Sorting-based algorithms: First Come First Served (FCFS); Earliest Deadline First (EDF), least-laxity-first-served (LLFS), round_robin (RR), priority-levels
# Optimization-based algorithms: myopic (only current period), multi-period (with and without future arrivals) --> without routing
# Dynamic programming/RL algos: multi-period
# --> Each of these algorithms must return a charging schedule for each vehicle!!!
# --> this is the so-called charging profile which can then be given to the EVSE (CS) and executed
# --> Charging schedules are updated every planning period of length n


# UNCONTROLLED CHARGING#


def uncontrolled(env, connected_vehicles, charging_capacity, planning_period_length):
    """
    :param env:
    :param requests:
    :param charging_capacity:
    :param planning_period_length:
    :return:
    """
    # get connected vehicles
    # connected_vehicles = [x for x in requests if x.mode=="Connected" or x.mode=="Arrived" or x.mode=="Charging"]
    # connected_vehicles = [x for x in requests if x.assigned_charger]
    # connected_vehicles = [x for x in requests if x.mode == 'Connected']

    lg.info(
        f"{len(connected_vehicles)} connected at start of planning period {env.now}"
    )
    # sort list according to algo rule (arrival, and secondary remaining energy to charge for tie breaking)
    for vehicle in connected_vehicles:
        power = min(
            vehicle.assigned_charger.power,
            vehicle.remaining_energy_deficit / (planning_period_length / 60),
        )
        vehicle.charging_power = power


# SORTING-BASED ALGORITHMS
# First Come First Served (FCFS)
# Earliest Deadline First (EDF)
# least-laxity-first-served (LLFS)
# round_robin (RR) #TODO
# custom_priority #TODO


def first_come_first_served(
    env,
    connected_vehicles,
    charging_stations,
    charging_capacity,
    free_grid_capacity,
    planning_period_length,
):
    """
    Serves based on FCFS basis charging at min(max_capa, rate to fulfill charging request fully during period)
    :param env:
    :param requests:
    :param charging_stations:
    :param charging_capacity:
    :param free_grid_capacity:
    :param planning_period_length:
    :return:
    """
    if type(free_grid_capacity) == list:
        free_grid_capacity = free_grid_capacity[
            0
        ]  # need to index 1st element since this is a myopic algo

    free_grid_capacity = free_grid_capacity

    # get connected vehicles
    # connected_vehicles = [x for x in requests if x.mode=="Connected" or x.mode=="Arrived" or x.mode=="Charging"]
    # connected_vehicles = [x for x in requests if x.assigned_charger]
    # connected_vehicles = [x for x in requests if x.mode == 'Connected']
    lg.info(
        f"{len(connected_vehicles)} connected at start of planning period {env.now}"
    )
    # sort list according to algo rule (arrival, and secondary remaining energy to charge for tie breaking)
    connected_vehicles_sorted = sorted(
        connected_vehicles,
        key=lambda x: (x.arrival_period, -x.remaining_energy_deficit),
    )

    # assign available power until exhausted
    available_power = free_grid_capacity
    lg.info(
        f"{round(available_power)} kW free capacity available at start of planning period starting {env.now}"
    )

    # iterate through EVs and assign power
    if len(connected_vehicles_sorted) > 0:
        # reset vehicle charge rate to 0
        for vehicle in connected_vehicles_sorted:
            vehicle.charging_power = 0

        # reallocate capacity for next planning period
        vehicle_counter = 0
        assigned_power_total = 0  # tracks total power
        assigned_power_per_CS = {}  # tracks power per charger
        for c in charging_stations:
            assigned_power_per_CS[c.id] = 0  # initialize

        for vehicle in connected_vehicles_sorted:

            # break condition 1: all power assigned
            if (
                assigned_power_total >= available_power
            ):  # ensures that vehicle can still be served at full capa
                break

            # break condition 2: all vehicles served
            if vehicle_counter == len(connected_vehicles_sorted):
                break

            if assigned_power_per_CS[vehicle.assigned_charger.id] > 0:
                vehicle_counter += 1

            else:
                vehicle_counter += 1
                power = min(
                    vehicle.assigned_charger.power,  # max power at charger (not strictly needed as condition)
                    available_power
                    - assigned_power_total,  # max remaining grid + battery power
                    (
                        vehicle.assigned_charger.power
                        - assigned_power_per_CS[vehicle.assigned_charger.id]
                    ),  # max remaining power at charger
                    vehicle.remaining_energy_deficit / (planning_period_length / 60),
                )  # remaining requested energy of vehicle
                power = max(power, 0)
                vehicle.charging_power = power

                assigned_power_total += power
                assigned_power_per_CS[vehicle.assigned_charger.id] += power

        lg.info(
            f"{assigned_power_total} kW was assigned based on FCFS rule for planning period starting {env.now}"
        )


def earliest_deadline_first(
    env,
    connected_vehicles,
    charging_stations,
    charging_capacity,
    free_grid_capacity,
    planning_period_length,
):
    """
    Serves based on EDF basis charging at min(max_capa, rate to fulfill charging request fully during period)
    :param env:
    :param requests:
    :param charging_stations:
    :param charging_capacity:
    :param free_grid_capacity:
    :param planning_period_length:
    :return:
    """
    if type(free_grid_capacity) == list:
        free_grid_capacity = free_grid_capacity[
            0
        ]  # need to index 1st element since this is a myopic algo

    free_grid_capacity = free_grid_capacity

    # get connected vehicles (i.e., those that have an assigned charges at the beginning of the planning window)
    # connected_vehicles = [x for x in requests if x.assigned_charger]
    # connected_vehicles = [x for x in requests if x.mode == 'Connected']
    lg.info(
        f"{len(connected_vehicles)} connected at start of planning period {env.now}"
    )

    # sort list according to algo rule (departure (ascending), and secondary remaining energy to charge for tie breaking (descending))
    connected_vehicles_sorted = sorted(
        connected_vehicles,
        key=lambda x: (x.departure_period, -x.remaining_energy_deficit),
    )

    # assign available power until exhausted
    available_power = free_grid_capacity
    lg.info(
        f"{round(available_power)} kW free capacity available at start of planning period starting {env.now}"
    )

    # iterate through EVs and assign power
    if len(connected_vehicles_sorted) > 0:
        # reset vehicle charge rate to 0
        for vehicle in connected_vehicles_sorted:
            vehicle.charging_power = 0

        # reallocate capacity for next planning period
        vehicle_counter = 0
        assigned_power_total = 0  # tracks total power
        assigned_power_per_CS = {}  # tracks power per charger
        for c in charging_stations:
            assigned_power_per_CS[c.id] = 0  # initialize

        for vehicle in connected_vehicles_sorted:

            # break condition 1: all power assigned
            if (
                assigned_power_total >= available_power
            ):  # ensures that vehicle can still be served at full capa
                break

            # break condition 2: all vehicles served
            if vehicle_counter == len(connected_vehicles_sorted):
                break

            if assigned_power_per_CS[vehicle.assigned_charger.id] > 0:
                vehicle_counter += 1

            else:
                vehicle_counter += 1
                power = min(
                    vehicle.assigned_charger.power,  # max power at charger (not strictly needed as condition)
                    available_power
                    - assigned_power_total,  # max remaining grid + battery power
                    (
                        vehicle.assigned_charger.power
                        - assigned_power_per_CS[vehicle.assigned_charger.id]
                    ),  # max remaining power at charger
                    vehicle.remaining_energy_deficit / (planning_period_length / 60),
                )  # remaining requested energy of vehicle
                vehicle.charging_power = power
                assigned_power_total += power
                assigned_power_per_CS[vehicle.assigned_charger.id] += power

        lg.info(
            f"{assigned_power_total} kW was assigned based on EDF rule for planning period starting {env.now}"
        )


def least_laxity_first(
    env,
    connected_vehicles,
    charging_stations,
    charging_capacity,
    free_grid_capacity,
    planning_period_length,
):
    """
    Serves based on Least-Laxity-First basis charging at min(max_capa, rate to fulfill charging request fully during period)
    :param env:
    :param requests:
    :param charging_stations:
    :param charging_capacity:
    :param free_grid_capacity:
    :param planning_period_length:
    :return:
    """
    if type(free_grid_capacity) == list:
        free_grid_capacity = free_grid_capacity[
            0
        ]  # need to index 1st element since this is a myopic algo

    free_grid_capacity = free_grid_capacity
    # get connected vehicles
    # connected_vehicles = [x for x in requests if x.mode == "Connected" or x.mode == "Arrived" or x.mode == "Charging"]
    # connected_vehicles = [x for x in requests if x.assigned_charger]
    # connected_vehicles = [x for x in requests if x.mode == 'Connected']
    lg.info(
        f"{len(connected_vehicles)} connected at start of planning period {env.now}"
    )

    # update laxity
    for vehicle in connected_vehicles:
        vehicle.laxity = vehicle.remaining_park_duration - (
            vehicle.remaining_energy_deficit / vehicle.assigned_charger.power
        )

    # sort list according to algo rule (laxity (ascending))
    connected_vehicles_sorted = sorted(connected_vehicles, key=lambda x: (x.laxity))

    # assign available power until exhausted
    available_power = free_grid_capacity
    lg.info(
        f"{round(available_power)} kW free capacity available at start of planning period starting {env.now}"
    )

    # iterate through EVs and assign power
    if len(connected_vehicles_sorted) > 0:
        # reset vehicle charge rate to 0
        for vehicle in connected_vehicles_sorted:
            vehicle.charging_power = 0

        # reallocate capacity for next planning period
        vehicle_counter = 0
        assigned_power_total = 0  # tracks total power
        assigned_power_per_CS = {}  # tracks power per charger
        for c in charging_stations:
            assigned_power_per_CS[c.id] = 0  # initialize

        for vehicle in connected_vehicles_sorted:

            # break condition 1: all power assigned
            if (
                assigned_power_total >= available_power
            ):  # ensures that vehicle can still be served at full capa
                break

            # break condition 2: all vehicles served
            if vehicle_counter == len(connected_vehicles_sorted):
                break

            # if assigned_power_per_CS[vehicle.assigned_charger.id] > 0:
            #     vehicle_counter += 1

            else:
                vehicle_counter += 1
                power = min(
                    vehicle.assigned_charger.power,  # max power at charger (not strictly needed as condition)
                    available_power
                    - assigned_power_total,  # max remaining grid + battery power
                    (
                        vehicle.assigned_charger.power
                        - assigned_power_per_CS[vehicle.assigned_charger.id]
                    ),  # max remaining power at charger
                    vehicle.remaining_energy_deficit / (planning_period_length / 60),
                )  # remaining requested energy of vehicle
                power = max(power, 0)
                vehicle.charging_power = power
                assigned_power_total += power
                assigned_power_per_CS[vehicle.assigned_charger.id] += power

        lg.info(
            f"{assigned_power_total} kW was assigned based on Least-Laxity-First rule for planning period starting {env.now}"
        )


def round_robin(charging_stations, charging_capacity, free_grid_capacity):

    return "schedule"


def custom_priority(charging_stations, charging_capacity, free_grid_capacity):
    # TODO: assign a priority to vehicles, e.g., types --> have priority attribute in vehicle class!

    return "schedule"


def equal_sharing(
    charging_stations, charging_capacity, free_grid_capacity, free_battery_capacity
):
    """
    Computes same-period charging power per each connected vehicle by cascading available grid power equally through the network
    :param charging_stations: list of charging station objects
    :param charging_capacity: available charging_capacity capacity per charger
    :param free_grid_capacity: list of available grid capacity
    :return:
    """

    if type(free_grid_capacity) == list:
        free_grid_capacity = free_grid_capacity[
            0
        ]  # need to index 1st element since this is a myopic algo

    free_grid_capacity = (
        free_grid_capacity  # + free_battery_capacity  # add battery capa to grid capa
    )

    num_charging_vehicles = 0
    num_active_chargers = 0
    for c in charging_stations:
        c.charging_vehicles = [x for x in c.connected_vehicles if x.mode == "Connected"]
        if len(c.charging_vehicles) > 0:
            num_active_chargers += 1
            num_charging_vehicles += len(c.charging_vehicles)
    total_power_consumption = num_active_chargers * charging_capacity
    if total_power_consumption > free_grid_capacity:
        lg.info("Grid capacity reached")
        for c in charging_stations:
            c.available_power = free_grid_capacity / num_active_chargers
            # len(c.charging_vehicles) / num_charging_vehicles * grid_capacity
    else:
        for charger in charging_stations:
            charger.available_power = charger.power

    for charger in charging_stations:
        number_of_charging_vehicles = len(
            [x for x in charger.connected_vehicles if x.mode == "Connected"]
        )
        if number_of_charging_vehicles > 0:
            number_of_charging_vehicles = len(
                [x for x in charger.connected_vehicles if x.mode == "Connected"]
            )
            power = charger.available_power / number_of_charging_vehicles
            for vehicle in charger.connected_vehicles:
                vehicle.charging_power = power


def average_power(
    env,
    connected_vehicles,
    charging_capacity,
    planning_period_length,
    free_grid_capacity,
):
    """
    :param env:
    :param requests:
    :param charging_capacity:
    :param planning_period_length:
    :return:
    """
    # get connected vehicles
    # connected_vehicles = [x for x in requests if x.mode=="Connected" or x.mode=="Arrived" or x.mode=="Charging"]
    # connected_vehicles = [x for x in requests if x.assigned_charger]
    # connected_vehicles = [x for x in requests if x.mode == 'Connected']
    if type(free_grid_capacity) == list:
        free_grid_capacity = free_grid_capacity[
            0
        ]  # need to index 1st element since this is a myopic algo

    total_power_consumption = 0

    lg.info(
        f"{len(connected_vehicles)} connected at start of planning period {env.now}"
    )
    # sort list according to algo rule (arrival, and secondary remaining energy to charge for tie breaking)
    for vehicle in connected_vehicles:
        power = min(
            vehicle.assigned_charger.power,
            vehicle.energy_requested / min(vehicle.park_duration, 100000) * 60,
        )
        # print(f'{power},{vehicle.energy_requested},{vehicle.park_duration}')
        vehicle.charging_power = power
        total_power_consumption += power
    if total_power_consumption > free_grid_capacity:
        for vehicle in connected_vehicles:
            vehicle.charging_power -= (
                total_power_consumption - free_grid_capacity
            ) / len(connected_vehicles)


def online_myopic(
    vehicles,
    charging_stations,
    env,
    grid_capacity,
    alpha=0,
    optimization_period_length=15,
):
    """
    It scales the charging power of all vehicles using a single period optimization
    :param alpha: Weight for the SoC in the objective function
    :param grid_capacity:
    :param env: Simpy environment
    :param charging_stations: all charging_stations
    :param vehicles: all charging vehicles
    Objective function: maximize the charging power for low flexibility
    :return: charging power for all vehicles
    """
    mdl = Model("myopic_scaling")
    vehicle_range = []
    for i in vehicles:
        vehicle_range.append(i.id)
    CS_range = []
    sub_vehicles = {}
    for j in charging_stations:
        CS_range.append(j.id)
        j.charging_vehicles = [x for x in j.connected_vehicles if x.mode == "Connected"]
        sub_vehicles[j] = j.charging_vehicles

    l = (
        {}
    )  # urgency (this is a measure remaining requested energy / remaining time --> The higher the more urgent)
    SoC = {}  # State of charge
    for i in vehicles:
        l[i.id] = (i.energy_requested - i.energy_charged) / (
            i.departure_period - env.now
        )
        SoC[i.id] = i.energy_charged / i.energy_requested

    x = mdl.continuous_var_dict(vehicle_range, lb=0, name="x")

    mdl.add_constraint(mdl.sum(x[i.id] for i in vehicles) <= grid_capacity[0], "C1")
    for j in charging_stations:
        mdl.add_constraint(mdl.sum(x[i.id] for i in sub_vehicles[j]) <= j.power, "C2")
    for i in vehicles:
        mdl.add_constraint(
            x[i.id] * optimization_period_length / 60
            <= i.energy_requested - i.energy_charged,
            "C3",
        )

    mdl.maximize(mdl.sum(x[i.id] * (l[i.id] - SoC[i.id] * alpha) for i in vehicles))

    assert mdl.solve()
    # mdl.report()
    for i in vehicles:
        i.charging_power = x[i.id].solution_value


def online_multi_period(
    vehicles,
    charging_stations,
    env,
    electricity_cost,
    sim_time,
    peak_load_history,
    free_grid_capa_actual,
    free_grid_capa_predicted,
    information_case="perfect_info",
    flex_margin=0.05,
    service_level=1,
    optimization_period_length=15,
    num_lookahead_planning_periods=2,
    vehicle_ramp_limit=4,
    transformer_ramp_limit=10000,
    peak_threshold=300,
):
    """
    It scales the charging power of all vehicles using multi-period optimization with perfect or predicted information
    :param service_level:
    :param sim_time: end of simulation
    :param electricity_cost: Time of use electricity tarrif
    :param grid_capacity:
    :param env: Simpy environment
    :param charging_stations: all charging_stations
    :param vehicles: all charging vehicles
    Objective function: maximize the charging power for low flexibility
    :return: charging power for all vehicles
    """
    # TODO: Considering a prediction of the future demand --> Use some basic expectation (e.g. cond. historical average of 1) arrival rate, 2) request per vehicle)
    # TODO: Consider future arrivals (true or predicted)
    # TODO: Set variable foresight!

    mdl = Model("perfect_info_scaling")
    vehicle_range = []
    for i in vehicles:
        vehicle_range.append(i.id)
    CS_range = []
    sub_vehicles = {}
    for j in charging_stations:
        CS_range.append(j.id)
        j.charging_vehicles = [x for x in j.connected_vehicles if x.mode == "Connected"]
        sub_vehicles[j] = j.charging_vehicles

    # range of simulation (in planning periods, shorten if sim_end would be exceededs)
    time_range = range(
        int(
            min(
                round((sim_time - env.now) / optimization_period_length) + 1,
                num_lookahead_planning_periods,
            )
        )
    )

    # historic peak demand (exceeding it will incur peak charge!)
    l_star = peak_threshold

    # define grid_capacity for planning horizon
    if information_case == "perfect_info":
        free_grid_capacity = free_grid_capa_actual[: len(time_range)]
    elif information_case == "predicted_info":
        free_grid_capacity = free_grid_capa_predicted[: len(time_range)]
    # print(free_grid_capacity)
    # add flex_marging on free grid_capacity for planning horizon
    flex_margin_list = []
    try:
        for t in time_range:
            if t == 0:
                flex_margin_list.append(
                    0
                )  # no safety margin in current period (we have full visibility of active vehicles)
            else:
                flex_margin_list.append(
                    free_grid_capacity[t] * flex_margin
                )  # safety margin in future periods to account for new arrivals
                # TODO: We might consider using a variable margin depending on arrival rates (more margin of more arrivals)!
    except:
        for t in range(len(free_grid_capacity), num_lookahead_planning_periods + 1):
            free_grid_capacity.append(free_grid_capacity[-1])
        # print(free_grid_capacity)
        for t in time_range:
            if t == 0:
                flex_margin_list.append(
                    0
                )  # no safety margin in current period (we have full visibility of active vehicles)
            else:
                flex_margin_list.append(free_grid_capacity[t] * flex_margin)  # saf
    # incorporate flex_margin into grid_capacity
    grid_capacity = []
    for t in time_range:
        grid_capacity.append(round(free_grid_capacity[t] - flex_margin_list[t], 2))
    # print(grid_capacity)

    # define parameters
    c_p = Configuration.instance().peak_cost
    D = {}  # Energy demand
    De = {}  # Departure time
    SOC = {}
    c = {}
    for t in time_range:
        for j in range(0, 24):
            if j * 60 <= env.now % 1440 <= (j + 1) * 60:
                hour = j
        c[t] = electricity_cost[hour]
    for i in vehicles:
        D[i.id] = i.energy_requested - i.energy_charged
        SOC[i.id] = i.energy_charged
        De[i.id] = int(
            round(
                min(
                    (
                        max(
                            (min(i.departure_period, sim_time) - env.now)
                            / optimization_period_length,
                            0,
                        )
                    ),
                    time_range[-1],
                )
            )
        )
        # print(De[i.id])

    x = mdl.continuous_var_matrix(
        keys1=vehicle_range, keys2=time_range, lb=0, ub=44, name="x"
    )
    b_direction = mdl.binary_var_dict(range(1), name="b_direction")
    b_charge = mdl.continuous_var_dict(range(1), lb=0, name="b_charge")
    b_discharge = mdl.continuous_var_dict(range(1), lb=0, name="b_discharge")
    SoC = mdl.continuous_var_dict(range(1), lb=0, name="SoC")
    p_star = mdl.continuous_var(lb=0, name="p_star")
    loss = mdl.continuous_var_dict(vehicle_range, lb=0, name="loss")
    # SoC = mdl.continuous_var_matrix(keys1=vehicle_range, keys2=time_range, lb=0, name='SoC')
    # print(storage.SoC)
    # mdl.add_constraint(SoC[0] == max(0, 0))
    # mdl.add_constraint(b_charge[0] <= b_direction[0] * 50)
    # mdl.add_constraint(b_discharge[0] <= (1 - b_direction[0]) * 50)
    # mdl.add_constraint(SoC[0] >= b_discharge[0])
    # mdl.add_constraint(b_charge[0] <= storage.max_energy_stored_kWh - 0)

    # if storage.max_energy_stored_kWh > 0:
    #     ### Temporal arbitrage
    #
    #     # if c[0] <= 0.1:
    #     #     mdl.add_constraint(b_charge[0] >= 50)
    #
    #     ### Peak shaving
    #
    #     usage = 0
    #     for i in vehicles:
    #         usage += i.charging_power
    #     if usage <= grid_capacity[0] / 6:
    #         mdl.add_constraint(b_charge[0] >=
    #                            min(50, grid_capacity[0] - usage - 50, storage.max_energy_stored_kWh))

    # for t in range(1, time_range[-1] + 1):
    #     mdl.add_constraint(SoC[t] == SoC[t - 1] + b_charge[t] - b_discharge[t])
    #     mdl.add_constraint(b_charge[t] <= b_direction[t] * 50)
    #     mdl.add_constraint(b_discharge[t] <= (1 - b_direction[t]) * 50)
    #     mdl.add_constraint(SoC[t - 1] >= b_discharge[t])
    #     mdl.add_constraint(b_charge[t] <= storage.max_energy_stored_kWh - SoC[t - 1])
    mdl.add_constraint(mdl.sum(x[i.id, 0] for i in vehicles) <= grid_capacity[0], "C1")
    # if vehicle_ramp_limit <= 22: # TODO: hard coded
    #     for i in vehicles:
    #         if i.charging_power > 0:
    #             mdl.add_constraint(x[i.id, 0] - i.charging_power <= vehicle_ramp_limit, 'C2')
    #             mdl.add_constraint(i.charging_power - x[i.id, 0] <= vehicle_ramp_limit, 'C3')
    mdl.add_constraint(
        mdl.sum(x[i.id, 0] - i.charging_power for i in vehicles)
        <= transformer_ramp_limit,
        "C2",
    )
    mdl.add_constraint(
        mdl.sum(-x[i.id, 0] + i.charging_power for i in vehicles)
        <= transformer_ramp_limit,
        "C2",
    )
    for j in charging_stations:
        mdl.add_constraint(
            mdl.sum(x[i.id, 0] for i in sub_vehicles[j]) <= j.power, "C4"
        )
    for t in range(1, time_range[-1]):
        mdl.add_constraint(
            mdl.sum(x[i.id, t] for i in vehicles) + 0 <= grid_capacity[t], "C5"
        )
        for j in charging_stations:
            mdl.add_constraint(
                mdl.sum(x[i.id, t] for i in sub_vehicles[j]) + 0 <= j.power, "C6"
            )  # TODO: We need to consider the l_predicted in EVCC capacity as well
    # for i in vehicles:
    #     mdl.add_constraint(loss[i.id] >= service_level * D[i.id] - mdl.sum(
    #         x[i.id, t] * planning_period_length/60 for t in range(De[i.id])), 'C7')
    for i in vehicles:
        mdl.add_constraint(
            loss[i.id]
            >= service_level * 1 * i.energy_requested
            - i.energy_charged
            - mdl.sum(
                x[i.id, t] * optimization_period_length / 60 for t in range(De[i.id])
            ),
            "C7",
        )
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(x[i.id, t] for i in vehicles) - l_star <= p_star, "C8"
        )
    # mdl.minimize(mdl.sum(x[i.id, t] * c[t] for i in vehicles for t in time_range) +
    #              mdl.sum(loss[i.id] * 10000 for i in vehicles) +
    #              c_p * p_star)
    mdl.maximize(
        mdl.sum(x[i.id, t] * (0.5 - c[t]) for i in vehicles for t in time_range)
        - c_p * p_star
    )

    assert mdl.solve()
    # print(mdl.report())
    # storage.charge_yn = b_direction[0].solution_value
    # storage.discharge_yn = 1 - b_direction[0].solution_value
    # storage.charging_power = b_charge[0].solution_value
    # storage.discharging_power = b_discharge[0].solution_value
    # print(storage.charging_power, storage.discharging_power)
    for i in vehicles:
        i.charging_power = x[i.id, 0].solution_value
        # print(f'loss_{i.id}={loss[i.id].solution_value}'
        #       f'x_{i.id}={x[i.id, 0].solution_value}')


def integrated_charging_storage(
    storage,
    vehicles,
    charging_stations,
    env,
    electricity_cost,
    sim_time,
    peak_load_history,
    free_grid_capa_actual,
    free_grid_capa_predicted,
    information_case="perfect_info",
    flex_margin=0.5,
    service_level=1,
    optimization_period_length=5,
    num_lookahead_planning_periods=6,
    vehicle_ramp_limit=4,
    transformer_ramp_limit=10000,
):
    """
    It scales the charging power of all vehicles using multi-period optimization with perfect or predicted information
    :param storage:
    :param service_level:
    :param sim_time: end of simulation
    :param electricity_cost: Time of use electricity tarrif
    :param grid_capacity:
    :param env: Simpy environment
    :param charging_stations: all charging_stations
    :param vehicles: all charging vehicles
    Objective function: maximize the charging power for low flexibility
    :return: charging power for all vehicles
    """
    # TODO: Considering a prediction of the future demand --> Use some basic expectation (e.g. cond. historical average of 1) arrival rate, 2) request per vehicle)
    # TODO: Consider future arrivals (true or predicted)
    # TODO: Set variable foresight!

    mdl = Model("integrated_charging_storage")
    vehicle_range = []
    for i in vehicles:
        vehicle_range.append(i.id)
    CS_range = []
    sub_vehicles = {}
    for j in charging_stations:
        CS_range.append(j.id)
        j.charging_vehicles = [x for x in j.connected_vehicles if x.mode == "Connected"]
        sub_vehicles[j] = j.charging_vehicles

    # range of simulation (in planning periods, shorten if sim_end would be exceededs)
    time_range = range(
        int(
            min(
                round((sim_time - env.now) / optimization_period_length) + 1,
                num_lookahead_planning_periods,
            )
        )
    )

    # historic peak demand (exceeding it will incur peak charge!)
    l_star = max(peak_load_history)

    # define grid_capacity for planning horizon
    if information_case == "perfect_info":
        free_grid_capacity = free_grid_capa_actual[: len(time_range)]
    elif information_case == "predicted_info":
        free_grid_capacity = free_grid_capa_predicted[: len(time_range)]
    # print(free_grid_capacity)
    # add flex_marging on free grid_capacity for planning horizon
    flex_margin_list = []
    try:
        for t in time_range:
            if t == 0:
                flex_margin_list.append(
                    0
                )  # no safety margin in current period (we have full visibility of active vehicles)
            else:
                flex_margin_list.append(
                    free_grid_capacity[t] * flex_margin
                )  # safety margin in future periods to account for new arrivals
                # TODO: We might consider using a variable margin depending on arrival rates (more margin of more arrivals)!
    except:
        for t in range(len(free_grid_capacity), num_lookahead_planning_periods + 1):
            free_grid_capacity.append(free_grid_capacity[-1])
        # print(free_grid_capacity)
        for t in time_range:
            if t == 0:
                flex_margin_list.append(
                    0
                )  # no safety margin in current period (we have full visibility of active vehicles)
            else:
                flex_margin_list.append(free_grid_capacity[t] * flex_margin)  # saf
    # incorporate flex_margin into grid_capacity
    grid_capacity = []
    for t in time_range:
        grid_capacity.append(round(free_grid_capacity[t] - flex_margin_list[t], 2))
    # print(grid_capacity)

    # define parameters
    c_p = Configuration.instance().peak_cost
    D = {}  # Energy demand
    De = {}  # Departure time
    SOC = {}
    c = {}
    for t in time_range:
        for j in range(0, 24):
            if j * 60 <= env.now % 1440 <= (j + 1) * 60:
                hour = j
        c[t] = electricity_cost[hour]
    for i in vehicles:
        D[i.id] = i.energy_requested - i.energy_charged
        SOC[i.id] = i.energy_charged
        De[i.id] = int(
            round(
                min(
                    (
                        max(
                            (min(i.departure_period, sim_time) - env.now)
                            / optimization_period_length,
                            0,
                        )
                    ),
                    time_range[-1],
                )
            )
        )
        # print(De[i.id])

    x = mdl.continuous_var_matrix(
        keys1=vehicle_range, keys2=time_range, lb=0, ub=50, name="x"
    )
    b_direction = mdl.binary_var_dict(range(1), name="b_direction")
    b_charge = mdl.continuous_var_dict(range(1), lb=0, name="b_charge")
    b_discharge = mdl.continuous_var_dict(range(1), lb=0, name="b_discharge")
    SoC = mdl.continuous_var_dict(range(1), lb=0, name="SoC")
    p_star = mdl.continuous_var(lb=0, name="p_star")
    loss = mdl.continuous_var_dict(vehicle_range, lb=0, name="loss")
    # SoC = mdl.continuous_var_matrix(keys1=vehicle_range, keys2=time_range, lb=0, name='SoC')
    # print(storage.SoC)
    mdl.add_constraint(SoC[0] == max(storage.SoC, 0))
    mdl.add_constraint(b_charge[0] <= b_direction[0] * storage.max_energy_stored_kWh)
    mdl.add_constraint(
        b_discharge[0] <= (1 - b_direction[0]) * storage.max_energy_stored_kWh
    )
    mdl.add_constraint(SoC[0] >= b_discharge[0])
    mdl.add_constraint(b_charge[0] <= storage.max_energy_stored_kWh - 0)

    # if storage.max_energy_stored_kWh>0:
    ### Temporal arbitrage

    # if c[0] <= 0.1:
    #     mdl.add_constraint(b_charge[0] >= 50)

    ## Peak shaving

    # usage =0
    # for i in vehicles:
    #     usage += i.charging_power
    # if usage <= grid_capacity[0]:
    #     mdl.add_constraint(b_charge[0] >=
    #                        min(storage.max_energy_stored_kWh, grid_capacity[0]))

    # for t in range(1, time_range[-1] + 1):
    #     mdl.add_constraint(SoC[t] == SoC[t - 1] + b_charge[t] - b_discharge[t])
    #     mdl.add_constraint(b_charge[t] <= b_direction[t] * 50)
    #     mdl.add_constraint(b_discharge[t] <= (1 - b_direction[t]) * 50)
    #     mdl.add_constraint(SoC[t - 1] >= b_discharge[t])
    #     mdl.add_constraint(b_charge[t] <= storage.max_energy_stored_kWh - SoC[t - 1])
    mdl.add_constraint(
        mdl.sum(x[i.id, 0] for i in vehicles)
        <= grid_capacity[0] - b_charge[0] + b_discharge[0],
        "C1",
    )
    # if vehicle_ramp_limit <= 22: # TODO: hard coded
    #     for i in vehicles:
    #         if i.charging_power > 0:
    #             mdl.add_constraint(x[i.id, 0] - i.charging_power <= vehicle_ramp_limit, 'C2')
    #             mdl.add_constraint(i.charging_power - x[i.id, 0] <= vehicle_ramp_limit, 'C3')
    mdl.add_constraint(
        mdl.sum(x[i.id, 0] - i.charging_power for i in vehicles)
        <= transformer_ramp_limit,
        "C2",
    )
    mdl.add_constraint(
        mdl.sum(-x[i.id, 0] + i.charging_power for i in vehicles)
        <= transformer_ramp_limit,
        "C2",
    )
    for j in charging_stations:
        mdl.add_constraint(
            mdl.sum(x[i.id, 0] for i in sub_vehicles[j]) <= j.power, "C4"
        )
    for t in range(1, time_range[-1]):
        mdl.add_constraint(
            mdl.sum(x[i.id, t] for i in vehicles) + 0 <= grid_capacity[t], "C5"
        )
        for j in charging_stations:
            mdl.add_constraint(
                mdl.sum(x[i.id, t] for i in sub_vehicles[j]) + 0 <= j.power, "C6"
            )  # TODO: We need to consider the l_predicted in EVCC capacity as well
    # for i in vehicles:
    #     mdl.add_constraint(loss[i.id] >= service_level * D[i.id] - mdl.sum(
    #         x[i.id, t] * planning_period_length/60 for t in range(De[i.id])), 'C7')
    for i in vehicles:
        mdl.add_constraint(
            loss[i.id]
            >= service_level * 1 * i.energy_requested
            - i.energy_charged
            - mdl.sum(
                x[i.id, t] * optimization_period_length / 60 for t in range(De[i.id])
            ),
            "C7",
        )
    for t in time_range:
        mdl.add_constraint(
            mdl.sum(x[i.id, t] for i in vehicles) - l_star <= p_star, "C8"
        )
    mdl.minimize(
        mdl.sum(x[i.id, t] * c[t] for i in vehicles for t in time_range)
        + mdl.sum(loss[i.id] * 10000 for i in vehicles)
        + c_p * p_star
        + mdl.sum((-b_discharge[0]) * c[0] for t in time_range) * 1
        + mdl.sum((+b_charge[0]) * c[0] for t in time_range)
    )

    assert mdl.solve()
    # print(mdl.report())
    storage.charge_yn = b_direction[0].solution_value
    storage.discharge_yn = 1 - b_direction[0].solution_value
    storage.charging_power = b_charge[0].solution_value
    storage.discharging_power = b_discharge[0].solution_value
    # print(storage.charging_power, storage.discharging_power)
    for i in vehicles:
        i.charging_power = x[i.id, 0].solution_value
        # print(f'loss_{i.id}={loss[i.id].solution_value}'
        #       f'x_{i.id}={x[i.id, 0].solution_value}')


##### LEGACY

'''def heuristic_power_sharing(charging_stations, charging_capacity, grid_capacity):
    """
    It shares the total grid capacity power among chargers based on the number of charging vehicles connected to that
    charger
    :param charging_stations:
    :param charging_capacity:
    :param grid_capacity:
    :return: charging power for each charger
    """
    num_charging_vehicles = 0
    num_active_chargers = 0
    for c in charging_stations:
        c.charging_vehicles = [x for x in c.connected_vehicles if x.mode == 'Charging']
        if len(c.charging_vehicles) > 0:
            num_active_chargers += 1
            num_charging_vehicles += len(c.charging_vehicles)
    total_power_consumption = num_active_chargers * charging_capacity
    if total_power_consumption > grid_capacity:
        lg.info('Grid capacity reached')
        for c in charging_stations:
            c.available_power = grid_capacity / num_active_chargers
                # len(c.charging_vehicles) / num_charging_vehicles * grid_capacity
    else:
        for c in charging_stations:
            c.available_power = c.power


def heuristic_power_scaling(charging_station, strategy='highest_power'):
    """
    It shares the available power of a charging station among the charging vehicles connected to it
    :param charging_station:
    :param strategy:
    :return: charging power for a vehicle
    """
    if strategy == 'highest_power':
        number_of_charging_vehicles = len([x for x in charging_station.connected_vehicles if x.mode == 'Charging'])
        power = charging_station.available_power / number_of_charging_vehicles
        for vehicle in charging_station.connected_vehicles:
            vehicle.charging_power = power
    return power'''
