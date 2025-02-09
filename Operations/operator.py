import random

import simpy
from Environment.helper.configuration.configuration import Configuration
import Operations.ChargingAlgorithms as charge_algos
import Operations.RoutingAlgorithms as route_algos
import Operations.IntegratedAlgorithms as integrate_algos
import Operations.StorageAlgorithms as store_algos
from Environment.log import lg
import numpy as np
import pandas as pd
import math

from Operations.NonLinearAlgorithms import nonlinear_pricing
from rl_pricing_env import convert_to_vector


class Operator:  # we also need a class for normal vehicles!!!

    def __init__(
        self,
        env,
        requests,
        chargers,
        routing_strategy,
        charging_strategy,
        storage_strategy,
        charging_capa,
        grid_capa,
        sim_time,
        electricity_tariff,
        connector_num,
        parking_spots,
        baseload,
        max_facility_baseload,
        non_dispatchable_generator,
        electric_storage,
        num_lookback_periods,
        planning_interval,
        optimization_period_length,
        num_lookahead_planning_periods,
        service_level,
        charging_hub,
        minimum_served_demand,
    ):

        self.env = env  # simulation environment
        self.sim_time = sim_time
        self.num_lookback_periods = num_lookback_periods  # how many pr sim periods to retrieve (history for prediction models)
        self.planning_interval = planning_interval
        self.optimization_period_length = optimization_period_length
        self.num_lookahead_planning_periods = num_lookahead_planning_periods  # how many planning periods do we look ahead?
        self.demand_threshold = 0  # min demand for serving request in kWh
        self.duration_threshold = (
            10000000  # min duration for serving request in sim periods (i.e., seconds)
        )
        self.routing_strategy = routing_strategy
        self.charging_strategy = charging_strategy
        self.storage_strategy = storage_strategy
        self.requests = requests
        self.chargers = chargers
        self.charging_capa = charging_capa
        self.grid_capa = grid_capa
        self.baseload = baseload
        self.non_dispatchable_generator = (
            non_dispatchable_generator  # this is an object
        )
        self.electric_storage = electric_storage  # this is an object
        self.free_grid_capa_actual = (
            0  # self.update_expected_free_grid_capacity()#0 #initialize as 0
        )
        self.free_grid_capa_predicted = (
            0  # self.update_expected_free_grid_capacity()#0 #initialize as 0
        )
        self.free_battery_load_capa = 0
        self.free_grid_capa_without_storage = 0
        self.peak_load_history = list(
            [int(max_facility_baseload)]
        )  # collects peak load over sim horizon (necessary to compute l_star), initialize with base facility peak load
        self.peak_load_history_inc_storage = list([int(max_facility_baseload)])
        self.electricity_tariff = electricity_tariff
        self.connector_num = connector_num
        self.parking_spots = parking_spots
        self.service_level = service_level
        self.arrival_event = env.event()
        self.routing_decision_event = env.event()
        self.charging_hub = charging_hub
        self.storage_object = charging_hub.electric_storage
        self.charging_agent = None
        self.storage_agent = None
        self.pricing_agent = None
        self.minimum_served_demand = minimum_served_demand
        self.agent_name = "PGMM"  # PGM or n_step
        self.action = np.zeros(121)
        self.storage_action = [0]
        self.pricing_action = None
        self.state = None
        self.pricing_state = None
        self.storage_state = None
        self.generation_min = 0
        self.peak_threshold = Configuration.instance().peak_threshold
        self.peak_cost = Configuration.instance().peak_cost
        self.energy_reward = 0
        self.objective = 0
        self.B2G = Configuration.instance().B2G
        self.price_pairs = (
            Configuration.instance().energy_prices
        )  # [minimum_average_power, price]
        self.price_history = pd.DataFrame()
        self.multiple_power = Configuration.instance().multiple_power
        self.parking_fee = Configuration.instance().parking_price
        self.pricing_parameters = Configuration.instance().price_parameters
        self.pricing_mode = Configuration.instance().pricing_mode
        self.organizer = simpy.Resource(self.env, capacity=1)
        if self.pricing_mode == "perfect_info":
            self.solve_pricing_with_perfect_info()
        if self.charging_strategy == "average_power":
            for i in requests:
                i.charging_power = i.energy_requested / (i.park_duration) * 60

    ##########################################
    # BELOW FUNCTIONS DERIVE DECISIONS

    # have this update automatically each planning_period
    def get_exp_free_grid_capacity(self):
        """
        Updates free grid capacity based on expected base load and non-dispatchable generation
        :param planning_period_length: length and planning window in sim periods
        :param num_lookahead_periods: number of planning periods for which plan is created
        :return:
        """
        # while True:
        current_time = self.env.now
        sim_time = self.sim_time
        final_time = min(
            sim_time,
            round(
                current_time
                + self.planning_interval * self.num_lookahead_planning_periods
            ),
        )

        # get periods in lookahead_window
        periods = []
        t = current_time
        while t < final_time:
            periods.append(t)
            t += self.planning_interval

        # get list of free grid capa
        free_capa_list_actual = []
        base_load_list = []
        generation_list = []
        free_capa_list_predicted = []

        for t in periods:
            # ACTUAL
            baseload_max = max(
                self.baseload.loc[t : t + self.planning_interval - 1][
                    "load_kw_rescaled"
                ]
            )
            self.baseload_min = min(
                self.baseload.loc[t : t + self.planning_interval - 1][
                    "load_kw_rescaled"
                ]
            )
            self.baseload_max = baseload_max
            generation_min = min(
                self.non_dispatchable_generator.generation_profile_actual.loc[
                    t : t + self.planning_interval - 1
                ]["pv_generation"]
            )
            self.generation_min = generation_min
            # TODO: there is a bug in here--> parallel charging and storage
            battery_max = min(
                self.electric_storage.kW_discharge_peak,
                (
                    (
                        self.electric_storage.SoC
                        - self.electric_storage.min_energy_stored_kWh
                    )
                    * (60 / self.planning_interval)
                ),
            )

            if self.charging_strategy in [
                "dynamic",
                "integrated_storage",
                "online_multi_period",
            ]:
                battery_max = 0
            battery_usage = 0
            if self.charging_hub.dynamic_pricing:

                battery_usage = (
                    self.electric_storage.discharging_power
                    - self.electric_storage.charging_power
                )
            free_capa_list_actual.append(
                self.grid_capa
                - baseload_max
                + generation_min
                + battery_max
                + battery_usage
            )
            # PREDICTED
            offset_period = (
                self.num_lookback_periods
            )  # how many periods to go back for prediction, here we fix to 1 day
            # TODO: This should be attached to peripheral_building.py object
            baseload_max_pred = max(
                self.baseload.loc[
                    (t - offset_period) : (t - offset_period)
                    + (self.planning_interval - 1)
                ]["load_kw_rescaled"]
            )
            generation_min_pred = min(
                self.non_dispatchable_generator.generation_profile_forecast.loc[
                    t : t + self.planning_interval - 1
                ]["pv_generation"]
            )
            battery_max = min(
                self.electric_storage.kW_discharge_peak,
                (
                    (
                        self.electric_storage.SoC
                        - self.electric_storage.min_energy_stored_kWh
                    )
                    * (60 / self.planning_interval)
                ),
            )
            free_capa_list_predicted.append(
                self.grid_capa
                - baseload_max_pred
                + generation_min_pred
                + battery_max
                + battery_usage
            )
            base_load_list.append(baseload_max_pred)
            generation_list.append(generation_min_pred)

        # update free_grid_capa
        self.free_grid_capa_actual = free_capa_list_actual
        self.base_load_list = base_load_list
        self.generation_list = generation_list
        self.free_grid_capa_without_storage = (
            free_capa_list_actual[0] - battery_max - battery_usage
        )
        self.free_grid_capa_predicted = free_capa_list_predicted

        # return free_capa_list
        # yield self.env.timeout(self.planning_period_length)

    def get_available_battery_load(self):
        """
        Updates available battery capacity for planning period as minimum of peak discharge rate and
        load that discharges SoC over planning period.
        :return:
        """
        if self.storage_object.max_energy_stored_kWh > 0:
            return
        # max_remaining_charge = self.electric_storage.max_energy_stored_kWh - self.electric_storage.SoC
        max_remaining_discharge = (
            self.electric_storage.SoC - self.electric_storage.min_energy_stored_kWh
        )

        battery_max = min(
            self.electric_storage.kW_discharge_peak,
            (max_remaining_discharge * 60 / self.planning_interval),
        )

        self.free_battery_load_capa = battery_max

    # pricing for different power usages

    def get_power_prices(self, mode):
        """
        Routes new arrivals to charging stations. This is on a discrete event basis, i.e. per each arrival (as opposed to the ca)
        :param routing_strategy:
        :return:
        """
        while True:
            self.get_exp_free_grid_capacity()
            self.update_vehicles_status()
            self.take_pricing_action()
            self.price_history = pd.concat(
                [self.price_history, pd.DataFrame(self.price_pairs[:, 1]).transpose()]
            )
            self.update_peak_load_history()
            if mode == "discrete_time":
                yield self.env.timeout(self.planning_interval)
            if mode == "discrete_event":
                yield self.arrival_event
            self.update_pricing_agent()

    # routing
    def get_routing_instructions(self, request):
        """
        Routes new arrivals to charging stations. This is on a discrete event basis, i.e. per each arrival (as opposed to the ca)
        :param routing_strategy:
        :return:
        """
        if self.routing_strategy == "random":
            charger = route_algos.random_charger_assignment(
                charging_stations=self.chargers,
                number_of_connectors=self.connector_num,
                request=request,
                demand_threshold=self.demand_threshold,
                duration_threshold=self.duration_threshold,
            )
        if self.routing_strategy == "lowest_occupancy_first":
            charger = route_algos.lowest_occupancy_first_charger_assignment(
                charging_stations=self.chargers,
                number_of_connectors=self.connector_num,
                request=request,
                demand_threshold=self.demand_threshold,
                duration_threshold=self.duration_threshold,
            )
        if self.routing_strategy == "fill_one_after_other":
            charger = route_algos.fill_one_after_other_charger_assignment(
                charging_stations=self.chargers,
                number_of_connectors=self.connector_num,
                request=request,
                demand_threshold=self.demand_threshold,
                duration_threshold=self.duration_threshold,
            )
        if self.routing_strategy == "lowest_utilization_first":
            charger = route_algos.lowest_utilization_first_charger_assignment(
                charging_stations=self.chargers,
                number_of_connectors=self.connector_num,
                request=request,
                demand_threshold=self.demand_threshold,
                duration_threshold=self.duration_threshold,
            )
        if self.routing_strategy == "perfect_info":
            charger = request.assigned_charger
        if self.routing_strategy == "perfect_info_with_storage":
            charger = request.assigned_charger
        if self.routing_strategy == "matching_supply_demand":
            charger = route_algos.matching_supply_demand_level(
                charging_stations=self.chargers, request=request
            )
        if self.routing_strategy == "minimum_power_requirement":
            charger = route_algos.assign_to_the_minimum_power(
                charging_stations=self.chargers, request=request
            )
        return charger

    # charging
    def get_charging_schedules_and_prices(self, charging_strategy, mode):
        """
        Periodically updates charging schedule based on selected strategy. Decides which vehicle charges and how much!
        This is on a discrete time basis
        :param scheduling_mode: simulation mode (discrete-time or discrete-event)
        :param charging_strategy:
        :param planning_period_length: length of period (in unit sim time). Schedule is re-computed every n(=period_length) time steps
        :return: n/a
        """
        first_scheduling = False
        while True:
            if charging_strategy == "perfect_info":
                if first_scheduling == False:
                    self.get_exp_free_grid_capacity()
                    connected_vehicles = [
                        x for x in self.requests if x.mode is None and x.ev == 1
                    ]
                    integrate_algos.perfect_info_charging_routing(
                        vehicles=connected_vehicles,
                        charging_stations=self.chargers,
                        env=self.env,
                        grid_capacity=self.free_grid_capa_actual,
                        electricity_cost=self.electricity_tariff,
                        baseload=self.base_load_list,
                        sim_time=self.sim_time,
                        generation=self.generation_list,
                    )
                    first_scheduling = True
                hour = int((self.env.now) / 60)
                for request in self.requests:
                    if request.ev == 1:
                        request.charging_power = request.charge_schedule[hour]
                if charging_strategy == "perfect_info_with_storage":
                    if first_scheduling == False:
                        self.get_exp_free_grid_capacity()
                        connected_vehicles = [
                            x for x in self.requests if x.mode is None and x.ev == 1
                        ]
                        if len(connected_vehicles) > 0:
                            integrate_algos.perfect_info_charging_routing_storage(
                                vehicles=connected_vehicles,
                                charging_stations=self.chargers,
                                env=self.env,
                                grid_capacity=self.free_grid_capa_actual,
                                electricity_cost=self.electricity_tariff,
                                sim_time=self.sim_time,
                                storage=self.electric_storage,
                                baseload=self.baseload_list,
                            )
                        first_scheduling = True
                    hour = int((self.env.now % 1440) / 60)
                    for request in self.requests:
                        if request.ev == 1:
                            request.charging_power = request.charge_schedule[hour]
                    if self.storage_object.max_capacity_kWh > 0:
                        storage_power = self.electric_storage.charge_schedule[hour]
                        if storage_power >= 0:
                            self.electric_storage.charge_yn = 1
                            self.electric_storage.discharge_yn = 0
                            self.electric_storage.discharging_power = 0
                            self.electric_storage.charging_power = storage_power
                        else:
                            self.electric_storage.charge_yn = 0
                            self.electric_storage.discharge_yn = 1
                            self.electric_storage.discharging_power = storage_power
                            self.electric_storage.charging_power = 0

            if self.charging_hub.dynamic_pricing:
                self.get_exp_free_grid_capacity()
                self.update_vehicles_status()
                self.take_pricing_action()
                if self.pricing_mode == "Discrete":
                    self.price_history = pd.concat(
                        [
                            self.price_history,
                            pd.DataFrame(self.price_pairs[:, 1]).transpose(),
                        ]
                    )
                if self.pricing_mode == "Continuous":
                    # print([self.pricing_parameters[1], self.parking_fee])
                    self.price_history = pd.concat(
                        [
                            self.price_history,
                            pd.DataFrame(
                                [self.pricing_parameters[0], self.pricing_parameters[1]]
                            ).transpose(),
                        ]
                    )

            else:
                if self.pricing_mode == "ToU":
                    hour = int((self.env.now % 1440) / 60)
                    self.pricing_parameters[0] = (
                        self.electricity_tariff[hour]
                        / max(self.electricity_tariff)
                        * Configuration.instance().max_price_ToU
                    )
                    # print(self.pricing_parameters[0])
                # for i in range(len(self.price_pairs[:,1])):
                #     self.price_pairs[i,1] = Configuration.instance().prices[i]
                # self.pricing_parameters[1] = 0.002 + ((23 - hour) / 600)
                if self.pricing_mode == "perfect_info":
                    if Configuration.instance().dynamic_fix_term_pricing:
                        self.pricing_parameters[1] = self.price_schedules[1][hour]
                        self.pricing_parameters[0] = self.price_schedules[0][hour]
                    else:
                        self.pricing_parameters[1] = self.price_schedules[hour]
                if self.pricing_mode == "Discrete":
                    self.price_history = pd.concat(
                        [
                            self.price_history,
                            pd.DataFrame(self.price_pairs[:, 1]).transpose(),
                        ]
                    )
                if self.pricing_mode == "Continuous" or self.pricing_mode == "ToU":
                    self.price_history = pd.concat(
                        [
                            self.price_history,
                            pd.DataFrame(
                                [self.pricing_parameters[0], self.pricing_parameters[1]]
                            ).transpose(),
                        ]
                    )

            # if len([x for x in self.requests if
            #         x.mode == 'Connected']) > 0:  # only execute if there are connected vehicles

            # Charging algos that DO NOT require foresight

            if charging_strategy == "uncontrolled":
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                charge_algos.uncontrolled(
                    env=self.env,
                    connected_vehicles=connected_vehicles,
                    charging_capacity=self.charging_capa,
                    planning_period_length=self.planning_interval,
                )

            if charging_strategy == "average_power":
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                charge_algos.average_power(
                    env=self.env,
                    connected_vehicles=connected_vehicles,
                    charging_capacity=self.charging_capa,
                    planning_period_length=self.planning_interval,
                    free_grid_capacity=self.free_grid_capa_actual,
                )

            if charging_strategy == "first_come_first_served":
                self.get_exp_free_grid_capacity()
                self.get_available_battery_load()
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                charge_algos.first_come_first_served(
                    env=self.env,
                    connected_vehicles=connected_vehicles,
                    charging_stations=self.chargers,
                    charging_capacity=self.charging_capa,
                    free_grid_capacity=self.free_grid_capa_actual,
                    # free_battery_capacity = self.free_battery_load_capa,
                    planning_period_length=self.planning_interval,
                )

            if charging_strategy == "earliest_deadline_first":
                self.get_exp_free_grid_capacity()
                self.get_available_battery_load()
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                charge_algos.earliest_deadline_first(
                    env=self.env,
                    connected_vehicles=connected_vehicles,
                    charging_stations=self.chargers,
                    charging_capacity=self.charging_capa,
                    free_grid_capacity=self.free_grid_capa_actual,
                    # free_battery_capacity=self.free_battery_load_capa,
                    planning_period_length=self.planning_interval,
                )

            if charging_strategy == "least_laxity_first":
                self.get_exp_free_grid_capacity()
                self.get_available_battery_load()
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                # print([x.id for x in connected_vehicles])
                charge_algos.least_laxity_first(
                    env=self.env,
                    connected_vehicles=connected_vehicles,
                    charging_stations=self.chargers,
                    charging_capacity=self.charging_capa,
                    free_grid_capacity=self.free_grid_capa_actual,
                    # free_battery_capacity = self.free_battery_load_capa,
                    planning_period_length=self.planning_interval,
                )

            if charging_strategy == "equal_sharing":
                self.get_exp_free_grid_capacity()
                self.get_available_battery_load()
                charge_algos.equal_sharing(
                    charging_stations=self.chargers,
                    charging_capacity=self.charging_capa,
                    free_grid_capacity=self.free_grid_capa_actual,
                    free_battery_capacity=self.free_battery_load_capa,
                )

            if charging_strategy == "online_myopic":
                self.get_exp_free_grid_capacity()
                self.get_available_battery_load()
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                charge_algos.online_myopic(
                    vehicles=connected_vehicles,
                    charging_stations=self.chargers,
                    env=self.env,
                    grid_capacity=self.free_grid_capa_actual,
                    optimization_period_length=self.optimization_period_length,
                    alpha=0,
                )

            # Charging algos that DO require foresight

            if charging_strategy == "online_multi_period":
                self.get_available_battery_load()
                self.get_exp_free_grid_capacity()
                if max(self.charging_hub.grid.grid_usage) > self.peak_threshold:
                    self.peak_threshold = max(self.charging_hub.grid.grid_usage)
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                if len(connected_vehicles) > 0:
                    charge_algos.online_multi_period(
                        vehicles=connected_vehicles,
                        charging_stations=self.chargers,
                        env=self.env,
                        free_grid_capa_actual=self.free_grid_capa_actual,
                        free_grid_capa_predicted=self.free_grid_capa_predicted,
                        peak_load_history=self.peak_load_history,
                        electricity_cost=self.electricity_tariff,
                        sim_time=self.sim_time,
                        service_level=self.service_level,
                        optimization_period_length=self.optimization_period_length,
                        num_lookahead_planning_periods=4,
                        flex_margin=0.5,
                        peak_threshold=self.peak_threshold,
                    )

            if charging_strategy == "integrated_storage":
                self.get_exp_free_grid_capacity()
                self.get_available_battery_load()
                connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
                if len(connected_vehicles) > 0:
                    charge_algos.integrated_charging_storage(
                        storage=self.electric_storage,
                        vehicles=connected_vehicles,
                        charging_stations=self.chargers,
                        env=self.env,
                        free_grid_capa_actual=self.free_grid_capa_actual,
                        free_grid_capa_predicted=self.free_grid_capa_predicted,
                        peak_load_history=self.peak_load_history,
                        electricity_cost=self.electricity_tariff,
                        sim_time=self.sim_time,
                        service_level=self.service_level,
                        optimization_period_length=self.optimization_period_length,
                        num_lookahead_planning_periods=12,
                        flex_margin=0.5,
                    )

            if charging_strategy == "dynamic":

                self.get_exp_free_grid_capacity()
                self.update_vehicles_status()
                self.take_action()
                self.conduct_charging_action()

                ### active these lines if we have separate battery agent
                # self.get_exp_free_grid_capacity()
                # self.take_storage_action()
                # self.conduct_storage_action()

            # update peak load history
            self.update_peak_load_history()

            # yield until next planning period
            if mode == "discrete_time":
                yield self.env.timeout(self.planning_interval)
            if mode == "discrete_event":
                yield self.arrival_event
            if charging_strategy == "dynamic":
                # if len(connected_vehicles) > 0:
                self.update_agent()
                ### active these lines if we have separate battery agent
                # self.update_storage_agent()
            if self.charging_hub.dynamic_pricing:
                self.update_pricing_agent()

    def take_action(self):
        self.state = self.charging_hub.charging_agent.environment.get_state(
            self.charging_hub, self.env
        )
        self.charging_agent.state = self.state

        eval_ep = self.charging_agent.do_evaluation_iterations
        self.charging_agent.episode_step_number_val = 0
        # while not self.done:
        action = self.charging_agent.pick_action(eval_ep, self.charging_hub)
        self.charging_agent.action = self.charging_agent.rescale_action(action)
        self.action = self.charging_agent.action

    def take_pricing_action(self):
        self.pricing_state = self.pricing_agent.environment.get_state(
            self.charging_hub, self.env
        )
        self.pricing_agent.state = self.pricing_state

        eval_ep = self.pricing_agent.do_evaluation_iterations

        if Configuration.instance().pricing_mode == "Discrete":
            # self.pricing_agent.episode_step_number_val = 0
            # while not self.done:
            if self.pricing_agent.agent_name == "DQN":
                self.pricing_agent.action = self.pricing_agent.pick_action()

            if self.pricing_agent.agent_name == "SAC":
                self.pricing_agent.action = self.pricing_agent.pick_action(
                    eval_ep, self.charging_hub
                )

            self.pricing_action = self.pricing_agent.action

            if self.pricing_agent.agent_name == "SAC":
                rescaled_actions = self.pricing_agent.environment.rescale_action(
                    self.pricing_action
                )
                number_of_power_options = len(self.price_pairs[:, 1])
                final_pricing = rescaled_actions[0:number_of_power_options]
                # for i in range(len(final_pricing)):
                self.price_pairs[0, 1] = final_pricing[0]
                self.price_pairs[1, 1] = min(final_pricing[1], 1.5)
                # if Configuration.instance().limiting_grid_capa:
                #     self.grid_capa = rescaled_actions[number_of_power_options]
                # if len(rescaled_actions) >= number_of_power_options + 2:
                #     self.storage_action = [rescaled_actions[number_of_power_options+1]]
                # self.conduct_storage_action()
            if self.pricing_agent.agent_name == "DQN":
                if len(self.price_pairs[:, 1]) > 1:
                    vector_prices = convert_to_vector(self.pricing_action)
                else:
                    vector_prices = [self.pricing_action]
                final_pricing = self.pricing_agent.environment.get_final_prices_DQN(
                    vector_prices
                )
                for i in range(len(final_pricing)):
                    self.price_pairs[i, 1] = final_pricing[i]

        if Configuration.instance().pricing_mode == "Continuous":
            self.pricing_agent.action = self.pricing_agent.pick_action(
                eval_ep, self.charging_hub
            )
            self.pricing_action = self.pricing_agent.action
            rescaled_actions = self.pricing_agent.environment.rescale_action(
                self.pricing_action
            )
            # self.pricing_parameters[0] = rescaled_actions[0]
            if (
                not Configuration.instance().dynamic_fix_term_pricing
                and Configuration.instance().capacity_pricing
            ):
                self.pricing_parameters[1] = rescaled_actions[0]
            if (
                Configuration.instance().dynamic_fix_term_pricing
                and not Configuration.instance().capacity_pricing
            ):
                self.pricing_parameters[0] = rescaled_actions[0]
                if Configuration.instance().dynamic_parking_fee:
                    self.parking_fee = rescaled_actions[1]
            if (
                Configuration.instance().dynamic_fix_term_pricing
                and Configuration.instance().capacity_pricing
            ):
                self.pricing_parameters[0] = rescaled_actions[0]
                self.pricing_parameters[1] = rescaled_actions[1]
            if Configuration.instance().limiting_grid_capa:

                self.grid_capa = rescaled_actions[1]
            if Configuration.instance().dynamic_storage_scheduling:
                self.storage_action = [rescaled_actions[1]]
            self.conduct_storage_action()

        self.charging_hub.grid.reset_reward()

    def take_storage_action(self):
        self.storage_state = self.charging_hub.storage_agent.environment.get_state(
            self.charging_hub, self.env
        )
        self.storage_agent.state = self.storage_state

        # self.get_battery_max_min()

        eval_ep = self.storage_agent.do_evaluation_iterations
        self.storage_agent.episode_step_number_val = 0
        # while not self.done:
        self.storage_agent.action = self.storage_agent.pick_action(
            eval_ep, self.charging_hub
        )
        self.storage_action = self.storage_agent.action

    def get_battery_max_min(self):
        bound_1 = (
            (self.storage_object.max_energy_stored_kWh - self.storage_object.SoC)
            * 60
            / self.charging_hub.planning_interval
        )
        bound_2 = self.charging_hub.operator.free_grid_capa_actual[0]
        bound_3 = self.electric_storage.kW_charge_peak
        charging_bound = min(bound_1, bound_2, bound_3)
        self.charging_hub.max_battery_charging_rate = charging_bound
        hub_generation_kW, hub_demand_kW, max_grid_capa = (
            self.get_hub_generation_kW(),
            self.get_hub_load_kW(),
            self.grid_capa,
        )
        bound_1 = -(hub_demand_kW - hub_generation_kW)
        bound_2 = -(self.storage_object.SoC) * 60 / self.charging_hub.planning_interval
        bound_3 = -self.electric_storage.kW_discharge_peak
        if not self.B2G:
            discharging_bound = max(bound_2, bound_3)
        else:
            discharging_bound = max(bound_1, bound_2, bound_3)
        self.charging_hub.max_battery_discharging_rate = discharging_bound
        self.storage_agent.action_range = [discharging_bound, charging_bound]
        # print(self.charging_hub.max_battery_charging_rate, self.electric_storage.SoC ,self.charging_hub.max_battery_discharging_rate)

    def check_storage(self):
        storage_power = self.storage_action[0]
        if storage_power >= 0:
            if (
                self.storage_object.SoC
                + storage_power / 60 * self.charging_hub.planning_interval
                > self.storage_object.max_energy_stored_kWh
            ):
                storage_power = (
                    (
                        self.storage_object.max_energy_stored_kWh
                        - self.storage_object.SoC
                    )
                    * 60
                    / self.charging_hub.planning_interval
                )
            storage_power = min(
                storage_power, self.charging_hub.operator.free_grid_capa_without_storage
            )

            self.charging_hub.electric_storage.charge_yn = 1
            self.charging_hub.electric_storage.charging_power = storage_power
            self.charging_hub.electric_storage.discharge_yn = 0
            self.charging_hub.electric_storage.discharging_power = 0
        hub_generation_kW, hub_demand_kW, max_grid_capa = (
            self.get_hub_generation_kW(),
            self.get_hub_load_kW(),
            self.grid_capa,
        )
        if storage_power < 0:
            if not self.B2G:
                if storage_power + hub_demand_kW - hub_generation_kW < 0:
                    storage_power = -(hub_demand_kW - hub_generation_kW)
            if (
                self.storage_object.SoC
                + (storage_power / 60 * self.charging_hub.planning_interval)
                < 0
            ):
                storage_power = -max(
                    (self.storage_object.SoC)
                    * 60
                    / self.charging_hub.planning_interval,
                    0,
                )
            if self.storage_object.SoC <= 0:
                storage_power = 0

            self.charging_hub.electric_storage.charge_yn = 0
            self.charging_hub.electric_storage.charging_power = 0
            self.charging_hub.electric_storage.discharge_yn = 1
            self.charging_hub.electric_storage.discharging_power = -storage_power
        self.charging_hub.reward["feasibility_storage"] += abs(
            self.storage_action[0] - storage_power
        )
        # print(self.charging_hub.electric_storage.charging_power, self.charging_hub.electric_storage.discharging_power)
        # print(self.storage_action[0], storage_power, self.electric_storage.SoC)
        # self.storage_action[0] = storage_power

    def check_charging_power(self):
        evaluation = False  # self.charging_agent.do_evaluation_iterations
        # Checking storage action
        # First we ensure that the charging load is less that storage capacity and free grid power
        storage_power = 0  # self.action[0]
        # if storage_power >= 0:
        #     if self.storage_object.SoC + storage_power / 60 * self.charging_hub.planning_interval > \
        #             self.storage_object.max_energy_stored_kWh:
        #         storage_power = (
        #                                     self.storage_object.max_energy_stored_kWh - self.storage_object.SoC) * 60 / self.charging_hub.planning_interval
        #     storage_power = min(storage_power, self.charging_hub.operator.free_grid_capa_actual[0])
        #
        #     self.charging_hub.electric_storage.charge_yn = 1
        #     self.charging_hub.electric_storage.charging_power = storage_power
        #     self.charging_hub.electric_storage.discharge_yn = 0
        #     self.charging_hub.electric_storage.discharging_power = 0
        # hub_generation_kW, hub_demand_kW, max_grid_capa = self.get_hub_generation_kW(), self.get_hub_load_kW(), self.grid_capa
        # if storage_power < 0:
        #     if not self.B2G:
        #         if storage_power + hub_demand_kW - hub_generation_kW < 0:
        #             storage_power = - (hub_demand_kW - hub_generation_kW)
        #     if self.storage_object.SoC + (storage_power / 60 * self.charging_hub.planning_interval) < 0:
        #         storage_power = - max((self.storage_object.SoC) * 60 / self.charging_hub.planning_interval, 0)
        #     if self.storage_object.SoC <= 0:
        #         storage_power = 0
        #
        #     self.charging_hub.electric_storage.charge_yn = 0
        #     self.charging_hub.electric_storage.charging_power = 0
        #     self.charging_hub.electric_storage.discharge_yn = 1
        #     self.charging_hub.electric_storage.discharging_power = - storage_power
        # self.charging_hub.reward['feasibility_storage'] += abs(self.action[0] - storage_power)
        # # print(self.storage_action[0], storage_power, self.electric_storage.SoC)
        # if evaluation:
        #     self.action[0] = storage_power

        all_charging_vehicles = np.asarray([])
        i = 0
        for charger in self.charging_hub.chargers:
            charging_vehicles = charger.charging_vehicles
            charger_usage = 0
            charging_power_list = np.array(
                [v.charging_power for v in charging_vehicles]
            )
            for j in range(charger.number_of_connectors):
                try:
                    charging_vehicles[j].action_id = i + 1
                    if charging_vehicles[j].remaining_energy_deficit <= 0:
                        self.charging_hub.reward["feasibility"] += self.action[i + 1]
                        if evaluation:
                            self.action[i + 1] = 0
                except:
                    self.charging_hub.reward["feasibility"] += self.action[i + 1]
                    if evaluation:
                        self.action[i + 1] = 0
                i += 1

            for vehicle in charging_vehicles:
                if vehicle.remaining_energy_deficit <= 0:
                    vehicle.charging_power = 0
                charger_usage += vehicle.charging_power
                all_charging_vehicles = np.append(all_charging_vehicles, vehicle)
            self.charging_hub.reward["feasibility"] += max(
                charger_usage - charger.power, 0
            )
            while charging_power_list.sum() > charger.power:
                number_active_chargers = len(charging_power_list)
                surplus_per_charger = (
                    max(charging_power_list.sum() - charger.power + 0.1, 0)
                    / number_active_chargers
                )
                for vehicle in charging_vehicles:
                    vehicle.charging_power -= surplus_per_charger
                    vehicle.charging_power = max(vehicle.charging_power, 0)
                charging_power_list = np.array(
                    [v.charging_power for v in charging_vehicles]
                )

        total_charging_power_list = np.array(
            [v.charging_power for v in all_charging_vehicles]
        )
        grid_capa = min(self.action[0], self.free_grid_capa_actual[0])
        self.charging_hub.reward["feasibility"] += max(
            total_charging_power_list.sum() - grid_capa, 0
        )

        while total_charging_power_list.sum() + storage_power > grid_capa + 1:
            number_active_chargers = len(total_charging_power_list)
            surplus_per_charger = (
                max(total_charging_power_list.sum() + storage_power - grid_capa, 0)
                / number_active_chargers
            )
            for vehicle in all_charging_vehicles:
                vehicle.charging_power -= surplus_per_charger
                vehicle.charging_power = max(vehicle.charging_power, 0)
                if evaluation:
                    self.action[vehicle.action_id] = vehicle.charging_power
            total_charging_power_list = np.array(
                [v.charging_power for v in all_charging_vehicles]
            )
        if evaluation:
            self.charging_hub.reward["feasibility"] = 0

    def conduct_storage_action(self):
        storage_power = self.storage_action[0]
        if storage_power >= 0:
            self.charging_hub.electric_storage.charge_yn = 1
            self.charging_hub.electric_storage.charging_power = storage_power
            self.charging_hub.electric_storage.discharge_yn = 0
            self.charging_hub.electric_storage.discharging_power = 0
        elif storage_power < 0:
            self.charging_hub.electric_storage.charge_yn = 0
            self.charging_hub.electric_storage.charging_power = 0
            self.charging_hub.electric_storage.discharge_yn = 1
            self.charging_hub.electric_storage.discharging_power = -storage_power
        self.check_storage()

    def conduct_charging_action(self):
        action = self.action
        # storage_power = action[0]
        # if storage_power >= 0:
        #     self.charging_hub.electric_storage.charge_yn = 1
        #     self.charging_hub.electric_storage.charging_power = storage_power
        #     self.charging_hub.electric_storage.discharge_yn = 0
        #     self.charging_hub.electric_storage.discharging_power = 0
        # elif storage_power < 0:
        #     self.charging_hub.electric_storage.charge_yn = 0
        #     self.charging_hub.electric_storage.charging_power = 0
        #     self.charging_hub.electric_storage.discharge_yn = 1
        #     self.charging_hub.electric_storage.discharging_power = - storage_power
        i = 0
        for charger in self.charging_hub.chargers:
            charging_vehicles = charger.charging_vehicles
            for connector in range(charger.number_of_connectors):
                if action[i + 1] > 0:
                    try:
                        charging_vehicles[connector].charging_power = action[i + 1]
                    except:
                        pass
                i += 1
        self.check_charging_power()
        self.charging_hub.grid.reset_reward()

    def update_pricing_agent(self):
        self.update_vehicles_status()
        if not self.charging_agent:
            self.charging_hub.reward["missed"] = (
                self.reward_computing()
            )  # TODO: do we need to recalculate it?

        if self.pricing_agent.agent_name == "new_SAC":
            if len(self.pricing_agent.memory) > self.pricing_agent.config.batch_size:
                # Number of updates per step in environment
                for i in range(self.pricing_agent.config.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = (
                        self.pricing_agent.update_parameters(
                            self.pricing_agent.memory,
                            self.pricing_agent.batch_size,
                            self.pricing_agent.updates,
                        )
                    )
                    self.pricing_agent.updates += 1

            next_state, reward, done, _ = self.pricing_agent.environment.step(
                self.pricing_agent.action, self.charging_hub, self.env
            )
            # print(self.pricing_agent.action)
            mask = (
                False
                if self.pricing_agent.global_step_number
                >= self.pricing_agent.environment._max_episode_steps
                else self.pricing_agent.done
            )
            self.pricing_agent.memory.push(
                self.pricing_state, self.pricing_action, reward, next_state, mask
            )
            self.pricing_agent.state = next_state

        # SAC
        if self.pricing_agent.agent_name == "SAC":
            self.pricing_agent.conduct_action(
                self.pricing_action, self.charging_hub, self.env
            )
            eval_ep = self.pricing_agent.do_evaluation_iterations
            if self.pricing_agent.time_for_critic_and_actor_to_learn():
                if not eval_ep:
                    for _ in range(
                        self.pricing_agent.hyperparameters[
                            "learning_updates_per_learning_session"
                        ]
                    ):
                        self.pricing_agent.learn()
            mask = (
                False
                if self.pricing_agent.global_step_number
                >= self.pricing_agent.environment._max_episode_steps
                else self.pricing_agent.done
            )
            # if not eval_ep:
            action = self.pricing_action
            # action = self.pricing_agent.descale_action(self.pricing_action, self.charging_hub)
            self.pricing_agent.save_experience(
                experience=(
                    self.pricing_state,
                    action,
                    self.pricing_agent.reward,
                    self.pricing_agent.next_state,
                    mask,
                )
            )

        if self.pricing_agent.agent_name == "DQN":
            self.pricing_agent.conduct_action(
                self.pricing_action, self.charging_hub, self.env
            )
            if self.pricing_agent.time_for_q_network_to_learn():
                for _ in range(
                    self.pricing_agent.hyperparameters["learning_iterations"]
                ):
                    self.pricing_agent.learn()
            action = self.pricing_action
            self.pricing_agent.save_experience(
                experience=(
                    self.pricing_state,
                    action,
                    self.pricing_agent.reward,
                    self.pricing_agent.next_state,
                    False,
                )
            )

        self.pricing_agent.global_step_number += 1

    def update_storage_agent(self):

        eval_ep = self.storage_agent.do_evaluation_iterations
        action = self.storage_agent.descale_action(
            self.storage_action, self.charging_hub
        )
        self.storage_agent.conduct_action(action, self.charging_hub, self.env)
        if self.storage_agent.time_for_critic_and_actor_to_learn():
            for _ in range(
                self.storage_agent.hyperparameters[
                    "learning_updates_per_learning_session"
                ]
            ):
                self.storage_agent.learn()
        mask = (
            False
            if self.storage_agent.episode_step_number_val
            >= self.storage_agent.environment._max_episode_steps
            else self.storage_agent.done
        )
        if mask:
            print("there is problem with mask")
        # if not eval_ep:

        self.storage_agent.save_experience(
            experience=(
                self.storage_state,
                action,
                self.storage_agent.reward,
                self.storage_agent.next_state,
                mask,
            )
        )
        self.storage_agent.global_step_number += 1
        self.storage_agent.step_counter += 1

    def update_agent(self):
        self.update_vehicles_status()
        self.charging_hub.reward["missed"] = self.reward_computing()

        eval_ep = self.charging_agent.do_evaluation_iterations
        self.charging_agent.conduct_action(self.action, self.charging_hub, self.env)
        if self.charging_agent.time_for_critic_and_actor_to_learn():
            if not eval_ep:
                for _ in range(
                    self.charging_agent.hyperparameters[
                        "learning_updates_per_learning_session"
                    ]
                ):
                    self.charging_agent.learn()
        mask = (
            False
            if self.charging_agent.episode_step_number_val
            >= self.charging_agent.environment._max_episode_steps
            else self.charging_agent.done
        )
        if mask:
            print("there is problem with mask")
        # if not eval_ep:
        action = self.charging_agent.descale_action(self.action, self.charging_hub)
        self.charging_agent.save_experience(
            experience=(
                self.state,
                action,
                self.charging_agent.reward,
                self.charging_agent.next_state,
                mask,
            )
        )
        self.charging_agent.global_step_number += 1
        self.charging_agent.step_counter += 1

    def get_storage_schedule(self, storage_strategy, mode):
        """
        Get schedule for battery ops (receives charge schedules, PV schedules and tariff info, etc., optimizes accordingly)
        :return:
        """

        # if storage_strategy == 'testing':
        while True:
            # get charging load
            t = self.env.now
            ev_charging_load = sum(
                [x.charging_power for x in self.requests if x.mode == "Connected"]
            )  # previously defined by charging algo
            # Here we use actuals but since is highly predictable it should be fine
            max_base_load = max(
                self.baseload.loc[t : t + self.planning_interval - 1][
                    "load_kw_rescaled"
                ]
            )
            min_PV_generation = min(
                self.non_dispatchable_generator.generation_profile_actual.loc[
                    t : t + self.planning_interval - 1
                ]["pv_generation"]
            )

            if storage_strategy == "uncontrolled":
                store_algos.uncontrolled(
                    env=self.env, storage_object=self.electric_storage
                )

            if storage_strategy == "temporal_arbitrage":
                store_algos.temporal_arbitrage(
                    env=self.env,
                    storage_object=self.electric_storage,
                    planning_interval=self.planning_interval,
                    electricity_tariff=self.electricity_tariff,
                    free_grid_capacity=self.free_grid_capa_without_storage,
                    ev_charging_load=ev_charging_load,
                )

            if storage_strategy == "peak_shaving":
                store_algos.peak_shaving(
                    env=self.env,
                    storage_object=self.electric_storage,
                    planning_interval=self.planning_interval,
                    free_grid_capacity=self.free_grid_capa_actual,
                    ev_charging_load=ev_charging_load,
                    max_base_load=max_base_load,
                    min_PV_generation=min_PV_generation,
                    peak_history_inc_storage=self.peak_load_history_inc_storage,
                )

            # update load history
            self.update_peak_load_history_inc_storage()

            # yield until next planning period
            if mode == "discrete_time":
                yield self.env.timeout(self.planning_interval)
            if mode == "discrete_event":
                yield self.arrival_event

    def update_peak_load_history(self):
        """
        Calculates peak load in planning period and appends to peak_load history
        :param self:
        :return:
        """
        # LOAD SOURCES AND SINKS

        t = self.env.now
        charging_load = sum(
            [x.charging_power for x in self.requests if x.mode == "Connected"]
        )
        baseload_max = max(
            self.baseload.loc[t : t + self.planning_interval - 1]["load_kw_rescaled"]
        )
        generation_min = min(
            self.non_dispatchable_generator.generation_profile_actual.loc[
                t : t + self.planning_interval - 1
            ]["pv_generation"]
        )
        # battery_charge = self.electric_storage.charge_yn * self.electric_storage.charging_power
        # battery_discharge = self.electric_storage.discharge_yn * self.electric_storage.discharging_power

        planning_window_peak_load = charging_load + baseload_max - generation_min
        # planning_window_peak_load = charging_load + baseload_max + battery_charge - generation_min - battery_discharge
        self.peak_load_history.append(planning_window_peak_load)

    def update_peak_load_history_inc_storage(self):
        """
        Calculates peak load in planning period and appends to peak_load history
        :param self:
        :return:
        """
        # LOAD SOURCES AND SINKS

        t = self.env.now
        charging_load = sum(
            [x.charging_power for x in self.requests if x.mode == "Connected"]
        )
        baseload_max = max(
            self.baseload.loc[t : t + self.planning_interval - 1]["load_kw_rescaled"]
        )
        generation_min = min(
            self.non_dispatchable_generator.generation_profile_actual.loc[
                t : t + self.planning_interval - 1
            ]["pv_generation"]
        )
        battery_charge = (
            self.electric_storage.charge_yn * self.electric_storage.charging_power
        )
        battery_discharge = (
            self.electric_storage.discharge_yn * self.electric_storage.discharging_power
        )

        planning_window_peak_load = (
            charging_load
            + baseload_max
            + battery_charge
            - generation_min
            - battery_discharge
        )
        self.peak_load_history_inc_storage.append(planning_window_peak_load)

        # print("Peak load planning window (post charging)",planning_window_peak_load)

    def get_hub_load_kW(self):
        """
        Retrieves total load in current period
        :param self:
        :return:
        """
        # TODO: GET FORECASTS FOR t+n

        t = self.env.now
        charging_load = sum(
            [x.charging_power for x in self.requests if x.mode == "Connected"]
        )
        baseload = self.baseload.loc[t]["load_kw_rescaled"]

        return charging_load + baseload

    def get_hub_generation_kW(self):
        """
        Retrieves total generation (i.e., PV) supply in current period
        :param self:
        :return:
        """
        # TODO: GET FORECASTS FOR t+n

        t = self.env.now

        generation_current_period = (
            self.non_dispatchable_generator.generation_profile_actual.loc[t][
                "pv_generation"
            ]
        )

        return generation_current_period

    ##########################################
    # BELOW FUNCTIONS EXECUTE DECISIONS

    def request_queueing(self):  # WHAT EXACTLY DOES THIS FUNCTION DO?
        while True:
            not_arrived_requests = [x for x in self.requests if x.mode is None]
            if len(not_arrived_requests) > 0:
                # request = not_arrived_requests[0]
                # interarrival_time = (request.arrival_period - self.env.now)
                # yield self.env.timeout(interarrival_time)
                # request.mode = 'Arrived'
                # if self.multiple_power:
                #     request.adjust_request_demand_based_on_pricing(self.price_pairs)
                # self.env.process(self.assign_parking_charging_resources(request))
                # self.env.process(self.request_process(request))
                request = not_arrived_requests[0]
                interarrival_time = request.arrival_period - self.env.now
                yield self.env.timeout(interarrival_time)
                requests = [
                    x for x in not_arrived_requests if x.arrival_period <= self.env.now
                ]
                for request in requests:
                    request.mode = "Arrived"
                    if self.multiple_power:
                        request.adjust_request_demand_based_on_pricing(
                            self.price_pairs, self.pricing_parameters, self.parking_fee
                        )
                    self.env.process(self.assign_parking_charging_resources(request))
                    for charger in self.charging_hub.chargers:
                        charger.status_update()
                    yield self.env.timeout(0.01)
                    self.env.process(self.request_process(request))
            else:
                return

    # SERVING A REQUEST --> We need to talk about this section as I am not quite sure about it.
    def assign_parking_charging_resources(self, request):
        """
        Executing the process of charging for EVs and parking for Non-EVs
        :param duration_threshold:
        :param demand_threshold:
        :param request: object of request
        """
        lg.info(f"Request {request.id} arrived at {self.env.now}")
        request.mode = "Arrived"

        # get charger for request
        charging_station = self.get_routing_instructions(request=request)

        if charging_station:
            with charging_station.connectors.request() as charging_req:
                charging_station.in_queue_vehicles.append(request)
                yield charging_req
                charging_station.in_queue_vehicles.remove(request)
                with self.parking_spots.request() as parking_req:
                    yield parking_req
                    charging_station.connected_vehicles.append(request)
                    # charging_req = charging_station.connectors.request()
                    # parking_req = self.parking_spots.request()
                    # yield charging_req and parking_req
                    request.assigned_charger = charging_station
                    request.is_assigned = True
                    request.mode = "Assigned"
                    lg.info(
                        f"Request {request.id} (EV={request.ev}; requested charge = {request.energy_requested} kW) assigned to charging station {request.assigned_charger.id}"
                    )
                    self.arrival_event.succeed()
                    self.arrival_event = self.env.event()
                    request.mode = "Connected"
                    request.assigned_time = self.env.now
                    request.waiting_time = (
                        request.assigned_time - request.arrival_period
                    )
                    lg.info(
                        f"Request {request.id} (EV={request.ev}; requested charge = {request.energy_requested} kW) got connected to charging station {request.assigned_charger.id}"
                    )
                    yield request.event_departure
                    # charging_station.connectors.release(charging_req)
                    # charging_req.cancel()
                    # self.parking_spots.release(parking_req)
                    # parking_req.cancel()
                    request.mode = "Left"
                    # request.assigned_charger = None
                    lg.info(
                        f"Request {request.id} got {request.energy_charged} with requested energy"
                        f" {request.energy_requested}"
                    )
                    charging_station.connected_vehicles.remove(request)

        else:
            request.is_assigned = False
            # lg.info('No charger assigned charger')

            with self.parking_spots.request() as req:
                yield req
                # lg.info(f'Request {request.id} starts parking')
                request.mode = "Parking"
                yield request.event_departure
                request.mode = "Left"

    def storage_process(self):
        """
        Executing the charging and discharging schedule of the storage
        :param request: object of request
        """
        while True:
            hub_generation_kW, hub_demand_kW, max_grid_capa = (
                self.get_hub_generation_kW(),
                self.get_hub_load_kW(),
                self.grid_capa,
            )

            self.electric_storage.deploy(
                B2G=self.B2G,
                hub_demand_kW=hub_demand_kW,
                hub_generation_kW=hub_generation_kW,
                max_grid_capa=max_grid_capa,
            )
            # print(self.electric_storage.charging_power, self.electric_storage.discharging_power, self.electric_storage.SoC)/
            yield self.env.timeout(1)

    # TODO: Physical execution of above decisions to be added to infrastrcuture objects
    def request_process(self, request):
        """
        Tracking the events of arrival, stop-charging and departure of each request (i.e., each vehicle)
        :param request: object of request
        """
        while True:
            # if request.arrival_period == self.env.now:
            # request.event_arrival.succeed()
            # request.event_arrival = self.env.event()
            # self.env.process(self.charging_parking_task(request))
            if Configuration.instance().remove_low_request_EVs:
                if request.energy_requested == 0:
                    request.mode = "Left"
                    request.event_departure.succeed()
                    return
            if request.ev == 1:
                if request.mode == "Connected":
                    # print(f'{request.id} is charging with power {request.charging_power}')
                    request.energy_charged += (
                        request.charging_power / 60
                    )  # sim unit time is minutes so need to divide by 60
                    request.calculate_profit_reward(
                        self.charging_hub.penalty_for_missed_kWh,
                        self.electricity_tariff,
                    )
                    if request.charging_power < 0:
                        lg.info(
                            f"charging power of {request.id} is negative{request.charging_power}"
                        )
                if (
                    request.mode == "Connected"
                    and request.energy_charged >= request.energy_requested
                ):
                    request.event_stop_charging.succeed()
                    request.event_stop_charging = self.env.event()
                    request.stop_charging_time = self.env.now
                    request.mode = "Fully_charged"
                    request.charging_power = 0
                    lg.info(f"Request {request.id} stopped charging at {self.env.now}")
                    lg.info(
                        f"Request {request.id} got {request.energy_charged} with requested energy"
                        f" {request.energy_requested}"
                    )
                    if request.energy_charged < 0:
                        lg.info(f"request.energy_charged is negative for {request.id}")
            if request.departure_period <= self.env.now:
                lg.info(f"Request {request.id} left at {self.env.now}")
                # self.charging_hub.reward['missed'] += self.request_reward_computing(request)
                if request.energy_charged < 0:
                    lg.info(f"request.energy_charged is negative for {request.id}")
                request.event_departure.succeed()
                return
            elif self.env.now == self.sim_time - 1:
                lg.info(f"Request {request.id} left at {self.env.now}")
                # self.charging_hub.reward['missed'] += self.request_reward_computing(request)
                if request.energy_charged < 0:
                    lg.info(f"request.energy_charged is negative for {request.id}")
                request.mode = "Left"
                request.event_departure.succeed()
                return
            yield self.env.timeout(1)

    ############################################################################
    # MONITOR EV CHARGING
    def request_monitoring(self, request):
        """
        Monitoring the state of charge and the mode of each request every time step
        """
        request.info["SOC"] = []
        request.info["mode"] = []
        request.info["charging_power"] = []
        while True:
            request.info["SOC"].append(request.energy_charged)
            request.info["mode"].append(request.mode)
            request.info["charging_power"].append(request.charging_power)
            yield self.env.timeout(1)

    def reward_computing(self):
        reward = 0

        if max(self.charging_hub.grid.grid_usage) > self.peak_threshold:
            reward += (
                max(self.charging_hub.grid.grid_usage) - self.peak_threshold
            ) * self.peak_cost
            self.peak_threshold = max(self.charging_hub.grid.grid_usage)
        new_objective = self.charging_hub.update_objective_function(self.peak_threshold)
        reward -= new_objective - self.objective
        self.objective = new_objective
        # print(reward, self.objective, self.peak_threshold)
        return reward

    def update_vehicles_status(self):
        for request in [
            request
            for request in self.requests
            if request.mode in ["Connected", "Fully_charged"]
        ]:
            request.update_status()

    def solve_pricing_with_perfect_info(self):
        first_pricing = False
        if first_pricing == False:
            self.get_exp_free_grid_capacity()
            connected_vehicles = [
                x
                for x in self.requests
                if x.mode is None and x.ev == 1 and x.energy_requested > 0
            ]
            # integrate_algos.perfect_info_pricing_charging_routing(vehicles=connected_vehicles,
            #                                               charging_stations=self.chargers, env=self.env,
            #                                               grid_capacity=self.free_grid_capa_actual,
            #                                               electricity_cost=self.electricity_tariff,
            #                                               baseload=self.base_load_list,
            #                                               sim_time=self.sim_time,
            #                                               generation=self.generation_list)
            solution = nonlinear_pricing(
                vehicles_list=connected_vehicles,
                electricity_cost=self.electricity_tariff,
                PV=self.generation_list,
                base_load=self.base_load_list,
                sim_time=self.sim_time,
            )
            if Configuration.instance().dynamic_fix_term_pricing:

                self.price_schedules = [solution["p_0"], solution["alpha"]]
            else:
                self.price_schedules = solution["alpha"]
            first_scheduling = True
            # hour = int((self.env.now) / 60)
            # for request in self.requests:
            #     if request.ev == 1:
            #         request.charging_power = request.charge_schedule[hour]
