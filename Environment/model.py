import simpy

from Environment.helper.configuration.configuration import Configuration
from Environment.log import lg
from Infrastructure.grid import GridCapacity
from Infrastructure.parking_lot import ParkingLot
from Operations.operator import Operator
import pandas as pd
from Infrastructure.ev_charger import EVCharger
from Infrastructure.electric_generator import NonDispatchableGenerator
from Infrastructure.electric_storage import ElectricStorage
import Utilities.visualization as viz
import Utilities.sim_input_processing as prep

from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.agents.actor_critic_agents.SAC import (
    SAC,
)
from Environment.helper.configuration.SAC_configuration import config
from SC_env import ChargingHubInvestmentEnv

# NOTE: unit sim time is defined as 1 minute real time!
# from Preferences.request_generator import RequestGenerator
from Preferences.vehicle import Vehicle
from Utilities.sim_output_processing import get_load_curve


class EVCC_Sim_Model:
    # There is only one instance of this class.
    __instance = None

    @staticmethod
    def instance() -> "EVCC_Sim_Model":
        """Static access method."""
        if EVCC_Sim_Model.__instance is None:
            raise Exception("World was not initialized")
        else:
            return EVCC_Sim_Model.__instance

    @staticmethod
    def init(
        env,
        search_engine,
        payment_engine,
        vehicle_assignment_engine,
        vehicle_choice_behavior,
        population_factory,
        reset=False,
    ):
        """Static access method."""
        if EVCC_Sim_Model.__instance is None or reset is True:
            EVCC_Sim_Model(
                env,
                search_engine,
                payment_engine,
                vehicle_assignment_engine,
                vehicle_choice_behavior,
                population_factory,
                reset,
            )
        return EVCC_Sim_Model.__instance

    def __init__(
        self,
        base_path,
        raw_output_save_path,
        visuals_save_path,
        cache_path,
        post_fix,
        env,
        sim_season,
        sim_start_date,
        day_types,
        sim_duration,
        facility_list,
        ev_share,
        demand_gen_approach,
        geography,
        limit_requests_to_capa,
        parking_capa,
        grid_capa,
        transformer_num,
        charging_capa,
        min_facility_baseload,
        max_facility_baseload,
        installed_capa_PV,
        installed_storage,
        charging_num,
        connector_num,
        electricity_tariff,
        prices,
        year,
        planning_interval,
        optimization_period_length,
        lookahead,
        lookback,
        routing_algo,
        charging_algo,
        storage_algo,
        scheduling_mode,
        service_level,
        minimum_served_demand,
        penalty_for_missed_kWh,
        planning,
        objective,
        charging_agent,
        storage_agent,
        pricing_agent,
        chargers_type="single",
    ):

        self.planning = planning
        self.objective = objective
        # path settings
        self.base_path = base_path
        self.raw_output_save_path = raw_output_save_path
        self.post_fix = post_fix
        self.visuals_save_path = visuals_save_path
        self.cache_path = cache_path
        # sim/env parameter settings
        self.env = env
        self.sim_season = sim_season
        self.sim_start_date = sim_start_date
        self.day_types = day_types
        self.sim_duration = sim_duration  # in days
        self.sim_time = sim_duration * 24 * 60  # in minutes
        self.facility_list = facility_list
        self.ev_share = ev_share
        self.geography = geography
        self.demand_gen_approach = demand_gen_approach
        self.limit_requests_to_capa = limit_requests_to_capa  # boolean of whether the limit parking requests to parking capacizy
        self.planning_interval = planning_interval  # at which intervals (in unit sim time) to do the routing/charging re-planning
        self.optimization_period_length = optimization_period_length  # how long a period is in the optimization (in unit sim time)
        self.lookahead = lookahead  # how many optimization periods ops algo looks ahead
        self.lookback = lookback  # how many pre-sim periods to include in data retrieval (this is the history)
        self.electricity_tariff = electricity_tariff
        self.year = year
        self.prices = prices
        self.prices["peak"] = Configuration.instance().peak_cost
        # infrastucture settings
        self.chargers_type = chargers_type
        self.charging_capa = charging_capa
        self.charging_num = charging_num
        if chargers_type in ["single"]:
            self.charging_capa = charging_capa["fast"]
            self.charging_num = charging_num["fast_one"]
        self.connector_num = connector_num
        self.min_facility_baseload = min_facility_baseload
        self.max_facility_baseload = max_facility_baseload
        self.parking_capa = parking_capa
        self.base_load = prep.get_sim_baseload_curve(
            base_path=self.base_path,
            cache_path=self.cache_path,
            sim_start_day=self.sim_start_date,
            num_lookback_periods=self.lookback,
            sim_duration=self.sim_duration,
            min_facility_baseload=self.min_facility_baseload,
            max_facility_baseload=self.max_facility_baseload,
        )
        self.transformer_num = transformer_num  # Number
        self.grid_capa = grid_capa + transformer_num * 200  # in Kw
        self.installed_capa_PV = installed_capa_PV  # in kW
        self.storage_capacity = installed_storage  # in kWh

        # charging requests
        self.request_data = prep.get_sim_charging_requests(
            base_path=self.base_path,
            cache_path=self.cache_path,
            demand_gen_approach=self.demand_gen_approach,
            limit_requests_to_capa=self.limit_requests_to_capa,
            parking_capacity=self.parking_capa,
            sim_start_day=self.sim_start_date,
            day_types=self.day_types,
            sim_duration=self.sim_duration,
            sim_seasons=self.sim_season,
            facility_list=self.facility_list,
            ev_share=self.ev_share,
            max_charge_rate=self.charging_capa,
            geography=self.geography,
        )
        # self.demand_factory = None#RequestGenerator(self.env)
        self.benchmarking = Configuration.instance().benchmarking
        self.requests = self.EVCC_sim_setup(
            self.request_data
        )  # vehicles are the requests
        # ops settings
        self.routing_algo = routing_algo
        self.charging_algo = charging_algo
        self.storage_algo = storage_algo
        self.scheduling_mode = scheduling_mode
        self.costs = dict(investment=0, operations=0)
        self.service_level = 0
        self.minimum_served_demand = minimum_served_demand
        self.penalty_for_missed_kWh = penalty_for_missed_kWh
        self.objective_function = 0
        self.total_energy_charged = 0
        self.reward = dict(costs=0, missed=0, feasibility=0, feasibility_storage=0)
        self.dynamic_pricing = Configuration.instance().dynamic_pricing
        self.operator = Operator(
            env=env,
            requests=self.requests,
            chargers=self.chargers,
            routing_strategy=self.routing_algo,
            charging_strategy=self.charging_algo,
            storage_strategy=self.storage_algo,
            charging_capa=self.charging_capa,
            grid_capa=self.grid_capa,
            non_dispatchable_generator=self.non_dispatchable_generator,
            electric_storage=self.electric_storage,
            sim_time=self.sim_time,
            electricity_tariff=self.electricity_tariff,
            connector_num=connector_num,
            parking_spots=self.parking_spots,
            baseload=self.base_load,
            max_facility_baseload=self.max_facility_baseload,
            planning_interval=self.planning_interval,
            optimization_period_length=self.optimization_period_length,
            num_lookahead_planning_periods=self.lookahead,
            num_lookback_periods=self.lookback,
            service_level=service_level,
            charging_hub=self,
            minimum_served_demand=minimum_served_demand,
        )
        # decision-making agents
        self.charging_agent = charging_agent
        if self.charging_agent:
            self.charging_agent.environment.state = (
                self.charging_agent.environment.get_state(self, self.env)
            )
            self.charging_agent.environment.env = self.env
            self.charging_agent.reset_game()

        self.pricing_agent = pricing_agent
        self.pricing_agent.environment.state = self.pricing_agent.environment.get_state(
            self, self.env
        )
        self.pricing_agent.environment.env = self.env
        self.pricing_agent.reset_game()

        if storage_agent:
            self.storage_agent = storage_agent
            self.storage_agent.environment.state = (
                self.storage_agent.environment.get_state(self, self.env)
            )
            self.storage_agent.environment.env = self.env
            self.storage_agent.reset_game()

        self.operator.charging_agent = charging_agent
        self.operator.storage_agent = storage_agent
        self.operator.pricing_agent = pricing_agent

        self.peak_threshold = Configuration.instance().peak_threshold

    ##############################################################
    # SETUP ENVIRONMENT

    def EVCC_sim_setup(self, request_data):

        # infrastaructur
        self.initialize_infrastructure()

        # TODO: This is just a hack to deal with requests with the same arrival period
        # request_data['EntryMinutesFromSimStart'] = request_data['EntryMinutesFromSimStart'].astype('int64')
        # request_data = request_data.drop_duplicates(subset=['EntryMinutesFromSimStart'])
        # request_data = request_data.reset_index(drop=True)

        # requests
        requests = self.initialize_vehicle_population(request_data)

        return requests

    ##############################################################
    # INITIALIZE

    def initialize_infrastructure(self):
        """
        Initializing the charging, parking and grid infrastructure
        """
        # parking spots
        self.parking_lot = ParkingLot(env=self.env, parking_capacity=100000)
        ParkingSpots = self.parking_lot.parking_spots

        # chargers
        Chargers = []
        if self.chargers_type not in ["single"]:
            id_indicator = 0
            for i in range(self.charging_num["fast_one"]):
                charger = EVCharger(
                    env=self.env,
                    id=id_indicator,
                    power=self.charging_capa["fast"],
                    period_length=self.planning_interval,
                    number_of_connectors=1,
                )
                id_indicator += 1
                Chargers.append(charger)
            for i in range(self.charging_num["fast_two"]):
                charger = EVCharger(
                    env=self.env,
                    id=id_indicator,
                    power=self.charging_capa["fast"],
                    period_length=self.planning_interval,
                    number_of_connectors=2,
                )
                id_indicator += 1
                Chargers.append(charger)
            for i in range(self.charging_num["fast_four"]):
                charger = EVCharger(
                    env=self.env,
                    id=id_indicator,
                    power=self.charging_capa["fast"],
                    period_length=self.planning_interval,
                    number_of_connectors=4,
                )
                id_indicator += 1
                Chargers.append(charger)
            for i in range(self.charging_num["slow_one"]):
                charger = EVCharger(
                    env=self.env,
                    id=id_indicator,
                    power=self.charging_capa["slow"],
                    period_length=self.planning_interval,
                    number_of_connectors=1,
                )
                id_indicator += 1
                Chargers.append(charger)
            for i in range(self.charging_num["slow_two"]):
                charger = EVCharger(
                    env=self.env,
                    id=id_indicator,
                    power=self.charging_capa["slow"],
                    period_length=self.planning_interval,
                    number_of_connectors=2,
                )
                id_indicator += 1
                Chargers.append(charger)
            for i in range(self.charging_num["slow_four"]):
                charger = EVCharger(
                    env=self.env,
                    id=id_indicator,
                    power=self.charging_capa["slow"],
                    period_length=self.planning_interval,
                    number_of_connectors=4,
                )
                id_indicator += 1
                Chargers.append(charger)
        else:
            for i in range(self.charging_num):
                charger = EVCharger(
                    env=self.env,
                    id=i,
                    power=self.charging_capa,
                    period_length=self.planning_interval,
                    number_of_connectors=self.connector_num,
                )
                Chargers.append(charger)

        # grid capacity
        Grid = GridCapacity(self.env, self.grid_capa)

        # PV capacity
        PV = NonDispatchableGenerator(
            env=self.env,
            kW_peak=self.installed_capa_PV,
            base_path=self.base_path,
            cache_path=self.cache_path,
            sim_start_day=self.sim_start_date,
            sim_duration=self.sim_duration,
            num_lookback_periods=self.lookback,
        )

        # electric storage
        Storage = ElectricStorage(env=self.env, max_capacity_kWh=self.storage_capacity)

        # save to objects
        (
            self.parking_spots,
            self.chargers,
            self.grid,
            self.electric_storage,
            self.non_dispatchable_generator,
        ) = (ParkingSpots, Chargers, Grid, Storage, PV)

    def initialize_vehicle_population(self, request_data):
        """
        Generate ordered list of Vehicle objects entering the EVCC on day=day
        :param request_data: A dataframe of parking and charging requests
        :param data: empirical preference data
        :return: generator object containing
        """
        requests = []
        for i in range(len(request_data)):
            id = i  # int
            facility = request_data.loc[i, "SiteID"]
            user_type = request_data.loc[i, "ClusterName"]
            arrival_date = request_data.loc[i, "EntryDate"]  # datetime object
            departure_date = request_data.loc[i, "ExitDate"]  # datetime object
            arrival_time = request_data.loc[i, "EntryDateTime"]  # datetime object
            departure_time = request_data.loc[i, "ExitDateTime"]  # datetime object
            arrival_period = request_data.loc[
                i, "EntryMinutesFromSimStart"
            ]  # minutes from start of sim
            departure_period = request_data.loc[
                i, "ExitMinutesFromSimStart"
            ]  # minutes from start of sim
            if self.benchmarking:
                arrival_period = (
                    int(arrival_period / 60) * 60
                )  # Activate it only when we use perfect info
                departure_period = (
                    int(min(departure_period, self.sim_time) / 60) * 60
                )  # Activate it only when we use perfect info
                # departure_period = arrival_period + 4
            ev = request_data.loc[i, "EV_yn"]  # EV yes or no
            energy_requested = (
                request_data.loc[i, "final_kWhRequested_updated"] * 2
            )  # kwh
            energy_charged = 0  # initialize to 0
            battery_size = request_data.loc[i, "BatterySize"]  # kwh

            vehicle_i = Vehicle(
                self.env,
                id=id,
                user_type=user_type,
                facility=facility,
                arrival_date=arrival_date,
                departure_date=departure_date,
                arrival_time=arrival_time,
                departure_time=departure_time,
                arrival_period=arrival_period,
                departure_period=departure_period,
                sim_time=self.sim_time,
                ev=ev,
                energy_requested_input=energy_requested,
                energy_charged=energy_charged,
                battery_size=battery_size,
            )

            requests.append(vehicle_i)
        # if self.planning is True:
        requests = [i for i in requests if i.ev == 1]
        return requests

    ##############################################################
    # RUN SIMULATION

    def run(self):
        """
        Running the whole process of serving the request and monitoring the resources
        """
        self.env.process(
            self.operator.request_queueing()
        )  # vehicle arrivals and assignment to charging station
        self.env.process(
            self.operator.get_charging_schedules_and_prices(
                self.charging_algo, mode=self.scheduling_mode
            )
        )
        if self.charging_algo not in [
            "integrated_storage",
            "perfect_info_with_storage",
        ]:
            if self.storage_capacity > 0:
                if not Configuration.instance().dynamic_storage_scheduling:
                    self.env.process(
                        self.operator.get_storage_schedule(
                            storage_strategy=self.storage_algo,
                            mode=self.scheduling_mode,
                        )
                    )
        self.env.process(self.operator.storage_process())
        self.env.process(
            self.grid.monitor(
                self.base_load,
                self.chargers,
                self.non_dispatchable_generator,
                self.electric_storage,
                energy_costs=self.electricity_tariff,
                vehicles=self.requests,
            )
        )
        if self.planning is False:
            for charging_station in self.chargers:
                self.env.process(charging_station.monitor())
            self.env.process(self.parking_lot.monitor())
            self.env.process(self.electric_storage.monitor())

    ############################################################################
    # SAVE AND PLOT SIMULATION RESULTS

    def convert_to_int_if_none(self, x):
        time = self.env
        if x:
            return int(x)

    def save_results(self, method="RL", year=9, week=1, post_fix=""):
        """
        Saving the results of request transitions, state of chargers, state of parking_spots and state of requests
        """
        # # Retrieving and saving request operational data
        requests_info = []
        for i in self.requests:
            info = {
                "facility": i.facility,
                "vehicle_id": i.id,
                "ev_yn": i.ev,
                "user_type": i.user_type,
                "arrival_time": i.arrival_time,
                "arrival_period": self.convert_to_int_if_none(i.arrival_period),
                "departure_time": i.departure_time,
                "departure_period": self.convert_to_int_if_none(i.departure_period),
                "assigned_charger": i.assigned_charger,
                "assigned_parking": i.assigned_parking,
                "assigned_time": self.convert_to_int_if_none(i.assigned_time),
                "estimated_waiting_time": i.estimated_waiting_time,
                "waiting_time": self.convert_to_int_if_none(i.waiting_time),
                "stop_charging_time": i.stop_charging_time,
                "energy_requested": i.energy_requested,
                "energy_charged": i.energy_charged,
                "charging_price": i.charging_price,
                "charging_max_power": i.max_charging_power,
                "average_power_requirement": i.average_power_requirement,
            }
            requests_info.append(info)
        results = pd.DataFrame(requests_info)
        results.to_csv(
            self.raw_output_save_path
            + f"requests_{method}_{year}_{week}_{post_fix}.csv"
        )

        # COMMENTED OUT FOR NOW
        # pd_ve = pd.DataFrame()
        # for j in self.requests:
        #    pd_ve = pd_ve.append(pd.DataFrame([j.info["mode"], j.info['SOC'], j.info['charging_power']]))
        # pd_ve.to_csv(f'/tmp/pycharm_project_194/Results/output/requests_details.csv')
        # pd_ve.to_csv(results_save_path+'requests_details.csv')

        # Retrieving and saving charger operational data
        pd_cs = pd.DataFrame()
        for c in self.chargers:
            # initialize and fill new df per each charger
            df = pd.DataFrame(
                [c.info["Connected"], c.info["Charging"], c.info["Consumption"]]
            )
            df["cs_id"] = c.id
            df["info"] = [
                "num_vehicles_connected",
                "num_vehicles_charging",
                "kWh_consumption",
            ]
            # append to combined df
            pd_cs = pd_cs.append(df)
        # save combined df
        pd_cs.to_csv(
            self.raw_output_save_path
            + f"CSs_{method}_{year}_{week}_{post_fix}_{Configuration.instance().pricing_agent_name}.csv"
        )

        # Retrieving and saving storage operational data
        storage_data = pd.DataFrame(self.electric_storage.info)
        storage_data["type"] = self.electric_storage.storage_type
        storage_data.to_csv(
            self.raw_output_save_path + f"storage_{method}_{year}_{week}_{post_fix}.csv"
        )

        # Saving historical price data
        self.operator.price_history.to_csv(
            self.raw_output_save_path
            + f"price_history_{method}_{year}_{week}_{post_fix}.csv"
        )

        # Calculating the operation and investment costs
        # self.objective_function_calculation()

    def update_objective_function(self, peak_threshold):
        self.costs["operations"] = 0
        self.costs["operations"] += self.grid.energy_costs
        # peak_charge = (max(self.grid.grid_usage) - peak_threshold) * self.prices['peak']
        # self.costs["operations"] += peak_charge  # Peak charge has already been discounted to daily charge in input!
        total_revenue = 0
        requests = [i for i in self.requests if i.ev == 1]

        for request in requests:
            energy_requested_adj = request.energy_requested * self.minimum_served_demand
            if request.is_assigned:
                if request.energy_requested > 0:
                    total_revenue += (
                        min(request.energy_charged, energy_requested_adj)
                        * request.charging_price
                    )  # overserving request does not count!
                    total_revenue += request.park_duration * request.parking_fee
            if self.env.now >= 1440 - 60:
                total_revenue -= (
                    max(energy_requested_adj - request.energy_charged, 0)
                    * request.charging_price
                    * Configuration.instance().energy_missed_penalty
                )
            elif request.mode == "Left":
                total_revenue -= (
                    max(energy_requested_adj - request.energy_charged, 0)
                    * request.charging_price
                    * Configuration.instance().energy_missed_penalty
                )
            total_revenue -= (
                max((request.raw_energy_demand - request.energy_requested), 0) * 0
            )

        # TODO: fix this
        # activate it when we have a single price
        # return (total_revenue * self.penalty_for_missed_kWh - (self.costs["operations"]))
        return total_revenue - (self.costs["operations"])

    def calculate_objective_function(self, initial_grid_capa):
        """
        Calculates investment cost, operational cost and service level, which is feedback for the greedy search algorithm
        :return:
        """
        ### Activate for investment problem
        ### TODO: define it as an option
        self.costs["operations"] = 0
        self.costs["operations"] += self.grid.energy_costs
        peak_charge = max(
            (max(self.grid.grid_usage) - self.peak_threshold) * self.prices["peak"], 0
        )
        self.costs[
            "operations"
        ] += peak_charge  # Peak charge has already been discounted to daily charge in input!
        lg.info(
            f'Daily Operations Costs = {self.costs["operations"]}, Daily Investment Costs = {self.costs["investment"]}'
        )

        # Obtaining the average service level at the end
        total_energy_requested = 0
        total_energy_charged = 0
        total_energy_missed = 0
        total_energy_canceled = 0
        total_revenue = 0
        served_demand_proportion = 0
        number_request = 0
        extra_charge = 0

        requests = [i for i in self.requests if i.ev == 1]

        for request in requests:
            total_energy_canceled += max(
                (request.raw_energy_demand - request.energy_requested), 0
            )
            if request.energy_requested > 0:
                energy_requested_adj = max(
                    request.energy_requested * self.minimum_served_demand, 0
                )
                total_energy_missed += max(
                    energy_requested_adj - request.energy_charged, 0
                )
                total_energy_requested += (
                    energy_requested_adj  # request.energy_requested
                )
                total_energy_charged += min(
                    request.energy_charged, energy_requested_adj
                )  # overserving request does not count!
                extra_charge += max((request.energy_charged - energy_requested_adj), 0)
                if request.is_assigned:
                    total_revenue += (
                        min(request.energy_charged, energy_requested_adj)
                        * request.charging_price
                    )
                    total_revenue += request.park_duration * request.parking_fee
                    total_revenue -= (
                        max(energy_requested_adj - request.energy_charged, 0)
                        * request.charging_price
                        * Configuration.instance().energy_missed_penalty
                    )
                    served_demand_proportion += min(
                        1, request.energy_charged / request.energy_requested
                    )
                    number_request += 1
        lg.info(f"total missed demand is {total_energy_missed}")
        lg.info(
            f"total missed demand is {total_energy_missed / (total_energy_requested + 1)}"
        )
        # self.service_level = min(round(total_energy_charged / (total_energy_requested+1), 2), 1.00)
        if Configuration.instance().benchmarking:
            total_revenue += extra_charge * 0.15
        self.service_level = min(
            round(served_demand_proportion / max(number_request, 1), 2), 1.00
        )
        self.total_energy_charged = total_energy_charged
        self.total_energy_canceled = total_energy_canceled
        lg.error(
            f"service_level = {self.service_level}, energy_canceled = {total_energy_canceled}, "
            f"energy_charged = {total_energy_charged}, energy_missed = {total_energy_missed}"
        )
        if self.objective == "min_costs":
            self.objective_function = (
                total_energy_missed * self.penalty_for_missed_kWh
                + (self.costs["operations"])
            )
        if self.objective == "max_profits":
            self.objective_function = total_revenue - (self.costs["operations"])

    def visualize_results(
        self, model, sim_start_date, post_fix, visuals_save_path, palette="mako"
    ):
        """
        Run plottig routines on results output data
        :return:
        """

        viz.get_visuals(
            model=model,
            palette=palette,
            sim_start_date=sim_start_date,
            visuals_save_path=visuals_save_path,
            post_fix=post_fix,
        )
