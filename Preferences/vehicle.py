from Environment.log import lg
import numpy as np

from Environment.helper.configuration.configuration import Configuration
from Preferences.EV_user_decision_making import ev_decision_making


class Vehicle:
    """
    Class for tracking vehicle state
    """

    def __init__(
        self,
        env,
        id,
        user_type,
        facility,
        arrival_date,
        departure_date,
        arrival_time,
        departure_time,
        arrival_period,
        departure_period,
        ev,
        energy_requested_input,
        sim_time,
        energy_charged,
        battery_size,
    ):
        self.env = env  # simulation environment
        self.id = id  # int
        self.user_type = user_type  # which cluster?
        self.facility = facility  # string
        self.arrival_date = arrival_date  # datetime object
        self.departure_date = departure_date  # datetime object
        self.arrival_time = arrival_time  # datetime object
        self.departure_time = departure_time  # datetime object
        self.arrival_period = arrival_period  # minutes from sim start
        self.departure_period = departure_period  # minutes from sim start
        self.sim_time = sim_time  # sim duration in minutes
        self.park_duration = max(departure_period - arrival_period, 60)  # minutes
        self.remaining_park_duration = self.park_duration
        self.stop_charging_time = None
        self.ev = ev  # EV_yn
        self.energy_requested = self.adjust_energy_request(
            energy_requested_input
        )  # energy_requested  # kWh
        self.raw_energy_demand = self.energy_requested.copy()
        self.energy_requested = min(self.energy_requested, self.park_duration / 60 * 50)
        self.energy_charged = energy_charged  # kWh
        self.remaining_energy_deficit = self.energy_requested
        self.laxity = None
        self.average_power_requirement = self.energy_requested / (
            self.park_duration / 60
        )
        self.average_power_requirement_level = (
            self.set_average_power_requirement_level()
        )
        self.charging_power = 0  # kW #evtl. this will become a list that gets updates for non-myopic algos
        self.charge_schedule = dict()
        self.battery_size = battery_size  # kwh
        self.mode = (
            None  # tracks mode of vehicle [parking, connected, charging, leaving]
        )
        self.assigned_charger = None  # tracks charger request
        self.assigned_parking = None  # tracks parking request
        self.assigned_time = None
        self.estimated_waiting_time = None
        self.waiting_time = None
        self.event_arrival = env.event()  # event of arrival
        self.event_departure = env.event()  # event of departure
        self.event_stop_charging = env.event()  # event of stop_charging
        self.info = {}  # information for final results
        self.remaining_laxity = None
        self.checked_reward = False
        self.profit_reward = 0
        self.action_id = None
        self.is_assigned = None
        self.is_paid = None
        self.utility_constant = Configuration.instance().get_utility_constant()
        self.utility_beta = Configuration.instance().get_utility_beta(
            self.arrival_time.hour, self.average_power_requirement
        )
        self.utility_beta_parking = Configuration.instance().get_utility_beta_parking(
            self.arrival_time.hour
        )
        self.charging_price = Configuration.instance().default_charging_price
        self.max_charging_power = Configuration.instance().default_charging_power
        self.base_price = Configuration.instance().get_base_price(
            self.arrival_time.hour, self.average_power_requirement
        )
        self.base_parking_fee = Configuration.instance().get_base_parking_fee(
            self.arrival_time.hour, self.average_power_requirement
        )
        self.base_power = Configuration.instance().get_base_power()
        self.request_adjusting_mode = Configuration.instance().request_adjusting_mode
        self.parking_fee = Configuration.instance().parking_price

    def adjust_energy_request(self, energy_requested_input):
        """
        Adjust energy request for vehicles that stay beyond end of simulation
        :param energy_requested_input:
        :return:
        """

        if self.departure_period <= self.sim_time:
            energy_request = energy_requested_input
        elif self.departure_period > self.sim_time:
            energy_request = energy_requested_input * (
                (self.sim_time - self.arrival_period) / self.park_duration
            )

        return energy_request

    def set_average_power_requirement_level(self):
        laxity = self.energy_requested / (self.park_duration / 60)
        if laxity >= 40:
            return 3
        elif laxity >= 20:
            return 2
        else:
            return 1

    @staticmethod
    def find_price_for_raw_demand(raw_demand, critical_points, prices):
        for i in range(len(critical_points)):
            if raw_demand < critical_points[i]:
                return prices[i]
        return prices[-1]

    def adjust_request_demand_based_on_pricing(
        self, price_pairs, pricing_parameters, parking_fee
    ):  # TODO: change the name to decision-making
        if self.request_adjusting_mode == "Discrete":
            max_offering_power = max(price_pairs[:, 0])
            raw_demand = self.energy_requested
            raw_power_demand = min(
                self.energy_requested / (self.park_duration / 60), max_offering_power
            )
            """ each vehicle has multiple demand decision alternative including demand for different prices and its raw 
            energy demand """
            max_demand_for_prices = price_pairs[:, 0]
            prices = price_pairs[:, 1]
            raw_demand_price = self.find_price_for_raw_demand(
                raw_power_demand, max_demand_for_prices, prices
            )
            critical_points = sorted(np.append(max_demand_for_prices, raw_demand))

            prices = sorted(np.append(prices, raw_demand_price))
            utilities = np.array([])
            for i in range(len(critical_points)):
                point = critical_points[i]
                price = prices[i]
                utility = self.energy_demand_utility_function(
                    price, point, raw_demand, 0
                )
                utilities = np.append(utilities, utility)

            # we consider a minimum utility for accepting the service
            base_utility = self.energy_demand_utility_function(
                self.base_price, self.base_power, raw_demand, 0
            )
            if all(utility < base_utility for utility in utilities):
                power_choice = 0
                price = 0
            else:
                power_choice = critical_points[utilities.argmax()]
                price = prices[utilities.argmax()]
            self.charging_price = price
            self.max_charging_power = power_choice
            lg.info(f"Request {self.id} request {power_choice} kW with price {price}")
            self.energy_requested = min(
                power_choice * (self.park_duration / 60), self.energy_requested
            )
            # print(price, power_choice, self.energy_requested, self.raw_energy_demand)
            if self.charging_price > Configuration.instance().maximum_price_taking:
                self.energy_requested = 0
        if self.request_adjusting_mode == "Continuous":
            # price_for_11_kW = Configuration.instance().price_function(pricing_parameters[0],
            #                                                               pricing_parameters[1], 11)
            # if price_for_11_kW > self.base_price or parking_fee > self.base_parking_fee:
            #     self.energy_requested = 0
            #     # print(price_for_11_kW, self.base_price, pricing_parameters[1])
            # else:
            # decisions = ev_decision_making(p_0=pricing_parameters[0], alpha=pricing_parameters[1],
            #                                p_p=parking_fee, max_power=50,
            #                                beta_0=self.utility_beta, beta_1=0.2,
            #                                D=self.energy_requested, T=self.park_duration)['xvars']
            # print(decisions, self.energy_requested, self.park_duration)
            # raw_park_duration = self.park_duration
            # raw_energy_requested = self.energy_requested
            # self.energy_requested = decisions[0]
            # self.park_duration = decisions[1]

            beta_parking, beta, delta, demand, p_0, alpha = (
                self.utility_beta_parking,
                self.utility_beta,
                self.park_duration / 60,
                self.energy_requested,
                pricing_parameters[0],
                pricing_parameters[1],
            )
            if Configuration.instance().dynamic_parking_fee:
                self.parking_fee = parking_fee
                self.park_duration = max(
                    (2 * beta_parking * delta * 60 - parking_fee) / (2 * beta_parking),
                    0,
                )
            if not Configuration.instance().capacity_pricing:
                self.energy_requested = max((2 * beta * demand - p_0) / (2 * beta), 0)
            else:
                self.energy_requested = min(
                    max(
                        ((2 * beta * demand - p_0) * delta)
                        / (2 * (beta * delta + alpha)),
                        0,
                    ),
                    demand,
                )
            # self.departure_period = self.arrival_period + self.park_duration
            # self.parking_fee = parking_fee
            charging_power = self.energy_requested / delta

            self.charging_price = Configuration.instance().price_function(
                p_0, alpha, charging_power
            )
            # print("charging_price", self.charging_price)
            if self.charging_price > Configuration.instance().maximum_price_taking:
                self.energy_requested = 0
            # print(f'Charging_price is {self.charging_price}, power is {charging_power}, '
            #       f'Parking_fee is {self.parking_fee}, '
            #       f'park_difference is {delta*60 - self.park_duration} '
            #       f'energy_difference is {demand - self.energy_requested}')

    def energy_demand_utility_function(self, price, point, raw_demand, parking_fee):
        energy_requested = min(point * self.park_duration / 60, raw_demand)
        # print("Energy requested: ", energy_requested, point, self.park_duration)
        utility = (
            -price * energy_requested
            - self.utility_beta * (raw_demand - energy_requested) ** 2
            - self.park_duration * parking_fee
        )
        # utility = self.utility_constant - price * energy_requested - \
        #           self.utility_beta * -np.log(energy_requested/raw_demand)
        return utility

    def update_status(self):
        self.remaining_energy_deficit = self.energy_requested - self.energy_charged
        self.remaining_park_duration = max(self.departure_period - self.env.now, 1)
        # self.remaining_laxity = self.remaining_energy_deficit/max(self.remaining_park_duration,1)
        # if self.mode in ['Connected']:
        #     print(f'id={self.id}, power={self.charging_power}')

    def reset_profit_reward(self):
        self.profit_reward = 0

    def calculate_profit_reward(self, energy_price, electricity_tariff):
        # hour = int((self.env.now % 1440 - self.env.now % 60) / 60)
        self.profit_reward += self.charging_power / 60 * (energy_price)
