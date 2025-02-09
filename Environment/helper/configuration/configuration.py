from enum import Enum
import random
import numpy as np


class Configuration:

    # There is only one instance of this class.
    __instance = None

    @staticmethod
    def instance() -> "Configuration":
        """Static access method."""
        if Configuration.__instance == None:
            Configuration()
        return Configuration.__instance

    @staticmethod
    def init():
        """Static access method."""
        if Configuration.__instance == None:
            Configuration()
        return Configuration.__instance

    def __init__(self):
        """Virtually private constructor."""
        if Configuration.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Configuration.__instance = self
            super(Configuration, self).__init__()

        self.peak_threshold = 500
        self.peak_cost = 15.48 / 6 * 6
        self.energy_missed_penalty = 1.1
        self.B2G = False
        self.benchmarking = True
        # if self.benchmarking:
        #     self.peak_cost = 0
        self.remove_low_request_EVs = False
        self.evaluation_after_training = True
        self.request_adjusting_mode = "Continuous"  #'Discrete, Continuous'
        self.pricing_mode = "Continuous"  #'Discrete' ,Continuous, 'perfect_info', ToU
        mode = "capa"  # menu, capa, tra
        if mode == "menu":
            self.request_adjusting_mode = "Discrete"
            self.pricing_mode = "Discrete"
        facility_size = 200
        self.facility_size = facility_size
        price_sensitivity = "m"  # l, m, h
        self.price_sensitivity = price_sensitivity
        peak_penalty = "m"  # l, m, h
        self.adjust_peak_penalty(peak_penalty)
        charging_algorithm = "average_power"  # average_power, least_laxity_first
        self.charging_algorithm = charging_algorithm
        self.peak_penalty = peak_penalty
        PV = 500
        self.PV = PV
        grid = 50  # 2, 50
        self.grid = grid
        self.pricing_agent_name = f"pricing_double_{mode}_{PV}_{grid}_{charging_algorithm}_{price_sensitivity}_{facility_size}_{peak_penalty}_post_tuning"  #'pricing_double_BL_PV', 'pricing_double_PV_p0_alpha_low', 'pricing_double_PV_discrete_low'
        prices = [1, 0.2]
        power = [11, 50]
        parking_price = 0
        price_parameters = [1.5, 0]
        self.max_price_ToU = 1.5
        self.degree_of_power_in_price_function = 1  # Don't change this :D
        # self.pricing_agent_name = 'pricing_single_22'
        # prices = [0.50]
        # power = [22]

        self.power = power
        self.prices = prices
        self.parking_price = parking_price
        self.price_parameters = price_parameters
        self.maximum_price_taking = 1.5
        energy_prices = np.array([])
        for i in range(len(prices)):
            energy_prices = np.append(energy_prices, np.array([power[i], prices[i]]))
        self.energy_prices = energy_prices.reshape(len(prices), 2)
        self._vehicle_configs = {}
        self.dynamic_pricing = True

        if mode == "tra":
            self.capacity_pricing = False

        else:
            self.capacity_pricing = True

        self.dynamic_fix_term_pricing = True

        self.multiple_power = True
        self.dynamic_parking_fee = False
        self.limiting_grid_capa = False
        self.dynamic_storage_scheduling = False

        self.default_charging_price = 0.7
        self.default_charging_power = 50

        # user utility parameters
        self.lower_utility_constant = 0
        self.upper_utility_constant = 0
        self.lower_utility_beta = 0.05
        self.upper_utility_beta = 0.05
        self.lower_base_price = 0
        self.higher_base_price = 0
        self.lower_base_parking_fee = 0.01
        self.upper_base_parking_fee = 0.02
        self.lower_base_power = 0
        self.higher_base_power = 0

    def price_function(self, fixed_term, rate_based_term, power):
        if not self.capacity_pricing:
            return fixed_term
        # print(fixed_term + rate_based_term * power**self.degree_of_power_in_price_function)
        return max(
            fixed_term
            + rate_based_term * power**self.degree_of_power_in_price_function,
            0,
        )

    def get_utility_constant(self):
        return random.uniform(self.lower_utility_constant, self.upper_utility_constant)

    def get_utility_beta(self, hour, power):
        # print(max(random.uniform(self.lower_utility_beta, self.upper_utility_beta) + ((23-hour) ** 2 * 0.0002),
        #             0.001))
        # return  max(random.uniform(self.lower_utility_beta, self.upper_utility_beta) + ((23-hour) ** 2 * 0.0003),
        #             0.001)
        # return max(random.uniform(self.lower_utility_beta, self.upper_utility_beta) * (1-hour/23),
        #             0.001)
        random.seed(42)
        x = 1
        if self.price_sensitivity == "m":
            x = 2
        elif self.price_sensitivity == "h":
            x = 3
        return random.uniform(0.01, 0.03) / x

    def get_utility_beta_parking(self, hour):
        # print(max(random.uniform(self.lower_utility_beta, self.upper_utility_beta) + ((23-hour) ** 2 * 0.0002),
        #             0.001))
        # return  max(random.uniform(self.lower_utility_beta, self.upper_utility_beta) + ((23-hour) ** 2 * 0.0003),
        #             0.001)
        # return max(random.uniform(self.lower_utility_beta, self.upper_utility_beta) * (1-hour/23),
        #             0.001)
        random.seed(42)
        return random.uniform(0.05, 0.07)

    def get_base_price(self, hour, power):
        # if power < 11:
        #     return random.uniform(0.5, 0.6) - (23 - hour)/50
        # print(max(random.uniform(self.lower_base_price, self.higher_base_price) - (23 - hour)/50,0.2))
        return max(
            random.uniform(self.lower_base_price, self.higher_base_price)
            + (23 - hour) / 500,
            0.2,
        )

    def get_base_parking_fee(self, hour, power):
        # if power < 11:
        #     return random.uniform(0.5, 0.6) - (23 - hour)/50
        # print(max(random.uniform(self.lower_base_parking_fee, self.upper_base_parking_fee) + (23 - hour)/100, 0.01))
        return max(
            random.uniform(self.lower_base_parking_fee, self.upper_base_parking_fee)
            + (23 - hour) / 100,
            0.01,
        )

    def get_base_power(self):
        return random.uniform(self.lower_base_power, self.higher_base_power)

    # @property
    # def l_star(self):
    #     return self.l_star
    def adjust_peak_penalty(self, peak_penalty):
        if peak_penalty == 'm':
            self.peak_cost = self.peak_cost * 2
        if peak_penalty == 'h':
            self.peak_cost = self.peak_cost * 3



class VehicleConfig:
    def __init__(
        self,
        price_per_min,
        max_state_of_charge,
        energy_consumption,
        velocity,
        max_available_seats,
    ):
        self.price_per_min = price_per_min
        self.max_state_of_charge = max_state_of_charge
        self.energy_consumption = energy_consumption
        self.velocity = velocity
        self.max_available_seats = max_available_seats


class TransitAgentConfig:
    def __init__(self, action_resolution, alpha, eps, gamma, n):
        self.action_resolution = int(action_resolution)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.gamma = float(gamma)
        self.n = int(n)


class RLAgentConfig:
    def __init__(
        self,
        use_discrete_action_mapping,
        use_tracked_past_demand_data,
        action_interval,
        pricing_spatial_resolution,
        number_of_pricing_hexagons,
        batch_size,
    ):
        self._use_discrete_action_mapping = use_discrete_action_mapping
        self._use_tracked_past_demand_data = use_tracked_past_demand_data
        self._episode = 0
        self._action_number = 0
        self._action_interval = action_interval
        self._pricing_spatial_resolution = pricing_spatial_resolution
        self._number_of_pricing_hexagons = number_of_pricing_hexagons
        self._batch_size = batch_size

    @property
    def action_interval(self):
        return self._action_interval

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def number_of_pricing_hexagons(self):
        return self._number_of_pricing_hexagons

    @property
    def pricing_spatial_resolution(self):
        return self._pricing_spatial_resolution

    def increase_episode(self):
        self._episode += 1

    @property
    def use_tracked_past_demand_data(self):
        return self._use_tracked_past_demand_data

    @property
    def use_discrete_action_mapping(self):
        return self._use_discrete_action_mapping

    @property
    def episode(self):
        return self._episode

    @property
    def get_number_of_simulated_days(self):
        number_actions_per_day = 24 * 60 / self._action_interval
        return int(self._action_number / number_actions_per_day)

    def increase_action_number(self):
        self._action_number += 1

    @property
    def action_number(self):
        return self._action_number
