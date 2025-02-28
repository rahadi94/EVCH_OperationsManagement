import gym
from gym import error, spaces, utils
import numpy as np
import logging
import pandas as pd

from Environment.helper.configuration.configuration import Configuration

k = 5


class PricingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, config, DQN=False):
        # Set these in ALL subclasses
        self.final_action_DQN = None
        if DQN == True:
            # self.action_space = spaces.Discrete(2*2*2*4*4*5*2*3*3)
            self.action_space = spaces.Discrete(k**config.number_power_options)
        else:
            number_of_actions = config.number_power_options - 1
            if (
                Configuration.instance().dynamic_fix_term_pricing
                and Configuration.instance().capacity_pricing
            ):
                number_of_actions = config.number_power_options
            if Configuration.instance().dynamic_parking_fee:
                number_of_actions = config.number_power_options
            if Configuration.instance().limiting_grid_capa:
                number_of_actions = config.number_power_options
            if Configuration.instance().dynamic_storage_scheduling:
                number_of_actions = config.number_power_options
            self.action_space = spaces.Box(
                low=0,
                high=config.maximum_power,
                shape=(number_of_actions,),
                dtype=np.float64,
            )
            if Configuration.instance().pricing_mode == "Discrete":
                # action_size = config.number_power_options
                action_size = 2
                self.action_space = spaces.Box(
                    low=0,
                    high=config.maximum_power,
                    shape=(action_size,),
                    dtype=np.float64,
                )
                self.action_space.low[0] = 0.3
                self.action_space.high[0] = 1.5
                self.action_space.low[1] = 0.5
                self.action_space.high[1] = 1.5
                # TODO: hard coded
                if config.number_power_options >= 3:
                    self.action_space.low[2], self.action_space.high[2] = 300, 800
                if config.number_power_options >= 4:
                    self.action_space.low[3], self.action_space.high[3] = -200, 200

            if Configuration.instance().pricing_mode == "Continuous":
                self.action_space.low[0] = 0
                self.action_space.high[0] = 1.5
                if Configuration.instance().limiting_grid_capa:
                    self.action_space.low[1] = 300
                    self.action_space.high[1] = 600
                if Configuration.instance().dynamic_storage_scheduling:
                    self.action_space.low[1] = -200
                    self.action_space.high[1] = +200
                if (
                    Configuration.instance().dynamic_fix_term_pricing
                    and Configuration.instance().capacity_pricing
                ):
                    self.action_space.low[0] = 0.5
                    self.action_space.high[0] = 1.5
                    self.action_space.low[1] = 0
                    self.action_space.high[1] = 0.4
                if (
                    Configuration.instance().dynamic_fix_term_pricing
                    and not Configuration.instance().capacity_pricing
                ):
                    self.action_space.low[0] = 0.6
                    self.action_space.high[0] = 1.5
                    if Configuration.instance().dynamic_parking_fee:
                        self.action_space.low[1] = 0
                        self.action_space.high[1] = 1 / 60
                # self.action_space.low[1] = 0.01
                # self.action_space.high[0] = 0.2
                # self.action_space.high[1] = 0.03

        # self.observation_space = spaces.Box(low=0, high=1000000, shape=
        # (config.number_chargers * 3 + 2 + 4, ), dtype=np.float64)
        observation_shape = 2 + 3 + 2
        if Configuration.instance().dynamic_storage_scheduling:
            observation_shape += 1
        self.observation_space = spaces.Box(
            low=0, high=1000000, shape=(observation_shape,), dtype=np.float64
        )
        self.charging_hub = None
        self.env = None
        self.id = 1
        self.episode = 0
        self.current_step = 0
        self.reward = 0
        self._max_episode_steps = 50000000
        self.config = config
        self.evaluation = config.evaluation
        self.total_reward = dict(
            missed=0, feasibility=0, energy=0, feasibility_storage=0, test=0
        )
        self.config = config
        self.action = None
        if not DQN:
            self.action_range = [self.action_space.low, self.action_space.high]

    def rescale_action(self, action):
        return (
            action * (self.action_range[1] - self.action_range[0]) / 2.0
            + (self.action_range[1] + self.action_range[0]) / 2.0
        )

    def get_final_prices_DQN(self, actions):

        final_action = actions.copy()
        if len(actions) == 1:
            for i in range(len(actions)):
                final_action[i] = actions[i] * 0.1 + 0.3
        if len(actions) == 2:
            for i in range(len(actions)):
                final_action[i] = actions[i] * 0.1 + 0.4 * i + 0.2 * (1 - i)
        self.final_action_DQN = final_action
        return final_action

    # def get_state(self, charging_hub=None, env=None):
    #     state = np.array([])
    #     if not env:
    #         hour = 0
    #         hour = np.array(hour)
    #         # hour = np.eye(24)[hour]
    #
    #         normalized_hour = hour / 24 / 4
    #
    #         # Map normalized hour to angle in radians
    #         angle = normalized_hour * 2 * np.pi
    #
    #         # Encode angle using sinusoidal functions
    #         sin_encoding = np.sin(angle)
    #         cos_encoding = np.cos(angle)
    #         day = 0
    #         day = np.array(day)
    #         day = np.eye(5)[day]
    #     else:
    #         hour = (env.now%1440 - env.now%charging_hub.planning_interval) / charging_hub.planning_interval
    #             hour = np.array(int(hour))
    #             normalized_hour = hour / 24 / (60/charging_hub.planning_interval)
    #
    #         # Map normalized hour to angle in radians
    #         angle = normalized_hour * 2 * np.pi
    #
    #         # Encode angle using sinusoidal functions
    #         sin_encoding = np.sin(angle)
    #         cos_encoding = np.cos(angle)
    #         # hour = np.eye(24)[hour]
    #
    #         day = (env.now - env.now % 1440)/1440
    #         day = np.array(int(day))
    #         day = np.eye(5)[day]
    #     state = np.append(state, np.array([sin_encoding, cos_encoding]))
    #
    #     # state = np.append(state, np.array([day]))
    #     if not charging_hub:
    #         storage_SoC = 0
    #         free_grid_capa = 0
    #         PV = 0
    #         electricity_price = 0
    #         peak_usage = 0
    #         avg_energy_demand = 0
    #         avg_power_demand = 0
    #         state = np.append(state, np.array([free_grid_capa, PV, electricity_price, peak_usage]))
    #         for i in range(self.config.number_chargers):
    #             for _ in range(4):
    #                 energy_demand = 0
    #                 charging_id = 0
    #                 # Time of Departure
    #                 ToD = 0
    #                 state = np.append(state, np.array([energy_demand, ToD, charging_id]))
    #     else:
    #         storage_SoC = charging_hub.electric_storage.SoC
    #         PV = charging_hub.operator.generation_min
    #         hour = (env.now % 1440 - env.now % 60) / 60
    #         electricity_price = charging_hub.electricity_tariff[int(hour)]
    #         peak_usage = charging_hub.operator.peak_threshold
    #         avg_energy_demand = 0
    #         avg_power_demand = 0
    #         if charging_hub.operator.free_grid_capa_actual == 0:
    #             free_grid_capa = charging_hub.operator.free_grid_capa_actual
    #         else:
    #             free_grid_capa = charging_hub.operator.free_grid_capa_actual[0]
    #
    #         state = np.append(state, np.array([free_grid_capa / 1000, PV / 10, electricity_price, peak_usage / 500]))
    #
    #         for charger in charging_hub.chargers:
    #             vehicles = charger.connected_vehicles
    #             charger_state = np.zeros(charger.number_of_connectors * 3)
    #             for j in range(len(vehicles)):
    #                 charger_state[j * 3 + 0] = vehicles[j].remaining_energy_deficit / 50
    #                 charger_state[j * 3 + 1] = vehicles[j].remaining_park_duration / 1000
    #                 charger_state[j * 3 + 2] = vehicles[j].charging_price / 4
    #             state = np.append(state, charger_state)
    #
    #     return state

    def get_state(self, charging_hub=None, env=None):
        state = np.array([])
        if not env:
            hour = 0
            hour = np.array(hour)
            # hour = np.eye(24)[hour]

            normalized_hour = hour / 24 / 4

            # Map normalized hour to angle in radians
            angle = normalized_hour * 2 * np.pi

            # Encode angle using sinusoidal functions
            sin_encoding = np.sin(angle)
            cos_encoding = np.cos(angle)
            day = 0
            day = np.array(day)
            day = np.eye(5)[day]
        else:
            hour = (
                env.now % 1440 - env.now % charging_hub.planning_interval
            ) / charging_hub.planning_interval
            hour = np.array(int(hour))
            normalized_hour = hour / 24 / (60 / charging_hub.planning_interval)

            # Map normalized hour to angle in radians
            angle = normalized_hour * 2 * np.pi

            # Encode angle using sinusoidal functions
            sin_encoding = np.sin(angle)
            cos_encoding = np.cos(angle)
            # hour = np.eye(24)[hour]

            day = (env.now - env.now % 1440) / 1440
            day = np.array(int(day))
            day = np.eye(5)[day]
        state = np.append(state, np.array([sin_encoding, cos_encoding]))

        # state = np.append(state, np.array([day]))
        if not charging_hub:
            storage_SoC = 0
            free_grid_capa = 0
            PV = 0
            electricity_price = 0
            peak_usage = 0
            avg_energy_demand = 0
            avg_power_demand = 0
            # state = np.append(state, np.array([free_grid_capa, PV, electricity_price, peak_usage, avg_energy_demand, avg_power_demand]))
            if Configuration.instance().dynamic_storage_scheduling:
                state = np.append(
                    state,
                    np.array(
                        [
                            storage_SoC,
                            PV,
                            electricity_price,
                            peak_usage,
                            avg_energy_demand,
                            avg_power_demand,
                        ]
                    ),
                )
            else:
                state = np.append(
                    state,
                    np.array(
                        [
                            PV,
                            electricity_price,
                            peak_usage,
                            avg_energy_demand,
                            avg_power_demand,
                        ]
                    ),
                )
        else:
            storage_SoC = charging_hub.electric_storage.SoC
            PV = charging_hub.operator.non_dispatchable_generator.generation_profile_actual.loc[
                env.now, "pv_generation"
            ]
            hour = (env.now % 1440 - env.now % 60) / 60
            electricity_price = charging_hub.electricity_tariff[int(hour)]
            peak_usage = charging_hub.operator.peak_threshold
            avg_energy_demand = 0
            avg_power_demand = 0
            if charging_hub.operator.free_grid_capa_actual == 0:
                free_grid_capa = charging_hub.operator.free_grid_capa_actual
            else:
                free_grid_capa = charging_hub.operator.free_grid_capa_actual[0]

            for charger in charging_hub.chargers:
                vehicles = charger.connected_vehicles
                for j in range(len(vehicles)):
                    avg_energy_demand += vehicles[j].remaining_energy_deficit
                    avg_power_demand += (
                        vehicles[j].remaining_energy_deficit
                        / vehicles[j].remaining_park_duration
                    )

            # state = np.append(state, np.array([free_grid_capa/1000, PV/500, electricity_price, peak_usage/1000,
            #                                    avg_energy_demand/1000, avg_power_demand/10]))
            if Configuration.instance().dynamic_storage_scheduling:
                state = np.append(
                    state,
                    np.array(
                        [
                            storage_SoC / 300,
                            PV / 500,
                            electricity_price,
                            peak_usage / 1000,
                            avg_energy_demand / 1000,
                            avg_power_demand / 10,
                        ]
                    ),
                )
            else:
                state = np.append(
                    state,
                    np.array(
                        [
                            PV / 500,
                            electricity_price,
                            peak_usage / 1000,
                            avg_energy_demand / 1000,
                            avg_power_demand / 10,
                        ]
                    ),
                )

        # print(state)

        return state

    def step(self, action, charging_hub=None, env=None):
        self.current_step += 1
        reward = self._take_action(action, charging_hub, env)
        done = self.current_step >= 100000000000000
        obs = self._next_observation(charging_hub, env)
        return obs, reward, done, {}

    def receive_action(self):
        return self.action

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.reward = 0
        # self.state = self.get_state()
        if not self.charging_hub:
            return self.get_state(None, None)
        return self.get_state(self.charging_hub, self.env)

    def render(self, mode="human", close=False):
        print(self.reward)

    def _take_action(self, action, charging_hub=None, env=None):

        reward = 0
        # hour = int((self.env.now % 1440) / 60)
        # prices = [0,0]
        # DQN_pricing = convert_to_vector(action)
        # for i in range(len(DQN_pricing)):
        #     DQN_pricing[i] = DQN_pricing[i] * 0.2 * (i + 1)
        # for i in range(2):
        #     prices[i] = Configuration.instance().prices[i] - hour / 4 / 20
        #     reward -= (prices[i] - DQN_pricing[i])**2

        reward -= charging_hub.reward["missed"]
        # reward -= charging_hub.reward['feasibility_storage'] * 0.1
        self.total_reward["missed"] += reward

        charging_hub.reward["missed"] = 0
        charging_hub.reward["feasibility_storage"] = 0
        charging_hub.reward["feasibility"] = 0
        return reward / 100

    def _next_observation(self, charging_hub, env):
        return self.get_state(charging_hub, env)


def convert_to_scalar(a):
    # print(a)
    action = 0
    for i in range(2):
        action += a[i] * (5) ** (1 - i)
    # print(action)
    return int(action)


def convert_to_vector(a, h=1):
    # print(a)
    action = np.zeros(2)
    j = 0
    for i in range(2):
        action[i] = int((a - a % (k ** (h - j))) / (k ** (h - j)))
        a = a % (k ** (h - j))
        j += 1
    # print(action)
    return action
