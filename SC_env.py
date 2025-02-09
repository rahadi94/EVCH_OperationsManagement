import gym
from gym import error, spaces, utils
import numpy as np
import logging
import pandas as pd


class ChargingHubInvestmentEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, config):
        # Set these in ALL subclasses
        self.action_space = spaces.Box(
            low=0,
            high=config.maximum_power,
            shape=(config.number_chargers + 1,),
            dtype=np.float64,
        )
        self.action_space.low[0] = 250
        self.action_space.high[0] = config.maximum_grid_usage
        self.observation_space = spaces.Box(
            low=0,
            high=1000000,
            shape=(config.number_chargers * 3 + 2 + 5,),
            dtype=np.float64,
        )
        self.charging_hub = None
        self.env = None
        self.id = 1
        self.episode = 0
        # vehicles_to_decide = [vehicle for vehicle in self.fleet.vehicles if vehicle.mode in ['idle','parking','circling']][0:10]
        # self.state = self.get_state(self.charging_hub, self.env)
        self.current_step = 0
        self.reward = 0
        self.results = np.ndarray((9, 0))
        # self.env = 'env'
        self._max_episode_steps = 50000000
        self.config = config
        self.evaluation = config.evaluation
        self.total_reward = dict(
            missed=0, feasibility=0, energy=0, feasibility_storage=0, test=0
        )
        self.config = config
        self.action = None

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
        if not charging_hub:
            storage_SoC = 0
            free_grid_capa = 0
            PV = 0
            electricity_price = 0
            peak_usage = 0
            state = np.append(
                state,
                np.array(
                    [storage_SoC, free_grid_capa, PV, electricity_price, peak_usage]
                ),
            )
            for i in range(self.config.number_chargers):
                for _ in range(4):
                    energy_demand = 0
                    charging_id = 0
                    # Time of Departure
                    ToD = 0
                    state = np.append(
                        state, np.array([energy_demand, ToD, charging_id])
                    )
        else:
            storage_SoC = charging_hub.electric_storage.SoC
            PV = charging_hub.operator.generation_min
            hour = (env.now % 1440 - env.now % 60) / 60
            electricity_price = charging_hub.electricity_tariff[int(hour)]
            peak_usage = charging_hub.operator.peak_threshold
            if charging_hub.operator.free_grid_capa_actual == 0:
                free_grid_capa = charging_hub.operator.free_grid_capa_actual
            else:
                free_grid_capa = charging_hub.operator.free_grid_capa_actual[0]
            state = np.append(
                state,
                np.array(
                    [storage_SoC, free_grid_capa, PV, electricity_price, peak_usage]
                ),
            )
            for charger in charging_hub.chargers:
                vehicles = charger.connected_vehicles
                charger_state = np.zeros(charger.number_of_connectors * 3)
                for j in range(len(vehicles)):
                    charger_state[j * 3 + 0] = vehicles[j].remaining_energy_deficit
                    charger_state[j * 3 + 1] = vehicles[j].remaining_park_duration
                    charger_state[j * 3 + 2] = charger.id
                state = np.append(state, charger_state)
        return state

    def step(self, action, charging_hub=None, env=None):
        # Execute one time step within the environment
        # the first action is charging/discharging of the battery
        # storage_power = action[0]
        # if storage_power >= 0:
        #     charging_hub.electric_storage.charge_yn = 1
        #     charging_hub.electric_storage.charging_power = storage_power
        # elif storage_power < 0:
        #     charging_hub.electric_storage.discharge_yn = 1
        #     charging_hub.electric_storage.discharging_power = - storage_power
        # for i in range(len(action)-1):
        #     charging_vehicles = charging_hub.chargers[i].charging_vehicles
        #     if len(charging_vehicles) > 0:
        #         charging_vehicles[0].charging_power = action[i+1]
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
        pd.DataFrame(self.results).to_csv("file.csv")
        if not self.charging_hub:
            return self.get_state(None, None)
        return self.get_state(self.charging_hub, self.env)

    def render(self, mode="human", close=False):
        print(self.reward)

    def _take_action(self, action, charging_hub, env):

        reward = 0
        penalty_ratio = 0.001
        reward -= charging_hub.reward["missed"]
        reward -= charging_hub.reward["feasibility"] * penalty_ratio
        # reward -= charging_hub.reward['feasibility_storage'] * penalty_ratio

        self.total_reward["missed"] -= charging_hub.reward["missed"]
        # print(f'charging:{self.total_reward["missed"]}')
        self.total_reward["feasibility"] -= (
            charging_hub.reward["feasibility"] * penalty_ratio
        )
        # self.total_reward['feasibility_storage'] -= charging_hub.reward['feasibility_storage'] * penalty_ratio
        self.total_reward["energy"] -= charging_hub.grid.energy_rewards * 0

        if not charging_hub.dynamic_pricing:
            charging_hub.reward["missed"] = 0
            charging_hub.reward["feasibility_storage"] = 0
            charging_hub.reward["feasibility"] = 0

        return reward / 100

    def _next_observation(self, charging_hub, env):
        return self.get_state(charging_hub, env)
