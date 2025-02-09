import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import logging
from Environment.log import stream_handler
import pandas as pd
from Utilities.sim_input_processing import sample_training_and_test_weeks

stream_handler.setLevel(logging.ERROR)
from main import run_single_simulation


TRAIN_WEEKS, TEST_WEEKS = sample_training_and_test_weeks(seed=42)
k = 5


class ChargingHubInvestmentEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    reward_range = (-float("inf"), float("inf"))
    spec = None

    def __init__(self, config, DQN=False):
        # Set these in ALL subclasses
        self.DQN = DQN
        if DQN == True:
            # self.action_space = spaces.Discrete(2*2*2*4*4*5*2*3*3)
            self.action_space = spaces.Discrete(k**9)
        else:
            self.action_space = spaces.Box(
                low=0,
                high=np.array([2, 2, 2, 20, 5, 2, 2, 2, 5]),
                shape=(9,),
                dtype=np.uint8,
            )
            # self.action_space = spaces.Box(low=0, high=np.array([10, 10, 10, 20, 20, 20, 4, 10, 20]), shape=(9,),
            #                                dtype=np.uint8)
        # Example for using image as input:
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(10, 1), dtype=np.uint8
        )
        self.episode = 0
        self.state = None
        self.current_step = 0
        self.reward = 0
        self.results = np.ndarray((9, 0))
        self.env = "env"
        self._max_episode_steps = 500
        self.config = config
        self.evaluation = config.evaluation

    def step(self, action):
        # Execute one time step within the environment
        reward = self._take_action(action)
        self.current_step += 1
        done = self.current_step >= 10
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        ChargingHubInvestmentEnv(config=self.config)
        self.current_step = 0
        self.reward = 0
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        pd.DataFrame(self.results).to_csv("file.csv")

        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def render(self, mode="human", close=False):
        print(self.reward)

    def _take_action(self, action):
        # print(f'action={action}')

        if self.DQN == False:
            action = [int(x) for x in action]
            NUM_CHARGER = {
                "fast_one": 0,
                "fast_two": 0,
                "fast_four": 0,
                "slow_one": 0,
                "slow_two": 0,
                "slow_four": 0,
            }
            NUM_CHARGER["fast_one"] = int(self.state[0] + action[0])
            NUM_CHARGER["fast_two"] = int(self.state[1] + action[1])
            NUM_CHARGER["fast_four"] = int(self.state[2] + action[2])
            NUM_CHARGER["slow_one"] = int(self.state[3] + action[3])
            NUM_CHARGER["slow_two"] = int(self.state[4] + action[4])
            NUM_CHARGER["slow_four"] = int(self.state[5] + action[5])
            TRANSFORMER_NUM = int(self.state[6] + action[6])
            PV_CAPA = int(self.state[7] + action[7] * 10)
            STORAGE_CAPA = int(self.state[8] + action[8] * 10)
            self.state += np.append(action, 1)
        else:
            NUM_CHARGER = {
                "fast_one": 0,
                "fast_two": 0,
                "fast_four": 0,
                "slow_one": 0,
                "slow_two": 0,
                "slow_four": 0,
            }
            action_list = convert_to_vector(action)

            NUM_CHARGER["fast_one"] = int(self.state[0] + action_list[0] * 3)
            NUM_CHARGER["fast_two"] = int(self.state[1] + action_list[1] * 3)
            NUM_CHARGER["fast_four"] = int(self.state[2] + action_list[2] * 3)
            NUM_CHARGER["slow_one"] = int(self.state[3] + action_list[3] * 5)
            NUM_CHARGER["slow_two"] = int(self.state[4] + action_list[4] * 5)
            NUM_CHARGER["slow_four"] = int(self.state[5] + action_list[5] * 5)
            # NUM_CHARGER['slow_one'] = int(self.state[3] + action_list[3]*2)
            # NUM_CHARGER['slow_two'] = int(self.state[4] + action_list[4]*2)
            # NUM_CHARGER['slow_four'] = int(self.state[5] + action_list[5]*2)
            TRANSFORMER_NUM = int(self.state[6] + action_list[6] * 2)
            PV_CAPA = int(self.state[7] + action_list[7] * 20)
            STORAGE_CAPA = int(self.state[8] + action_list[8] * 50)
            action_list = [int(a) for a in action_list]
            self.state += np.append(action_list, 1)

        # if self.evaluation:
        print(self.state)
        self.results = np.append(self.results, self.state)
        reward = 0
        NoC = 0

        NoP = [1, 2, 4, 1, 2, 4]
        j = 0
        for i in [
            "fast_one",
            "fast_two",
            "fast_four",
            "slow_one",
            "slow_two",
            "slow_four",
        ]:
            NoC += NUM_CHARGER[i] * NoP[j]
            j += 1
        if NoC > 140:
            reward += (NoC - 140) * 10
        # for i in ['fast_one', 'fast_two', 'fast_four']:
        #     if NUM_CHARGER[i] > 10:
        #         reward += (NUM_CHARGER[i] - 10) * 10000
        # for i in ['slow_one','slow_two','slow_four']:
        #     if NUM_CHARGER[i] > 40:
        #         reward += (NUM_CHARGER[i] - 40) * 10000
        # if NoC > 140:
        #     reward += (NoC-140) * 1000
        #     reward = + reward
        # if TRANSFORMER_NUM > 6:
        #     reward += (TRANSFORMER_NUM - 6) * 10000
        # if PV_CAPA > 100:
        #     reward += (PV_CAPA - 100) * 10000
        # if STORAGE_CAPA > 500:
        #     reward += (STORAGE_CAPA - 500) * 10000
        SL = 0
        investment_costs = 0
        operations_costs = 0
        random.seed(42)
        train_weeks = random.sample(TRAIN_WEEKS, 1)
        for week in train_weeks:
            objective, investment_cost, operations_cost, service_level = (
                run_single_simulation(
                    num_charger=NUM_CHARGER,
                    transformer_num=TRANSFORMER_NUM,
                    pv_capa=PV_CAPA,
                    storage_capa=STORAGE_CAPA,
                    turn_off_monitoring=True,
                    year=self.current_step,
                    start_day=week,
                )
            )
            reward += objective
            SL += service_level
            investment_costs += investment_cost
            operations_costs += operations_cost
        # print(f'service_level={SL/len(TRAIN_WEEKS)}, investment_costs={investment_costs/len(TRAIN_WEEKS)}, operations_costs={operations_costs/len(TRAIN_WEEKS)}, objective = {reward/len(TRAIN_WEEKS)}')
        reward /= len(train_weeks)
        self.reward = -reward
        return -reward

    def _next_observation(self):
        return self.state


def convert_to_vector(a, h=8):
    # print(a)
    action = np.zeros(9)
    j = 0
    for i in range(9):
        action[i] = int((a - a % (k ** (h - j))) / (k ** (h - j)))
        a = a % (k ** (h - j))
        j += 1
    # print(action)
    return action


def convert_to_scalar(a):
    # print(a)
    action = 0
    for i in range(9):
        action += a[i] * (k) ** (8 - i)
    # print(action)
    return int(action)


def multiply(y):
    output = 1
    x = [2, 2, 2, 4, 4, 5, 2, 3, 3]
    for i in range(y):
        output *= x[i]
    return output
