import random

from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.agents.actor_critic_agents.TD3 import (
    TD3,
)
from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.agents.DQN_agents.DDQN import (
    DDQN,
)
from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.agents.DQN_agents.DQN import (
    DQN,
)
from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.agents.actor_critic_agents.SAC import (
    SAC,
)
from Environment.log import stream_handler
import logging
import pandas as pd
import numpy as np
from RL_config_evaluation import config
from main import run_single_simulation, POST_FIX
from Utilities.sim_input_processing import sample_training_and_test_weeks
from rl_environment import ChargingHubInvestmentEnv, convert_to_vector

TRAIN_WEEKS, TEST_WEEKS = sample_training_and_test_weeks(seed=42)
method = "RL_TD3"
stream_handler.setLevel(logging.ERROR)
# new_population = numpy.random.random_integers(low=30, high=300, size=(20,3))
if method == "GA":
    experience = pd.read_csv(f"Cache/outputs/experience{POST_FIX}.csv").drop(
        "Unnamed: 0", axis=1
    )
    # lay_out = experience.loc[experience['90'].argmin(axis=0)].values
    lay_out = experience.loc[experience["9"].argmin(axis=0)].values
if method == "optimisation":
    lay_out = (
        pd.read_csv("investment_results_Facility_KoeBogen_2019-06-03.csv")
        .drop(["p_plus", "year"], axis=1)
        .values.reshape(
            90,
        )
    )
if method == "RL_TD3":
    config.hyperparameters = config.hyperparameters["Actor_Critic_Agents"]
    config.environment = ChargingHubInvestmentEnv(config=config, DQN=False)
    config.evaluation = True
    config.learnt_network = True
    config.hyperparameters["Actor"]["linear_hidden_units"] = [32, 32]
    config.hyperparameters["Critic"]["linear_hidden_units"] = [64, 64]
    config.level = "KoeBogen"
    RL_agent = TD3(config)
    RL_agent.reset_game()
if method == "RL_DQN":
    config.hyperparameters = config.hyperparameters["DQN_Agents"]
    config.environment = ChargingHubInvestmentEnv(config=config, DQN=True)
    config.learnt_network = True
    config.hyperparameters["linear_hidden_units"] = [64, 64]
    config.level = f"single_period_F3_LIR"
    RL_agent = DQN(config)
    RL_agent.turn_off_exploration = True
    RL_agent.reset_game()


def evaluation(i):
    if i is not None:
        i = [int(numeric_string) for numeric_string in i]
        print(i)
    # NUM_CHARGER_FINAL = 0
    # TRANSFORMER_NUM_FINAL = 0
    # PV_CAPA_FINAL = 0
    # STORAGE_CAPA_FINAL = 0
    # for j in range(10):
    #     for k in range(6):
    #         NUM_CHARGER_FINAL += i[j * 9 + k]
    #     TRANSFORMER_NUM_FINAL += i[j * 9 + 6]
    #     PV_CAPA_FINAL += i[j * 9 + 7]
    #     STORAGE_CAPA_FINAL += i[j * 9 + 8]
    # if NUM_CHARGER_FINAL > 200:
    #     return + (NUM_CHARGER_FINAL - 200) * 10000000
    # if TRANSFORMER_NUM_FINAL > 4:
    #     return + (TRANSFORMER_NUM_FINAL - 4) * 10000000
    # if PV_CAPA_FINAL > 100/10:
    #     return + (PV_CAPA_FINAL - 100/10) * 10000000
    # if STORAGE_CAPA_FINAL > 500/10:
    #     return + (STORAGE_CAPA_FINAL - 500/10) * 10000000
    objective = 0
    objective_operations = 0
    objective_investment = 0
    service_level = 0
    NUM_CHARGER = {
        "fast_one": 0,
        "fast_two": 0,
        "fast_four": 0,
        "slow_one": 0,
        "slow_two": 0,
        "slow_four": 0,
    }
    TRANSFORMER_NUM = 0
    PV_CAPA = 0
    STORAGE_CAPA = 0
    df_results_DVs = pd.DataFrame()
    df_results_KPIs = pd.DataFrame()
    for j in range(10):
        if method == "Manual":
            NUM_CHARGER["fast_one"] += 0
            NUM_CHARGER["fast_two"] += 0
            NUM_CHARGER["fast_four"] += 0
            NUM_CHARGER["slow_one"] += 80
            NUM_CHARGER["slow_two"] += 0
            NUM_CHARGER["slow_four"] += 10
            TRANSFORMER_NUM += 3
            PV_CAPA += 90
            STORAGE_CAPA += 500
        if method == "optimisation":
            NUM_CHARGER["slow_one"] += int(i[j * 9 + 0])
            NUM_CHARGER["slow_two"] += int(i[j * 9 + 1])
            NUM_CHARGER["slow_four"] += int(i[j * 9 + 2])
            NUM_CHARGER["fast_one"] += int(i[j * 9 + 3])
            NUM_CHARGER["fast_two"] += int(i[j * 9 + 4])
            NUM_CHARGER["fast_four"] += int(i[j * 9 + 5])
            TRANSFORMER_NUM += int(i[j * 9 + 6])
            PV_CAPA += int(i[j * 9 + 8])
            STORAGE_CAPA += int(i[j * 9 + 7])
        if method == "GA":
            NUM_CHARGER["fast_one"] += int(i[j * 9 + 0]) * 2
            NUM_CHARGER["fast_two"] += int(i[j * 9 + 1]) * 2
            NUM_CHARGER["fast_four"] += int(i[j * 9 + 2]) * 2
            NUM_CHARGER["slow_one"] += int(i[j * 9 + 3]) * 10
            NUM_CHARGER["slow_two"] += int(i[j * 9 + 4]) * 10
            NUM_CHARGER["slow_four"] += int(i[j * 9 + 5]) * 10
            TRANSFORMER_NUM += int(i[j * 9 + 6])
            PV_CAPA += int(i[j * 9 + 7]) * 10
            STORAGE_CAPA += int(i[j * 9 + 8]) * 50
        if method in ["RL_DQN", "RL_TD3"]:
            i = RL_agent.pick_action()
            print(i)
            if method == "RL_DQN":
                # i = int(checked_action(i, RL_agent.environment.state))
                i = convert_to_vector(i)
                i = [int(j) for j in i]
            RL_agent.environment.state += np.append(i, 1)
            RL_agent.state += np.append(i, 1)
            NUM_CHARGER["fast_one"] += int(i[0])
            NUM_CHARGER["fast_two"] += int(i[1])
            NUM_CHARGER["fast_four"] += int(i[2])
            NUM_CHARGER["slow_one"] += int(i[3])
            NUM_CHARGER["slow_two"] += int(i[4])
            NUM_CHARGER["slow_four"] += int(i[5])
            TRANSFORMER_NUM += int(i[6])
            if j == 1:
                TRANSFORMER_NUM += 0
            if j == 2:
                TRANSFORMER_NUM += 1
            if j == 3:
                NUM_CHARGER["fast_one"] += 1
            if j == 4:
                NUM_CHARGER["fast_one"] += 1
            if j in [3, 4, 5, 6, 7, 8, 9]:
                NUM_CHARGER["slow_two"] += 1
            PV_CAPA += int(i[7] * 10)
            STORAGE_CAPA += int(i[8] * 10)
        # print(NUM_CHARGER, TRANSFORMER_NUM, PV_CAPA, STORAGE_CAPA)
        decision_variables = NUM_CHARGER
        decision_variables["NoT"] = TRANSFORMER_NUM
        decision_variables["pv_capa"] = PV_CAPA
        decision_variables["storage_capa"] = STORAGE_CAPA
        print(decision_variables)
        df_results_DVs = df_results_DVs.append(
            pd.DataFrame(
                [decision_variables.values(), decision_variables.values()],
                columns=decision_variables.keys(),
            ).drop_duplicates()
        )
        # print(pd.DataFrame(data=decision_variables.values(),columns=decision_variables.keys()))
        inner_objective = 0
        inner_objective_operations = 0
        inner_objective_investment = 0
        inner_service_level = 0
        random.seed(42)
        test_weeks = random.sample(TRAIN_WEEKS, 1)
        week_number = 0
        for week in test_weeks:
            (
                new_objective,
                new_objective_investment,
                new_objective_operations,
                new_service_level,
            ) = (0, 0, 0, 0)
            results = [f"GA_{POST_FIX}", f"state{j}", f"week{week_number}"]
            # new_objective, new_objective_investment, new_objective_operations, new_service_level = run_single_simulation(num_charger=NUM_CHARGER,
            #                                                          transformer_num=TRANSFORMER_NUM,
            #                                                          pv_capa=PV_CAPA,
            #                                                          storage_capa=STORAGE_CAPA,
            #                                                          turn_on_results=False,turn_on_plotting=True,
            #                                                          turn_off_monitoring=False, year=j, start_day=week)
            inner_objective += new_objective
            inner_objective_investment += new_objective_investment
            inner_objective_operations += new_objective_operations
            inner_service_level += new_service_level
            week_number += 1
        objective += inner_objective / len(test_weeks)
        objective_operations += inner_objective_operations / len(test_weeks)
        objective_investment += inner_objective_investment / len(test_weeks)
        service_level = inner_service_level / len(test_weeks)
        KPIs = dict(
            objective=objective,
            objective_operations=objective_operations,
            objective_investment=objective_investment,
            service_level=service_level,
        )
        df_results_KPIs = df_results_KPIs.append(
            pd.DataFrame(
                [KPIs.values(), KPIs.values()], columns=KPIs.keys()
            ).drop_duplicates()
        )

    print(
        f"objective={objective}, operations_costs={objective_operations},investment_costs={objective_investment}"
    )
    # df_results_DVs.to_csv(f'Utilities/raw_output/Decision_Variables_GA_{POST_FIX}.csv')
    # df_results_KPIs.to_csv(f'Utilities/raw_output/KPIs_GA_{POST_FIX}.csv')
    print(service_level)
    # return objective


if method in ["RL_DQN", "RL_TD3", "Manual"]:
    evaluation(None)
else:
    evaluation(lay_out)
