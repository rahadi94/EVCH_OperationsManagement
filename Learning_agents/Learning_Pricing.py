from Environment.helper.configuration.configuration import Configuration
from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.agents.actor_critic_agents.SAC import (
    SAC,
)
from Environment.helper.configuration.SAC_configuration import (
    config,
    pricing_config,
)
from Utilities.RL_environments.rl_pricing_env import PricingEnv
from Utilities.sim_input_processing import sample_week
from Environment.log import lg
import pandas as pd
import numpy as np

from run_simulation import run_single_simulation

SIM_SEASON = Configuration.instance().SIM_SEASON
SUMMER_START = Configuration.instance().SUMMER_START
SUMMER_END = Configuration.instance().SUMMER_END
POST_FIX = Configuration.instance().POST_FIX

Configuration.instance().dynamic_pricing = True

evaluate_after_training = Configuration.instance().evaluation_after_training
number_of_chargers = 200
PV_CAPA = Configuration.instance().PV
STORAGE_CAPA = 0
max_cap = 50
max_grid_usage = 2000
TRANSFORMER_NUM = Configuration.instance().grid

# config.number_chargers = number_of_chargers
# config.maximum_power = max_cap
# config.maximum_grid_usage = max_grid_usage
# config.environment = ChargingHubInvestmentEnv(config=config)
# config.learnt_network = evaluate_after_training
# agent = SAC(config)
agent = None

# storage_config.number_chargers = 80
# storage_config.maximum_power = 50
# storage_config.maximum_grid_usage = 200
# storage_config.environment = StorageEnv(config=storage_config)
# storage_config.learnt_network = False
# storage_agent = SAC(storage_config)

def run_experiments():
    pricing_config.number_chargers = number_of_chargers
    pricing_config.maximum_power = max_cap
    pricing_config.maximum_grid_usage = max_grid_usage
    pricing_config.number_power_options = len(Configuration.instance().energy_prices) + 0
    pricing_config.environment = PricingEnv(config=pricing_config, DQN=False)
    pricing_config.learnt_network = evaluate_after_training
    pricing_config.evaluation_after_training = evaluate_after_training

    pricing_agent = SAC(pricing_config)
    training_results = pd.DataFrame([])
    episode = 1
    NUMBER_EPISODES = 501
    if Configuration.instance().pricing_mode == "perfect_info":
        NUMBER_EPISODES = 1
    output = []
    while episode <= NUMBER_EPISODES:
        START = sample_week(
            sim_seasons=SIM_SEASON,
            summer_start=SUMMER_START,
            summer_end=SUMMER_END,
            seed=42,
        )
        print(START)
        # week = random.sample(TRAIN_WEEKS, 1)
        # week = START
        results = None
        off_monitoring = True
        evaluation_episodes = 10
        time_to_learn = pricing_agent.hyperparameters["min_steps_before_learning"]
        if evaluate_after_training:
            off_monitoring = False
            evaluation_episodes = 1
            results = [f"{POST_FIX}", f"state{9}", f"week{1}"]
            time_to_learn = 0
        # chargers = {'fast_one': 5, 'fast_two': 40, 'fast_four': 5, 'slow_one': 5, 'slow_two': 10, 'slow_four': 0}
        chargers = {
            "fast_one": number_of_chargers,
            "fast_two": 0,
            "fast_four": 0,
            "slow_one": 0,
            "slow_two": 0,
            "slow_four": 0,
        }
        lg.error(f"episode: {episode}", extra={"clazz": "", "oid": ""})
        if (
            episode % evaluation_episodes == 0
            and pricing_agent.global_step_number >= time_to_learn
        ):
            if agent:
                agent.do_evaluation_iterations = True
            pricing_agent.do_evaluation_iterations = True
            ### activate when we have separate battery agent
            # storage_agent.do_evaluation_iterations = True
            df = run_single_simulation(
                charging_agent=agent,
                storage_agent=None,
                pricing_agent=pricing_agent,
                num_charger=chargers,
                turn_off_monitoring=False,
                turn_on_results=results,
                turn_on_plotting=True,
                transformer_num=TRANSFORMER_NUM,
                storage_capa=STORAGE_CAPA,
                pv_capa=PV_CAPA,
                year=9,
                start_day=START,
            )
            pricing_agent.update_lr(new_objective=df["profit"], episode=episode)
            print(
                pricing_agent.alpha,
                pricing_agent.learning_rate_actor,
                max(pricing_agent.objective_function),
                pricing_agent.hyperparameters["Critic"]["tau"],
                pricing_agent.hyperparameters["batch_size"],
                pricing_agent.action_size,
            )
        else:
            if agent:
                agent.do_evaluation_iterations = False
            pricing_agent.do_evaluation_iterations = False
            ### activate when we have separate battery agent
            # storage_agent.do_evaluation_iterations = False
            df = run_single_simulation(
                charging_agent=agent,
                storage_agent=None,
                pricing_agent=pricing_agent,
                num_charger=chargers,
                turn_off_monitoring=False,
                turn_on_results=results,
                turn_on_plotting=True,
                transformer_num=TRANSFORMER_NUM,
                storage_capa=STORAGE_CAPA,
                pv_capa=PV_CAPA,
                year=9,
                start_day=START,
            )

        episode += 1
        pricing_agent.episode_number += 1
        if not Configuration.instance().evaluation_after_training:
            training_results = pd.concat([training_results, df])
            training_results.to_csv(
                f"Utilities/raw_output/training_results_{pricing_agent.config.name}.csv"
            )
        output.append(df["profit"].values[0])
    # print(output)
    return output[9:-1:10][-10:]


def find_best_parameters():
    try:
        training_results = pd.read_csv(f'training_results_{config.path}.csv')
    except:
        training_results = pd.DataFrame(columns=['learning_rate', 'batch_size', 'tau', 'result'])
    training_dict = {}
    best_results = -10000000000
    best_parameters = {'learning_rate': 0, 'batch_size': 0, 'tau': 0}
    for lr in [5e-5, 1e-4, 5e-4, 1e-3]:
        for bs in [64, 256, 512]:
            for tau in [0.05, 0.1]:
                pricing_config.hyperparameters['batch_size'] = bs
                pricing_config.hyperparameters['Actor']['learning_rate'] = lr
                pricing_config.hyperparameters['Critic']['learning_rate'] = lr
                pricing_config.hyperparameters['Actor']['tau'] = tau
                pricing_config.hyperparameters['Critic']['tau'] = tau
                pricing_config.hyperparameters['min_steps_before_learning'] = max(bs, 256)
                mean_reward = run_experiments()
                # print('Mean reward: ', mean_reward)
                hyperparameters = {'learning_rate': lr, 'batch_size': bs, 'tau': tau}
                if np.array(mean_reward).mean() > best_results:
                    best_results = np.array(mean_reward).mean()
                    best_parameters = hyperparameters
                results_dict = {'result': mean_reward}
                training_results = pd.concat(
                    [pd.DataFrame([[lr, bs, tau, mean_reward]], columns=training_results.columns),
                     training_results], ignore_index=True)
                print(f'{hyperparameters}, {results_dict}, best: {best_results}, best_parameters: {best_parameters}')
                training_results.to_csv(f'Utilities/raw_output/training_results_{pricing_config.name}_tuning.csv', index=False)

# find_best_parameters()
run_experiments()