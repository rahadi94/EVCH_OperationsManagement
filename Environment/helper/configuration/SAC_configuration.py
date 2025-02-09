import argparse

from Deep_Reinforcement_Learning_Algorithms_with_PyTorch_master.utilities.data_structures.Config import (
    Config,
)
from Environment.helper.configuration.configuration import Configuration

config = Config()
config.seed = False
config.num_episodes_to_run = 1000000
config.file_to_save_data_results = "result/result"
config.file_to_save_results_graph = "result/plot"
config.visualise_individual_results = True
config.visualise_overall_agent_results = False
config.randomise_random_seed = False
config.runs_per_agent = 1
config.use_GPU = False
config.evaluation = False
config.learnt_network = False
config.average_score_required_to_win = 0
config.add_extra_noise = False
config.name = "charging"
config.automatically_tune_entropy_hyperparameter = True

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.001,
        "batch_size": 256,
        "buffer_size": 40000,
        "epsilon": 0.3,
        "epsilon_decay_rate_denominator": 200,
        "discount_rate": 0.99,
        "tau": 0.1,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [256, 256, 256],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "clip_rewards": False,
        "learning_iterations": 1,
    },
    "Actor_Critic_Agents": {
        "learning_rate": 0.0001,
        "linear_hidden_units": [64],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": None,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 0,
        "normalise_rewards": False,
        "automatically_tune_entropy_hyperparameter": config.automatically_tune_entropy_hyperparameter,
        "add_extra_noise": config.add_extra_noise,
        "min_steps_before_learning": 2000,
        "entropy_term_weight": 0.99,
        "do_evaluation_iterations": config.evaluation,
        "clip_rewards": False,
        "Actor": {
            "learning_rate": 0.00001,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.5,
            "gradient_clipping_norm": None,
        },
        "Critic": {
            "learning_rate": 0.00001,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 50000,
            "tau": 0.5,
            "gradient_clipping_norm": None,
        },
        "batch_size": 512,
        "mu": 0.0,  # for O-H noise
        "theta": 0.05,  # for O-H noise
        "sigma": 0.05,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0,
    },
}
config.hyperparameters = config.hyperparameters["Actor_Critic_Agents"]

pricing_config = Config()
pricing_config.seed = config.seed
pricing_config.num_episodes_to_run = config.num_episodes_to_run
pricing_config.file_to_save_data_results = config.file_to_save_data_results
pricing_config.file_to_save_results_graph = config.file_to_save_results_graph
pricing_config.visualise_individual_results = True
pricing_config.visualise_overall_agent_results = False
pricing_config.randomise_random_seed = False
pricing_config.runs_per_agent = 1
pricing_config.use_GPU = False
pricing_config.evaluation = config.evaluation
pricing_config.learnt_network = config.learnt_network
pricing_config.average_score_required_to_win = 0
pricing_config.add_extra_noise = False
pricing_config.name = Configuration.instance().pricing_agent_name
pricing_config.automatically_tune_entropy_hyperparameter = True
pricing_config.batch_size = 256
pricing_config.updates_per_step = 1
pricing_config.do_evaluation_iterations = False

pricing_config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.0001,
        "batch_size": 64,
        "buffer_size": 200000,
        "epsilon": 0.1,
        "epsilon_decay_rate_denominator": 20,
        "discount_rate": 0.99,
        "tau": 0.01,
        "alpha_prioritised_replay": 0.6,
        "beta_prioritised_replay": 0.4,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [64, 64],
        "final_layer_activation": "None",
        "batch_norm": False,
        "gradient_clipping_norm": None,
        "HER_sample_proportion": 0.8,
        "clip_rewards": False,
        "learning_iterations": 1,
    },
    "Actor_Critic_Agents": {
        "learning_rate": 0.0003,
        "linear_hidden_units": [64],
        "final_layer_activation": ["SOFTMAX", None],
        "gradient_clipping_norm": None,
        "discount_rate": 0.99,
        "epsilon_decay_rate_denominator": 0,
        "normalise_rewards": True,
        "automatically_tune_entropy_hyperparameter": pricing_config.automatically_tune_entropy_hyperparameter,
        "add_extra_noise": pricing_config.add_extra_noise,
        "min_steps_before_learning": 256,
        "entropy_term_weight": 0.8,
        "do_evaluation_iterations": pricing_config.evaluation,
        "clip_rewards": False,
        "Actor": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [512, 256, 512],
            "final_layer_activation": "TANH",
            "batch_norm": False,
            "tau": 0.05,
            "gradient_clipping_norm": None,
        },
        "Critic": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [512, 256, 512],
            "final_layer_activation": "None",
            "batch_norm": False,
            "buffer_size": 1000000,
            "tau": 0.05,
            "gradient_clipping_norm": None,
        },
        "batch_size": 64,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.15,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "HER_sample_proportion": 0.8,
        "exploration_worker_difference": 1.0,
    },
}
pricing_config.hyperparameters = pricing_config.hyperparameters["Actor_Critic_Agents"]


# storage_config = Config()
# storage_config.seed = config.seed
# storage_config.num_episodes_to_run = config.num_episodes_to_run
# storage_config.file_to_save_data_results = config.file_to_save_data_results
# storage_config.file_to_save_results_graph = config.file_to_save_results_graph
# storage_config.visualise_individual_results = True
# storage_config.visualise_overall_agent_results = False
# storage_config.randomise_random_seed = False
# storage_config.runs_per_agent = 1
# storage_config.use_GPU = False
# storage_config.evaluation = config.evaluation
# storage_config.learnt_network = config.learnt_network
# storage_config.average_score_required_to_win = 0
# storage_config.add_extra_noise = False
# storage_config.name = 'storage'
# storage_config.automatically_tune_entropy_hyperparameter = config.automatically_tune_entropy_hyperparameter
#
# storage_config.hyperparameters = {
#     "DQN_Agents": {
#         "learning_rate": 0.005,
#         "batch_size": 64,
#         "buffer_size": 40000,
#         "epsilon": 0.1,
#         "epsilon_decay_rate_denominator": 200,
#         "discount_rate": 0.99,
#         "tau": 0.1,
#         "alpha_prioritised_replay": 0.6,
#         "beta_prioritised_replay": 0.4,
#         "incremental_td_error": 1e-8,
#         "update_every_n_steps": 3,
#         "linear_hidden_units": [256, 256],
#         "final_layer_activation": "None",
#         "batch_norm": False,
#         "gradient_clipping_norm": 5,
#         "HER_sample_proportion": 0.8,
#         "clip_rewards": False,
#         "learning_iterations": 1
#     },
#
#     "Actor_Critic_Agents": {
#
#         "learning_rate": 0.0001,
#         "linear_hidden_units": [64],
#         "final_layer_activation": ["SOFTMAX", None],
#         "gradient_clipping_norm": None,
#         "discount_rate": 0.99,
#         "epsilon_decay_rate_denominator": 0,
#         "normalise_rewards": False,
#         "automatically_tune_entropy_hyperparameter": storage_config.automatically_tune_entropy_hyperparameter,
#         "add_extra_noise": storage_config.add_extra_noise,
#         "min_steps_before_learning": 512,
#         "entropy_term_weight": 0.99,
#         "do_evaluation_iterations": storage_config.evaluation,
#         "clip_rewards": False,
#
#         "Actor": {
#             "learning_rate": 0.0005,
#             "linear_hidden_units": [4],
#             "final_layer_activation": "TANH",
#             "batch_norm": False,
#             "tau": 0.005,
#             "gradient_clipping_norm": None
#         },
#
#         "Critic": {
#             "learning_rate": 0.0005,
#             "linear_hidden_units": [4],
#             "final_layer_activation": "None",
#             "batch_norm": False,
#             "buffer_size": 50000,
#             "tau": 0.005,
#             "gradient_clipping_norm": None
#         },
#
#         "batch_size": 32,
#         "mu": 0.0,  # for O-H noise
#         "theta": 0.05,  # for O-H noise
#         "sigma": 0.05,  # for O-H noise
#         "action_noise_std": 0.2,  # for TD3
#         "action_noise_clipping_range": 0.5,  # for TD3
#         "update_every_n_steps": 1,
#         "learning_updates_per_learning_session": 1,
#         "HER_sample_proportion": 0.8,
#         "exploration_worker_difference": 1.0
#
#     }
# }
# storage_config.hyperparameters = storage_config.hyperparameters['Actor_Critic_Agents']
