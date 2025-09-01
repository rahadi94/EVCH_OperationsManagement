"""
Hyperparameter Tuner Module

This module handles hyperparameter tuning for learnable agents in the EVCC simulation framework.
"""

from typing import Dict, Any
from resources.configuration.configuration import Configuration
from utilities.training_manager import run_standard_training
import pandas as pd
import numpy as np


def find_best_parameters(agent, config: Configuration) -> None:
    """
    Find best hyperparameters for the agent through grid search.
    
    Args:
        agent: The agent instance to tune
        config: Configuration instance
    """
    print("Starting hyperparameter tuning...")
    
    # Try to read existing training results
    try:
        training_results = pd.read_csv(f'{config.OUTPUT_DATA_PATH}training_results_{agent.config.name}.csv')
    except:
        training_results = pd.DataFrame(columns=['learning_rate', 'batch_size', 'tau', 'result'])
    
    best_results = -10000000000
    best_parameters = {'learning_rate': 0, 'batch_size': 0, 'tau': 0}
    
    # Hyperparameter grid to search
    learning_rates = [5e-5, 1e-4, 5e-4, 1e-3]
    batch_sizes = [64, 256, 512]
    tau_values = [0.05, 0.1]
    
    total_combinations = len(learning_rates) * len(batch_sizes) * len(tau_values)
    current_combination = 0
    
    for lr in learning_rates:
        for bs in batch_sizes:
            for tau in tau_values:
                current_combination += 1
                print(f"Testing combination {current_combination}/{total_combinations}: lr={lr}, bs={bs}, tau={tau}")
                
                # Update agent hyperparameters
                if hasattr(agent, 'hyperparameters'):
                    agent.hyperparameters['batch_size'] = bs
                    if 'Actor' in agent.hyperparameters:
                        agent.hyperparameters['Actor']['learning_rate'] = lr
                    if 'Critic' in agent.hyperparameters:
                        agent.hyperparameters['Critic']['learning_rate'] = lr
                        agent.hyperparameters['Critic']['tau'] = tau
                    if 'Actor' in agent.hyperparameters:
                        agent.hyperparameters['Actor']['tau'] = tau
                    agent.hyperparameters['min_steps_before_learning'] = max(bs, 256)
                
                # Run training experiment
                try:
                    mean_reward = run_standard_training(agent, "pricing", config, return_rewards=True)
                    
                    # Track results
                    hyperparameters = {'learning_rate': lr, 'batch_size': bs, 'tau': tau}
                    if np.array(mean_reward).mean() > best_results:
                        best_results = np.array(mean_reward).mean()
                        best_parameters = hyperparameters
                        print(f"New best result: {best_results} with parameters: {best_parameters}")
                    
                    # Save results
                    results_dict = {'result': mean_reward}
                    new_row = pd.DataFrame([[lr, bs, tau, mean_reward]], columns=training_results.columns)
                    training_results = pd.concat([new_row, training_results], ignore_index=True)
                    
                    # Save to CSV
                    training_results.to_csv(
                        f'{config.OUTPUT_DATA_PATH}training_results_{agent.config.name}_tuning.csv', 
                        index=False
                    )
                    
                    print(f'Parameters: {hyperparameters}, Results: {results_dict}')
                    print(f'Best so far: {best_results}, Best parameters: {best_parameters}')
                    
                except Exception as e:
                    print(f"Error during hyperparameter tuning for {hyperparameters}: {e}")
                    continue
    
    print(f"\nHyperparameter tuning completed!")
    print(f"Best result: {best_results}")
    print(f"Best parameters: {best_parameters}")
    
    # Save final best parameters
    best_params_df = pd.DataFrame([best_parameters])
    best_params_df.to_csv(
        f'{config.OUTPUT_DATA_PATH}best_parameters_{agent.config.name}.csv', 
        index=False
    )


def get_hyperparameter_grid() -> Dict[str, list]:
    """
    Get the default hyperparameter grid for tuning.
    
    Returns:
        Dictionary containing hyperparameter grids
    """
    return {
        'learning_rates': [5e-5, 1e-4, 5e-4, 1e-3],
        'batch_sizes': [64, 256, 512],
        'tau_values': [0.05, 0.1]
    }


def update_agent_hyperparameters(agent, hyperparameters: Dict[str, Any]) -> None:
    """
    Update agent hyperparameters safely.
    
    Args:
        agent: The agent instance to update
        hyperparameters: Dictionary of hyperparameters to update
    """
    if not hasattr(agent, 'hyperparameters'):
        print("Warning: Agent does not have hyperparameters attribute")
        return
    
    # Update batch size
    if 'batch_size' in hyperparameters:
        agent.hyperparameters['batch_size'] = hyperparameters['batch_size']
    
    # Update learning rates
    if 'learning_rate' in hyperparameters:
        lr = hyperparameters['learning_rate']
        if 'Actor' in agent.hyperparameters:
            agent.hyperparameters['Actor']['learning_rate'] = lr
        if 'Critic' in agent.hyperparameters:
            agent.hyperparameters['Critic']['learning_rate'] = lr
    
    # Update tau values
    if 'tau' in hyperparameters:
        tau = hyperparameters['tau']
        if 'Actor' in agent.hyperparameters:
            agent.hyperparameters['Actor']['tau'] = tau
        if 'Critic' in agent.hyperparameters:
            agent.hyperparameters['Critic']['tau'] = tau
    
    # Update min steps before learning
    if 'batch_size' in hyperparameters:
        agent.hyperparameters['min_steps_before_learning'] = max(hyperparameters['batch_size'], 256)
