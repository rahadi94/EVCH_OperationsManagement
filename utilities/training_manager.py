"""
Training Manager Module

This module handles training operations for learnable agents in the EVCC simulation framework.
"""

from typing import Optional, List
from resources.configuration.configuration import Configuration
from utilities.sim_input_processing import sample_week
from run_simulation import run_single_simulation
import pandas as pd


def run_learnable_agent_training(agent_type: str, agent, decision_type: str, config: Configuration) -> None:
    """
    Run training for learnable agents (RL agents).
    
    Args:
        agent_type: Type of agent (RL_SAC, RL_DQN, RL_DDPG)
        agent: The agent instance to train
        decision_type: Type of decision (pricing, charging, storage, routing)
        config: Configuration instance
    """
    print(f"Starting training for {decision_type} agent: {agent_type}")
    
    # Check if hyperparameter tuning is enabled
    enable_hyperparameter_tuning = getattr(config, 'enable_hyperparameter_tuning', False)
    
    if enable_hyperparameter_tuning and decision_type == "pricing":
        print("Hyperparameter tuning enabled - running find_best_parameters()")
        from utilities.hyperparameter_tuner import find_best_parameters
        find_best_parameters(agent, config)
    else:
        print("Running standard training...")
        run_standard_training(agent, decision_type, config)
    
    print(f"Training completed for {decision_type} agent: {agent_type}")


def run_standard_training(agent, decision_type: str, config: Configuration, return_rewards: bool = False):
    """
    Run standard training for the agent.
    
    Args:
        agent: The agent instance to train
        decision_type: Type of decision (pricing, charging, storage, routing)
        config: Configuration instance
        return_rewards: Whether to return reward history for hyperparameter tuning
        
    Returns:
        List of rewards if return_rewards=True, otherwise None
    """
    print(f"Running standard training for {decision_type} agent...")
    
    # Training parameters
    NUMBER_EPISODES = 301
    if config.pricing_mode == "perfect_info":
        NUMBER_EPISODES = 1
    
    training_results = pd.DataFrame([])
    episode = 1
    output = []
    
    while episode <= NUMBER_EPISODES:
        # Sample training week
        START = sample_week(
            sim_seasons=config.SIM_SEASON,
            summer_start=config.SUMMER_START,
            summer_end=config.SUMMER_END,
            seed=42,
        )
        print(f"Episode {episode}: Training on week starting {START}")
        
        # Set evaluation mode
        evaluation_episodes = 10
        time_to_learn = agent.hyperparameters.get("min_steps_before_learning", 1000)
        
        if config.evaluation_after_training:
            evaluation_episodes = 1
            time_to_learn = 0
        
        # Charger configuration
        chargers = {
            "fast_one": config.CHARGER_NUM[0],
            "fast_two": config.CHARGER_NUM[1],
            "fast_four": config.CHARGER_NUM[2],
            "slow_one": config.CHARGER_NUM[3],
            "slow_two": config.CHARGER_NUM[4],
            "slow_four": config.CHARGER_NUM[5],
        }
        
        # Check if evaluation is needed
        if (episode % evaluation_episodes == 0 and 
            hasattr(agent, 'global_step_number') and 
            agent.global_step_number >= time_to_learn):
            
            agent.do_evaluation_iterations = True
            print(f"Episode {episode}: Running evaluation")
        else:
            agent.do_evaluation_iterations = False
        
        # Run simulation
        try:
            # During training, we don't want to save results every episode
            # Only save during evaluation episodes or if explicitly requested
            is_evaluation_episode = (episode % evaluation_episodes == 0 and 
                                   hasattr(agent, 'global_step_number') and 
                                   agent.global_step_number >= time_to_learn)
            
            # Prepare results parameters only for evaluation episodes
            results_params = None
            if is_evaluation_episode:
                results_params = [f"{getattr(config, 'POST_FIX', 'sim')}", f"state{9}", f"week{episode}"]
            
            # Pass the correct agent based on decision type
            charging_agent = agent if decision_type == "charging" else None
            storage_agent = agent if decision_type == "storage" else None
            pricing_agent = agent if decision_type == "pricing" else None
            
            df = run_single_simulation(
                charging_agent=charging_agent,
                storage_agent=storage_agent,
                pricing_agent=pricing_agent,
                num_charger=chargers,
                turn_off_monitoring=False,
                turn_on_results=results_params,  # Only save results during evaluation
                turn_on_plotting=is_evaluation_episode,  # Only plot during evaluation
                transformer_num=config.TRANSFORMER_NUM,
                storage_capa=config.STORAGE_SIZE,
                pv_capa=config.PV_INSTALLED_CAPA,
                year=9,
                start_day=START,
                config=config,
            )
            
            # Update learning rate if supported
            if hasattr(agent, 'update_lr'):
                agent.update_lr(new_objective=df["profit"], episode=episode)
            
            # Save training results only during evaluation episodes and if enabled
            if (is_evaluation_episode and 
                not config.evaluation_after_training and 
                getattr(config, 'save_training_results', False)):
                training_results = pd.concat([training_results, df])
                training_results.to_csv(
                    f'{config.OUTPUT_DATA_PATH}training_results_{agent.config.name}.csv'
                )
            
            output.append(df["profit"].values[0])
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}: Profit = {df['profit'].values[0]:.2f}")
            
        except Exception as e:
            print(f"Error in episode {episode}: {e}")
            output.append(0)  # Default value on error
        
        episode += 1
        if hasattr(agent, 'episode_number'):
            agent.episode_number += 1
    
    print(f"Standard training completed for {decision_type} agent")
    
    if return_rewards:
        return output[9:-1:10][-10:] if len(output) > 20 else output
    return None
