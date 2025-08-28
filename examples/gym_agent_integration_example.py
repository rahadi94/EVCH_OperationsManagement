"""
Example: Integrating Standard Gym-Compatible RL Agents with EVCH Simulation

This example demonstrates how to use standard RL libraries (Stable Baselines3, RLlib, etc.)
with the EVCH simulation through the gym environment wrapper.

The simulation is completely decoupled from the RL agents, making it easy to plug in
any gym-compatible agent.
"""

import numpy as np
from typing import Dict, Any

# Example imports (these would need to be installed)
# from stable_baselines3 import SAC, PPO, DQN
# from stable_baselines3.common.vec_env import DummyVecEnv
# import gymnasium as gym

from utilities.rl_agents.factory import build_gym_agents_controller
from utilities.rl_environments.evch_gym_env import EVCHGymEnv, AgentType, EVCHConfig


def create_config_for_agent_type(agent_type: AgentType) -> Dict[str, Any]:
    """Create configuration dictionary for a specific agent type."""
    base_config = {
        "number_chargers": 10,
        "number_power_options": 2,
        "maximum_power": 800.0,
        "maximum_grid_usage": 1000.0,
        "evaluation": False,
        "pricing_mode": "Continuous",
        "dynamic_fix_term_pricing": True,
        "capacity_pricing": False,
        "dynamic_parking_fee": True,
        "limiting_grid_capa": True,
        "dynamic_storage_scheduling": True
    }
    
    # Add agent-specific configurations if needed
    if agent_type == AgentType.PRICING:
        base_config.update({
            "pricing_mode": "Continuous",
            "dynamic_fix_term_pricing": True,
            "capacity_pricing": False
        })
    elif agent_type == AgentType.CHARGING:
        base_config.update({
            "number_chargers": 15,
            "maximum_power": 1000.0
        })
    elif agent_type == AgentType.STORAGE:
        base_config.update({
            "dynamic_storage_scheduling": True
        })
    
    return base_config


def example_with_stable_baselines3():
    """
    Example using Stable Baselines3 agents.
    
    This shows how to integrate SAC, PPO, and DQN agents from Stable Baselines3.
    """
    print("=== Stable Baselines3 Integration Example ===")
    
    # Configuration for different agent types
    pricing_config = create_config_for_agent_type(AgentType.PRICING)
    charging_config = create_config_for_agent_type(AgentType.CHARGING)
    storage_config = create_config_for_agent_type(AgentType.STORAGE)
    
    # Create gym environments for training
    pricing_env = EVCHGymEnv(EVCHConfig(agent_type=AgentType.PRICING, **pricing_config))
    charging_env = EVCHGymEnv(EVCHConfig(agent_type=AgentType.CHARGING, **charging_config))
    storage_env = EVCHGymEnv(EVCHConfig(agent_type=AgentType.STORAGE, **storage_config))
    
    # Example: Create Stable Baselines3 agents (commented out as they're not installed)
    """
    # Pricing agent (SAC for continuous actions)
    pricing_agent = SAC(
        "MlpPolicy", 
        pricing_env, 
        verbose=1,
        learning_rate=0.0003,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256
    )
    
    # Charging agent (PPO for continuous actions)
    charging_agent = PPO(
        "MlpPolicy", 
        charging_env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64
    )
    
    # Storage agent (DQN for discrete actions if needed)
    storage_agent = DQN(
        "MlpPolicy", 
        storage_env, 
        verbose=1,
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts=1000
    )
    
    # Train the agents
    print("Training pricing agent...")
    pricing_agent.learn(total_timesteps=10000)
    
    print("Training charging agent...")
    charging_agent.learn(total_timesteps=10000)
    
    print("Training storage agent...")
    storage_agent.learn(total_timesteps=10000)
    
    # Create agents controller with trained agents
    agents_controller = build_gym_agents_controller(
        pricing_agent=pricing_agent,
        charging_agent=charging_agent,
        storage_agent=storage_agent,
        pricing_config=pricing_config,
        charging_config=charging_config,
        storage_config=storage_config
    )
    
    # Now you can use this controller with the simulation
    # operator = Operator(..., agents_controller=agents_controller)
    """
    
    print("Stable Baselines3 agents would be created and trained here.")
    print("The agents_controller can then be passed to the Operator.")


def example_with_custom_gym_agent():
    """
    Example using a custom gym-compatible agent.
    
    This shows how to create a simple custom agent that works with the gym interface.
    """
    print("\n=== Custom Gym Agent Example ===")
    
    class SimpleRandomAgent:
        """Simple random agent for demonstration."""
        
        def __init__(self, action_space):
            self.action_space = action_space
        
        def predict(self, observation, deterministic=True):
            """Predict action given observation."""
            if hasattr(self.action_space, 'sample'):
                action = self.action_space.sample()
            else:
                # Fallback for discrete spaces
                action = np.random.randint(0, self.action_space.n)
            return action, None
        
        def reset(self):
            """Reset the agent."""
            pass
    
    # Create configuration
    pricing_config = create_config_for_agent_type(AgentType.PRICING)
    
    # Create gym environment
    pricing_env = EVCHGymEnv(EVCHConfig(agent_type=AgentType.PRICING, **pricing_config))
    
    # Create custom agent
    custom_agent = SimpleRandomAgent(pricing_env.action_space)
    
    # Create adapter
    from utilities.rl_agents.adapters.gym_agent_adapter import create_gym_pricing_adapter
    pricing_adapter = create_gym_pricing_adapter(pricing_config, custom_agent)
    
    print("Custom gym agent created successfully!")
    print("This agent can be used with the simulation through the adapter.")


def example_with_rllib():
    """
    Example using RLlib agents.
    
    This shows how to integrate RLlib agents with the EVCH simulation.
    """
    print("\n=== RLlib Integration Example ===")
    
    # Configuration
    pricing_config = create_config_for_agent_type(AgentType.PRICING)
    
    # Example RLlib setup (commented out as RLlib is not installed)
    """
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.sac import SACConfig
    
    # Initialize Ray
    ray.init()
    
    # Configure SAC for pricing
    sac_config = SACConfig().environment(
        EVCHGymEnv, 
        env_config={"config": EVCHConfig(agent_type=AgentType.PRICING, **pricing_config)}
    ).framework("torch").training(
        learning_rate=0.0003,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256
    )
    
    # Train the agent
    results = tune.run(
        "SAC",
        config=sac_config.to_dict(),
        stop={"training_iteration": 100},
        checkpoint_freq=10
    )
    
    # Get the best agent
    best_agent = results.get_best_checkpoint("episode_reward_mean", "max")
    
    # Load the trained agent
    from ray.rllib.algorithms.sac import SAC
    trained_agent = SAC.from_checkpoint(best_agent)
    
    # Create adapter
    pricing_adapter = create_gym_pricing_adapter(pricing_config, trained_agent)
    
    # Use with simulation
    # agents_controller = build_gym_agents_controller(
    #     pricing_agent=trained_agent,
    #     pricing_config=pricing_config
    # )
    """
    
    print("RLlib agents would be created and trained here.")
    print("The trained agents can then be used with the simulation.")


def example_usage_with_simulation():
    """
    Example showing how to use the gym agents with the actual simulation.
    """
    print("\n=== Simulation Integration Example ===")
    
    # This would be the actual usage in your simulation
    """
    # 1. Create and train your gym agents (as shown in previous examples)
    
    # 2. Create agents controller
    agents_controller = build_gym_agents_controller(
        pricing_agent=trained_pricing_agent,
        charging_agent=trained_charging_agent,
        storage_agent=trained_storage_agent,
        pricing_config=pricing_config,
        charging_config=charging_config,
        storage_config=storage_config
    )
    
    # 3. Pass to operator
    operator = Operator(
        env=sim_env,
        requests=requests,
        chargers=chargers,
        # ... other parameters ...
        agents_controller=agents_controller
    )
    
    # 4. Run simulation
    # The operator will now use the gym-compatible agents through the adapters
    # All RL logic is completely decoupled from the simulation
    """
    
    print("The simulation would use the gym agents through the adapters.")
    print("All RL logic is completely decoupled from the simulation logic.")


if __name__ == "__main__":
    print("EVCH Gym Agent Integration Examples")
    print("=" * 50)
    
    # Run examples
    example_with_stable_baselines3()
    example_with_custom_gym_agent()
    example_with_rllib()
    example_usage_with_simulation()
    
    print("\n" + "=" * 50)
    print("Key Benefits of This Architecture:")
    print("1. Complete decoupling of RL agents from simulation")
    print("2. Easy integration with any gym-compatible RL library")
    print("3. Standardized interface for all agent types")
    print("4. No need to modify simulation code when changing RL agents")
    print("5. Support for both continuous and discrete action spaces")
    print("6. Easy testing and comparison of different RL algorithms")
