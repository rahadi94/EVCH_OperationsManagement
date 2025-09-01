from typing import Any, Dict, List, Optional, Union
import numpy as np
from utilities.rl_agents.interfaces import PricingAgent, ChargingAgent, StorageAgent
from utilities.rl_environments.evch_gym_env import EVCHGymEnv, AgentType


class GymAgentAdapter:
    """
    Adapter for standard gym-compatible RL agents to work with EVCH simulation.
    
    This adapter allows any gym-compatible RL agent (Stable Baselines3, RLlib, etc.)
    to be used with the EVCH simulation by providing a standardized interface.
    """
    
    def __init__(self, gym_env: EVCHGymEnv, gym_agent: Any):
        """
        Initialize the gym agent adapter.
        
        Args:
            gym_env: The EVCH gym environment
            gym_agent: The gym-compatible RL agent (must have predict() method)
        """
        self.gym_env = gym_env
        self.gym_agent = gym_agent
        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.next_state = None
        self.done = False
        
        # Validate that the agent has the required methods
        if not hasattr(self.gym_agent, 'predict'):
            raise ValueError("Gym agent must have a 'predict' method")
    
    def reset(self) -> None:
        """Reset the agent and environment."""
        self.current_state, _ = self.gym_env.reset()
        if hasattr(self.gym_agent, 'reset'):
            self.gym_agent.reset()
    
    def update_state(self, context: Dict[str, Any]) -> None:
        """
        Update the agent's state based on the current context.
        
        Args:
            context: Dictionary containing charging_hub and env
        """
        # Set the simulation context in the gym environment
        charging_hub = context.get("charging_hub")
        sim_env = context.get("env")
        if charging_hub and sim_env:
            self.gym_env.set_simulation_context(charging_hub, sim_env)
        
        # Get current state from the gym environment
        self.current_state = self.gym_env._get_state()
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select an action using the gym agent.
        
        Args:
            context: Dictionary containing charging_hub and env
            
        Returns:
            Dictionary containing the selected action
        """
        # Update state first
        self.update_state(context)
        
        # Use the gym agent to predict action
        if hasattr(self.gym_agent, 'predict'):
            # Standard gym agent interface
            action, _ = self.gym_agent.predict(self.current_state, deterministic=True)
        elif hasattr(self.gym_agent, 'act'):
            # Alternative interface
            action = self.gym_agent.act(self.current_state)
        else:
            # Fallback: assume agent is callable
            action = self.gym_agent(self.current_state)
        
        self.current_action = action
        
        # Return action in the format expected by the service
        if self.gym_env.agent_type == AgentType.PRICING:
            return {"pricing_parameters": action}
        elif self.gym_env.agent_type == AgentType.CHARGING:
            return {"charging_action": action}
        elif self.gym_env.agent_type == AgentType.STORAGE:
            return {"storage_action": action}
        else:
            return {"action": action}
    
    def learn(self, transition: Dict[str, Any]) -> None:
        """
        Learn from the transition experience.
        
        Args:
            transition: Dictionary containing state, action, reward, next_state, done
        """
        # Most gym agents handle learning internally during training
        # This method is called for compatibility but may not be used
        pass
    
    def train(self, total_timesteps: int = 1000) -> None:
        """
        Train the gym agent.
        
        Args:
            total_timesteps: Number of timesteps to train for
        """
        if hasattr(self.gym_agent, 'learn'):
            self.gym_agent.learn(total_timesteps=total_timesteps)
        else:
            raise NotImplementedError("Gym agent does not have a 'learn' method")


class GymPricingAgentAdapter(GymAgentAdapter, PricingAgent):
    """Adapter for gym agents used as pricing agents."""
    
    def __init__(self, gym_env: EVCHGymEnv, gym_agent: Any):
        super().__init__(gym_env, gym_agent)
        if gym_env.agent_type != AgentType.PRICING:
            raise ValueError("Gym environment must be configured for pricing agent")


class GymChargingAgentAdapter(GymAgentAdapter, ChargingAgent):
    """Adapter for gym agents used as charging agents."""
    
    def __init__(self, gym_env: EVCHGymEnv, gym_agent: Any):
        super().__init__(gym_env, gym_agent)
        if gym_env.agent_type != AgentType.CHARGING:
            raise ValueError("Gym environment must be configured for charging agent")
    
    def select_action(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select charging action based on vehicles and context.
        
        Args:
            vehicles: List of vehicles to consider for charging
            context: Dictionary containing charging_hub and env
            
        Returns:
            Dictionary containing the selected charging action
        """
        # Add vehicles to context for the gym environment
        context_with_vehicles = context.copy()
        context_with_vehicles["vehicles"] = vehicles
        
        return super().select_action(context_with_vehicles)


class GymStorageAgentAdapter(GymAgentAdapter, StorageAgent):
    """Adapter for gym agents used as storage agents."""
    
    def __init__(self, gym_env: EVCHGymEnv, gym_agent: Any):
        super().__init__(gym_env, gym_agent)
        if gym_env.agent_type != AgentType.STORAGE:
            raise ValueError("Gym environment must be configured for storage agent")


# Factory functions for easy adapter creation
def create_gym_pricing_adapter(config_dict: Dict[str, Any], gym_agent: Any, **kwargs) -> GymPricingAgentAdapter:
    """Create a gym pricing agent adapter."""
    from utilities.rl_environments.evch_gym_env import make_pricing_env
    gym_env = make_pricing_env(config_dict, **kwargs)
    return GymPricingAgentAdapter(gym_env, gym_agent)


def create_gym_charging_adapter(config_dict: Dict[str, Any], gym_agent: Any, **kwargs) -> GymChargingAgentAdapter:
    """Create a gym charging agent adapter."""
    from utilities.rl_environments.evch_gym_env import make_charging_env
    gym_env = make_charging_env(config_dict, **kwargs)
    return GymChargingAgentAdapter(gym_env, gym_agent)


def create_gym_storage_adapter(config_dict: Dict[str, Any], gym_agent: Any, **kwargs) -> GymStorageAgentAdapter:
    """Create a gym storage agent adapter."""
    from utilities.rl_environments.evch_gym_env import make_storage_env
    gym_env = make_storage_env(config_dict, **kwargs)
    return GymStorageAgentAdapter(gym_env, gym_agent)
