from typing import Any, Dict, List, Optional
from utilities.rl_agents.interfaces import ChargingAgent


class ChargingEnvAgentAdapter(ChargingAgent):
    """
    Adapter to wrap existing RL charging agents and environments to conform to the ChargingAgent interface.
    
    This adapter provides a standardized interface for charging agents, allowing easy swapping
    of different RL algorithms while maintaining compatibility with the charging service.
    """
    
    def __init__(self, rl_agent: Any, charging_env: Any):
        """
        Initialize the charging agent adapter.
        
        Args:
            rl_agent: The underlying RL agent (e.g., SAC, DQN)
            charging_env: The charging environment (e.g., ChargingHubInvestmentEnv)
        """
        self.rl_agent = rl_agent
        self.charging_env = charging_env
        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.next_state = None
        self.done = False
    
    def reset(self) -> None:
        """Reset the agent and environment."""
        self.current_state = self.charging_env.reset()
        self.rl_agent.reset_game()
    
    def update_state(self, context: Dict[str, Any]) -> None:
        """
        Update the agent's state based on the current context.
        
        Args:
            context: Dictionary containing charging_hub and env
        """
        charging_hub = context.get("charging_hub")
        env = context.get("env")
        self.current_state = self.charging_env.get_state(charging_hub, env)
        self.rl_agent.state = self.current_state
    
    def select_action(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select a charging action based on current state and vehicles.
        
        Args:
            vehicles: List of vehicles to consider for charging
            context: Dictionary containing charging_hub and env
            
        Returns:
            Dictionary containing the selected charging action
        """
        charging_hub = context.get("charging_hub")
        eval_ep = self.rl_agent.do_evaluation_iterations
        self.rl_agent.episode_step_number_val = 0
        
        # Get action from the RL agent
        action_raw = self.rl_agent.pick_action(eval_ep, charging_hub)
        self.current_action = action_raw
        
        # Rescale action if needed
        rescaled_action = self.rl_agent.rescale_action(action_raw)
        
        return {"charging_action": rescaled_action}
    
    def learn(self, transition: Dict[str, Any]) -> None:
        """
        Learn from the transition experience.
        
        Args:
            transition: Dictionary containing state, action, reward, next_state, done
        """
        # The RL agent's internal learn method is typically called
        # by the agent itself after its conduct_action.
        # If explicit learning is needed, it would be handled here.
        pass
