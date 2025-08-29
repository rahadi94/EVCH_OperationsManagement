from typing import Any, Dict, List, Optional
from utilities.rl_agents.interfaces import StorageAgent


class StorageEnvAgentAdapter(StorageAgent):
    """
    Adapter to wrap existing RL storage agents and environments to conform to the StorageAgent interface.
    
    This adapter provides a standardized interface for storage agents, allowing easy swapping
    of different RL algorithms while maintaining compatibility with the storage service.
    """
    
    def __init__(self, rl_agent: Any, storage_env: Any):
        """
        Initialize the storage agent adapter.
        
        Args:
            rl_agent: The underlying RL agent (e.g., SAC, DQN)
            storage_env: The storage environment (e.g., StorageEnv)
        """
        self.rl_agent = rl_agent
        self.storage_env = storage_env
        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.next_state = None
        self.done = False
    
    def reset(self) -> None:
        """Reset the agent and environment."""
        self.current_state = self.storage_env.reset()
        self.rl_agent.reset_game()
    
    def update_state(self, context: Dict[str, Any]) -> None:
        """
        Update the agent's state based on the current context.
        
        Args:
            context: Dictionary containing charging_hub and env
        """
        charging_hub = context.get("charging_hub")
        env = context.get("env")
        self.current_state = self.storage_env.get_state(charging_hub, env)
        self.rl_agent.state = self.current_state
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select a storage action based on current state.
        
        Args:
            context: Dictionary containing charging_hub and env
            
        Returns:
            Dictionary containing the selected storage action
        """
        charging_hub = context.get("charging_hub")
        eval_ep = self.rl_agent.do_evaluation_iterations
        self.rl_agent.episode_step_number_val = 0
        
        # Get action from the RL agent
        action_raw = self.rl_agent.pick_action(eval_ep)
        self.current_action = action_raw
        
        return {"storage_action": action_raw}
    
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
