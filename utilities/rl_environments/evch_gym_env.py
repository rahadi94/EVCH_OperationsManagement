import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from utilities.rl_environments.rl_pricing_env import PricingEnv
from utilities.rl_environments.SC_env import ChargingHubInvestmentEnv
from utilities.rl_environments.SC_storage_env import StorageEnv


class AgentType(Enum):
    """Enumeration of available agent types."""
    PRICING = "pricing"
    CHARGING = "charging"
    STORAGE = "storage"


@dataclass
class EVCHConfig:
    """Configuration for the EVCH gym environment."""
    agent_type: AgentType
    number_chargers: int
    number_power_options: int
    maximum_power: float
    maximum_grid_usage: float
    evaluation: bool = False
    pricing_mode: str = "Continuous"
    dynamic_fix_term_pricing: bool = False
    capacity_pricing: bool = False
    dynamic_parking_fee: bool = False
    limiting_grid_capa: bool = False
    dynamic_storage_scheduling: bool = False


class EVCHGymEnv(gym.Env):
    """
    Unified gym environment for EV Charging Hub operations.
    
    This environment completely decouples RL agents from the simulation,
    providing a standard gym interface that can be used with any gym-compatible
    RL library (Stable Baselines3, RLlib, etc.).
    
    The environment can be configured for different agent types:
    - PRICING: Dynamic pricing decisions
    - CHARGING: Charging optimization decisions  
    - STORAGE: Energy storage management decisions
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, config: EVCHConfig, charging_hub: Optional[Any] = None, sim_env: Optional[Any] = None):
        """
        Initialize the EVCH gym environment.
        
        Args:
            config: Configuration object specifying agent type and parameters
            charging_hub: Reference to the charging hub (will be set later if None)
            sim_env: Reference to the simulation environment (will be set later if None)
        """
        super().__init__()
        
        self.config = config
        self.agent_type = config.agent_type
        self.charging_hub = charging_hub
        self.sim_env = sim_env
        
        # Initialize the appropriate underlying environment
        self._init_underlying_env()
        
        # Set observation and action spaces
        self._set_spaces()
        
        # State tracking
        self.current_state = None
        self.current_action = None
        self.current_reward = 0.0
        self.done = False
        self.info = {}
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 1000  # Configurable
        
    def _init_underlying_env(self):
        """Initialize the underlying environment based on agent type."""
        if self.agent_type == AgentType.PRICING:
            self.underlying_env = PricingEnv(self.config, DQN=False)
        elif self.agent_type == AgentType.CHARGING:
            self.underlying_env = ChargingHubInvestmentEnv(self.config)
        elif self.agent_type == AgentType.STORAGE:
            self.underlying_env = StorageEnv(self.config)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    def _set_spaces(self):
        """Set observation and action spaces based on the underlying environment."""
        # Use the underlying environment's spaces
        self.observation_space = self.underlying_env.observation_space
        self.action_space = self.underlying_env.action_space
    
    def set_simulation_context(self, charging_hub: Any, sim_env: Any):
        """
        Set the simulation context (charging hub and environment).
        
        This method allows the gym environment to be connected to the actual
        simulation without tight coupling.
        
        Args:
            charging_hub: The charging hub object
            sim_env: The simulation environment
        """
        self.charging_hub = charging_hub
        self.sim_env = sim_env
        self.underlying_env.charging_hub = charging_hub
        self.underlying_env.env = sim_env
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset underlying environment
        if hasattr(self.underlying_env, 'reset'):
            self.current_state = self.underlying_env.reset()
        else:
            # Fallback: get initial state
            self.current_state = self._get_state()
        
        # Reset episode tracking
        self.episode_step = 0
        self.done = False
        self.current_reward = 0.0
        self.info = {}
        
        return self.current_state, self.info
    
    def step(self, action: Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take (numpy array or int)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.charging_hub is None or self.sim_env is None:
            raise RuntimeError("Simulation context not set. Call set_simulation_context() first.")
        
        # Store current action
        self.current_action = action
        
        # Apply action based on agent type
        reward = self._apply_action(action)
        
        # Get new state
        next_state = self._get_state()
        
        # Update state
        self.current_state = next_state
        self.current_reward = reward
        self.episode_step += 1
        
        # Check if episode is done
        terminated = self._is_episode_done()
        truncated = self.episode_step >= self.max_episode_steps
        self.done = terminated or truncated
        
        # Prepare info
        info = {
            "agent_type": self.agent_type.value,
            "episode_step": self.episode_step,
            "action": action,
            "reward": reward
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """Get the current state from the underlying environment."""
        if hasattr(self.underlying_env, 'get_state'):
            return self.underlying_env.get_state(self.charging_hub, self.sim_env)
        else:
            # Fallback: return zeros if no get_state method
            return np.zeros(self.observation_space.shape[0])
    
    def _apply_action(self, action: Union[np.ndarray, int]) -> float:
        """
        Apply the action and return the reward.
        
        Args:
            action: The action to apply
            
        Returns:
            The reward received
        """
        if self.agent_type == AgentType.PRICING:
            return self._apply_pricing_action(action)
        elif self.agent_type == AgentType.CHARGING:
            return self._apply_charging_action(action)
        elif self.agent_type == AgentType.STORAGE:
            return self._apply_storage_action(action)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
    
    def _apply_pricing_action(self, action: Union[np.ndarray, int]) -> float:
        """Apply pricing action and return reward."""
        # Store action in the charging hub's pricing agent
        if hasattr(self.charging_hub, 'pricing_agent'):
            self.charging_hub.pricing_agent.action = action
        
        # Apply the action using the underlying environment
        if hasattr(self.underlying_env, 'step'):
            _, reward, _, _, _ = self.underlying_env.step(action)
            return reward
        else:
            # Fallback: return 0 reward
            return 0.0
    
    def _apply_charging_action(self, action: np.ndarray) -> float:
        """Apply charging action and return reward."""
        # Store action in the charging hub's charging agent
        if hasattr(self.charging_hub, 'charging_agent'):
            self.charging_hub.charging_agent.action = action
        
        # Apply the action using the underlying environment
        if hasattr(self.underlying_env, 'step'):
            _, reward, _, _, _ = self.underlying_env.step(action)
            return reward
        else:
            # Fallback: return 0 reward
            return 0.0
    
    def _apply_storage_action(self, action: np.ndarray) -> float:
        """Apply storage action and return reward."""
        # Store action in the charging hub's storage agent
        if hasattr(self.charging_hub, 'storage_agent'):
            self.charging_hub.storage_agent.action = action
        
        # Apply the action using the underlying environment
        if hasattr(self.underlying_env, 'step'):
            _, reward, _, _, _ = self.underlying_env.step(action)
            return reward
        else:
            # Fallback: return 0 reward
            return 0.0
    
    def _is_episode_done(self) -> bool:
        """Check if the episode is done."""
        if hasattr(self.underlying_env, 'done'):
            return self.underlying_env.done
        else:
            # Fallback: episode is never done by default
            return False
    
    def render(self, mode: str = "human"):
        """Render the environment (placeholder for now)."""
        if mode == "human":
            print(f"EVCH Environment - Agent: {self.agent_type.value}, Step: {self.episode_step}")
        return None
    
    def close(self):
        """Close the environment."""
        if hasattr(self.underlying_env, 'close'):
            self.underlying_env.close()


# Factory functions for easy environment creation
def make_pricing_env(config_dict: Dict[str, Any], **kwargs) -> EVCHGymEnv:
    """Create a pricing environment."""
    config = EVCHConfig(agent_type=AgentType.PRICING, **config_dict)
    return EVCHGymEnv(config, **kwargs)


def make_charging_env(config_dict: Dict[str, Any], **kwargs) -> EVCHGymEnv:
    """Create a charging environment."""
    config = EVCHConfig(agent_type=AgentType.CHARGING, **config_dict)
    return EVCHGymEnv(config, **kwargs)


def make_storage_env(config_dict: Dict[str, Any], **kwargs) -> EVCHGymEnv:
    """Create a storage environment."""
    config = EVCHConfig(agent_type=AgentType.STORAGE, **config_dict)
    return EVCHGymEnv(config, **kwargs)


# Register environments with gym (optional, for gym.make() support)
try:
    from gym.envs.registration import register
    
    register(
        id='EVCH-Pricing-v0',
        entry_point='utilities.rl_environments.evch_gym_env:make_pricing_env',
    )
    
    register(
        id='EVCH-Charging-v0',
        entry_point='utilities.rl_environments.evch_gym_env:make_charging_env',
    )
    
    register(
        id='EVCH-Storage-v0',
        entry_point='utilities.rl_environments.evch_gym_env:make_storage_env',
    )
    
except ImportError:
    # gym registration not available
    pass
