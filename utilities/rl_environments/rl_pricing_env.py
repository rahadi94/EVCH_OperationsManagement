import gym
from gym import error, spaces, utils
import numpy as np
import logging
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass

from resources.configuration.configuration import Configuration


@dataclass
class PricingState:
    """Data class for pricing environment state information."""
    storage_soc: float
    pv_generation: float
    electricity_price: float
    peak_usage: float
    avg_energy_demand: float
    avg_power_demand: float
    free_grid_capacity: float


@dataclass
class PricingConfig:
    """Data class for pricing environment configuration."""
    number_power_options: int
    maximum_power: float
    evaluation: bool
    pricing_mode: str
    dynamic_fix_term_pricing: bool
    capacity_pricing: bool
    dynamic_parking_fee: bool
    limiting_grid_capa: bool
    dynamic_storage_scheduling: bool


class PricingEnv(gym.Env):
    """
    Gym environment for dynamic pricing in EV charging hub operations.
    
    This environment provides a standardized interface for RL agents to learn
    optimal pricing strategies for EV charging services.
    """
    
    metadata = {"render.modes": ["human"]}
    reward_range = (-float("inf"), float("inf"))
    spec = None
    
    # Constants
    K = 5  # Base for discrete action encoding
    MAX_EPISODE_STEPS = 50000000
    
    def __init__(self, config: Any, DQN: bool = False, charging_hub: Optional[Any] = None, env: Optional[Any] = None):
        """
        Initialize the pricing environment.
        
        Args:
            config: Configuration object containing environment parameters
            DQN: Whether to use discrete action space for DQN
            charging_hub: Reference to the charging hub
            env: Reference to the simulation environment
        """
        super().__init__()
        
        self.config = config
        self.evaluation = config.evaluation
        self.is_dqn = DQN
        
        # Single reward calculation approach
        # No longer using comprehensive rewards - keeping it simple
        
        # Initialize action and observation spaces
        self._init_action_space()
        self._init_observation_space()
        
        # Environment state
        self.charging_hub = charging_hub
        self.env = env
        self.current_step = 0
        self.reward = 0.0
        self.action = None
        self.final_action_DQN = None
        
        # Episode tracking
        self.episode = 0
        self.total_reward = {
            "profit": 0,
            "feasibility": 0,
            "energy": 0,
            "feasibility_storage": 0,
            "test": 0
        }
        
        # Action range for continuous actions
        if not DQN:
            self.action_range = [self.action_space.low, self.action_space.high]
    
    def _init_action_space(self) -> None:
        """Initialize the action space based on configuration."""
        if self.is_dqn:
            self._init_discrete_action_space()
        else:
            self._init_continuous_action_space()
    
    def _init_discrete_action_space(self) -> None:
        """Initialize discrete action space for DQN."""
        self.action_space = spaces.Discrete(self.K ** self.config.number_power_options)
    
    def _init_continuous_action_space(self) -> None:
        """Initialize continuous action space for other algorithms."""
        number_of_actions = self._calculate_number_of_actions()
        
        self.action_space = spaces.Box(
            low=0,
            high=self.config.maximum_power,
            shape=(number_of_actions,),
            dtype=np.float64
        )
        
        self._configure_action_space_bounds()
    
    def _calculate_number_of_actions(self) -> int:
        """Calculate the number of actions based on configuration."""
        config = Configuration.instance()
        number_of_actions = self.config.number_power_options - 1
        
        # Adjust based on pricing features
        if config.dynamic_fix_term_pricing and config.capacity_pricing:
            number_of_actions = self.config.number_power_options
        if config.dynamic_parking_fee:
            number_of_actions = self.config.number_power_options
        if config.limiting_grid_capa:
            number_of_actions = self.config.number_power_options
        if config.dynamic_storage_scheduling:
            number_of_actions = self.config.number_power_options
            
        return number_of_actions
    
    def _configure_action_space_bounds(self) -> None:
        """Configure action space bounds based on pricing mode."""
        config = Configuration.instance()
        
        if config.pricing_mode == "Discrete":
            self._configure_discrete_mode_bounds()
        elif config.pricing_mode == "Continuous":
            self._configure_continuous_mode_bounds()
    
    def _configure_discrete_mode_bounds(self) -> None:
        """Configure bounds for discrete pricing mode."""
        action_size = 2  # Default for discrete mode
        self.action_space = spaces.Box(
            low=0,
            high=self.config.maximum_power,
            shape=(action_size,),
            dtype=np.float64
        )
        
        # Set specific bounds for discrete pricing
        self.action_space.low[0] = 0.3
        self.action_space.high[0] = 1.5
        self.action_space.low[1] = 0.5
        self.action_space.high[1] = 1.5
        
        # Handle additional power options
        if self.config.number_power_options >= 3:
            self.action_space.low[2] = 300
            self.action_space.high[2] = 800
        if self.config.number_power_options >= 4:
            self.action_space.low[3] = -200
            self.action_space.high[3] = 200
    
    def _configure_continuous_mode_bounds(self) -> None:
        """Configure bounds for continuous pricing mode."""
        config = Configuration.instance()
        
        # Base bounds
        self.action_space.low[0] = 0
        self.action_space.high[0] = 1.5
        
        # Adjust based on features
        if config.limiting_grid_capa:
            self.action_space.low[1] = 300
            self.action_space.high[1] = 600
        if config.dynamic_storage_scheduling:
            self.action_space.low[1] = -200
            self.action_space.high[1] = 200
        if config.dynamic_fix_term_pricing and config.capacity_pricing:
            self.action_space.low[0] = 0.5
            self.action_space.high[0] = 1.5
            self.action_space.low[1] = 0
            self.action_space.high[1] = 0.4
        if config.dynamic_fix_term_pricing and not config.capacity_pricing:
            self.action_space.low[0] = 0.6
            self.action_space.high[0] = 1.5
            if config.dynamic_parking_fee:
                self.action_space.low[1] = 0
                self.action_space.high[1] = 1 / 60
    
    def _init_observation_space(self) -> None:
        """Initialize the observation space."""
        observation_shape = self._calculate_observation_shape()
        
        self.observation_space = spaces.Box(
            low=0,
            high=1000000,
            shape=(observation_shape,),
            dtype=np.float64
        )
    
    def _calculate_observation_shape(self) -> int:
        """Calculate the observation space shape."""
        base_shape = 2 + 3 + 2  # Time encoding + base features + demand features
        
        if Configuration.instance().dynamic_storage_scheduling:
            base_shape += 1
            
        return base_shape
    
    def rescale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale action from normalized range to actual range.
        
        Args:
            action: Normalized action from agent
            
        Returns:
            Rescaled action in actual range
        """
        return (
            action * (self.action_range[1] - self.action_range[0]) / 2.0
            + (self.action_range[1] + self.action_range[0]) / 2.0
        )
    
    def get_final_prices_DQN(self, actions: np.ndarray) -> np.ndarray:
        """
        Convert DQN actions to final pricing values.
        
        Args:
            actions: Raw actions from DQN agent
            
        Returns:
            Final pricing values
        """
        final_action = actions.copy()
        
        if len(actions) == 1:
            final_action[0] = actions[0] * 0.1 + 0.3
        elif len(actions) == 2:
            for i in range(len(actions)):
                final_action[i] = actions[i] * 0.1 + 0.4 * i + 0.2 * (1 - i)
        
        self.final_action_DQN = final_action
        return final_action
    
    def get_state(self, charging_hub: Optional[Any] = None, env: Optional[Any] = None) -> np.ndarray:
        """
        Get the current state of the environment.
        
        Args:
            charging_hub: Reference to the charging hub
            env: Reference to the simulation environment
            
        Returns:
            State vector as numpy array
        """
        state = np.array([])
        
        # Add time encoding
        time_encoding = self._get_time_encoding(charging_hub, env)
        state = np.append(state, time_encoding)
        
        # Add system state
        system_state = self._get_system_state(charging_hub, env)
        state = np.append(state, system_state)
        
        return state
    
    def _get_time_encoding(self, charging_hub: Optional[Any], env: Optional[Any]) -> np.ndarray:
        """
        Get time encoding for the current state.
        
        Args:
            charging_hub: Reference to the charging hub
            env: Reference to the simulation environment
            
        Returns:
            Time encoding vector
        """
        if not env:
            return self._get_default_time_encoding()
        else:
            return self._get_simulation_time_encoding(charging_hub, env)
    
    def _get_default_time_encoding(self) -> np.ndarray:
        """Get default time encoding when no simulation is running."""
        hour = 0
        normalized_hour = hour / 24 / 4
        angle = normalized_hour * 2 * np.pi
        
        sin_encoding = np.sin(angle)
        cos_encoding = np.cos(angle)
        
        return np.array([sin_encoding, cos_encoding])
    
    def _get_simulation_time_encoding(self, charging_hub: Any, env: Any) -> np.ndarray:
        """
        Get time encoding from simulation.
        
        Args:
            charging_hub: Reference to the charging hub
            env: Reference to the simulation environment
            
        Returns:
            Time encoding vector
        """
        hour = (
            env.now % 1440 - env.now % charging_hub.planning_interval
        ) / charging_hub.planning_interval
        hour = int(hour)
        
        normalized_hour = hour / 24 / (60 / charging_hub.planning_interval)
        angle = normalized_hour * 2 * np.pi
        
        sin_encoding = np.sin(angle)
        cos_encoding = np.cos(angle)
        
        return np.array([sin_encoding, cos_encoding])
    
    def _get_system_state(self, charging_hub: Optional[Any], env: Optional[Any]) -> np.ndarray:
        """
        Get system state information.
        
        Args:
            charging_hub: Reference to the charging hub
            env: Reference to the simulation environment
            
        Returns:
            System state vector
        """
        if not charging_hub:
            return self._get_default_system_state()
        else:
            return self._get_simulation_system_state(charging_hub, env)
    
    def _get_default_system_state(self) -> np.ndarray:
        """Get default system state when no simulation is running."""
        default_state = [0, 0, 0, 0, 0, 0]  # All zeros for default
        
        if Configuration.instance().dynamic_storage_scheduling:
            return np.array(default_state)
        else:
            return np.array(default_state[1:])  # Exclude storage SoC
    
    def _get_simulation_system_state(self, charging_hub: Any, env: Any) -> np.ndarray:
        """
        Get system state from simulation.
        
        Args:
            charging_hub: Reference to the charging hub
            env: Reference to the simulation environment
            
        Returns:
            System state vector
        """
        pricing_state = self._extract_pricing_state(charging_hub, env)
        
        # Normalize values
        normalized_state = [
            pricing_state.storage_soc / 300,
            pricing_state.pv_generation / 500,
            pricing_state.electricity_price,
            pricing_state.peak_usage / 1000,
            pricing_state.avg_energy_demand / 1000,
            pricing_state.avg_power_demand / 10
        ]
        
        if Configuration.instance().dynamic_storage_scheduling:
            return np.array(normalized_state)
        else:
            return np.array(normalized_state[1:])  # Exclude storage SoC
    
    def _extract_pricing_state(self, charging_hub: Any, env: Any) -> PricingState:
        """
        Extract pricing state from charging hub.
        
        Args:
            charging_hub: Reference to the charging hub
            env: Reference to the simulation environment
            
        Returns:
            PricingState object
        """
        # Extract storage state
        storage_soc = charging_hub.electric_storage.SoC
        
        # Extract PV generation with error handling
        try:
            pv_generation = charging_hub.operator.non_dispatchable_generator.generation_profile_actual.loc[
                env.now, "pv_generation"
            ]
        except KeyError:
            # If time index doesn't exist, return 0 (no generation)
            pv_generation = 0.0
        
        # Extract electricity price
        hour = (env.now % 1440 - env.now % 60) / 60
        electricity_price = charging_hub.electricity_tariff[int(hour)]
        
        # Extract peak usage
        peak_usage = charging_hub.operator.peak_threshold
        
        # Calculate demand metrics
        avg_energy_demand, avg_power_demand = self._calculate_demand_metrics(charging_hub)
        
        # Extract grid capacity
        free_grid_capacity = self._extract_grid_capacity(charging_hub)
        
        return PricingState(
            storage_soc=storage_soc,
            pv_generation=pv_generation,
            electricity_price=electricity_price,
            peak_usage=peak_usage,
            avg_energy_demand=avg_energy_demand,
            avg_power_demand=avg_power_demand,
            free_grid_capacity=free_grid_capacity
        )
    
    def _calculate_demand_metrics(self, charging_hub: Any) -> Tuple[float, float]:
        """
        Calculate average energy and power demand.
        
        Args:
            charging_hub: Reference to the charging hub
            
        Returns:
            Tuple of (avg_energy_demand, avg_power_demand)
        """
        avg_energy_demand = 0
        avg_power_demand = 0
        
        for charger in charging_hub.chargers:
            vehicles = charger.connected_vehicles
            for vehicle in vehicles:
                avg_energy_demand += vehicle.remaining_energy_deficit
                avg_power_demand += (
                    vehicle.remaining_energy_deficit / vehicle.remaining_park_duration
                )
        
        return avg_energy_demand, avg_power_demand
    
    def _extract_grid_capacity(self, charging_hub: Any) -> float:
        """
        Extract free grid capacity.
        
        Args:
            charging_hub: Reference to the charging hub
            
        Returns:
            Free grid capacity
        """
        try:
            free_grid_capa = charging_hub.operator.free_grid_capa_actual
            if isinstance(free_grid_capa, list):
                if len(free_grid_capa) > 0:
                    return free_grid_capa[0]
                else:
                    return 0.0
            else:
                return free_grid_capa
        except Exception as e:
            # If we can't access grid capacity, return 0
            return 0.0
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1
        reward = self._take_action(action)
        done = self.current_step >= self.MAX_EPISODE_STEPS
        obs = self._next_observation()
        
        return obs, reward, done, {}
    
    def receive_action(self) -> Optional[np.ndarray]:
        """Get the current action."""
        return self.action
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.reward = 0
        
        if not self.charging_hub:
            return self.get_state(None, None)
        return self.get_state(self.charging_hub, self.env)
    
    def render(self, mode: str = "human", close: bool = False) -> None:
        """Render the environment state."""
        if mode == "human":
            print(f"Pricing Environment - Reward: {self.reward}")
    
    def _take_action(self, action: np.ndarray) -> float:
        """
        Execute the action and calculate reward.
        
        Args:
            action: Action to execute
            
        Returns:
            Reward value
        """
        # Store the action for the simulation to use
        self.action = action
        
        # Calculate reward using the single reward calculation method
        reward = self._calculate_reward(action)
        
        return reward
    

    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward using the simulation-based approach with operator.reward_computing().
        
        This method uses the operator's reward computation as the primary source
        and includes additional reward components for better learning.
        
        Args:
            action: Action to execute
            
        Returns:
            float: Calculated reward value
        """
        reward = 0.0
        
        if self.charging_hub:
            # 1. OPERATOR REWARD COMPUTING (primary reward source)
            # Use the operator's reward_computing method which includes:
            # - Peak threshold violation penalties
            # - Objective function change rewards
            if hasattr(self.charging_hub, 'operator') and self.charging_hub.operator:
                try:
                    operator_reward = self.charging_hub.operator.reward_computing()
                    reward += operator_reward
                except Exception as e:
                    print(f"Operator reward computing failed: {e}")
            
            # 2. MISSED PENALTIES (from simulation)
            profit = self.charging_hub.reward.get("profit", 0)
            reward -= profit
            self.total_reward["profit"] += reward
            
            # Reset rewards for next step
            self.charging_hub.reward["profit"] = 0
            self.charging_hub.reward["feasibility_storage"] = 0
            self.charging_hub.reward["feasibility"] = 0
            
            # 3. FALLBACK REWARD (ensure non-zero rewards for learning)
            if reward == 0.0:
                reward = 0.01
            
            # Debug output (only print occasionally to avoid spam)
            if hasattr(self, 'current_step'):
                self.current_step += 1
            else:
                self.current_step = 1
        
        return reward
    
    def _next_observation(self) -> np.ndarray:
        """
        Get the next observation.
        
        Returns:
            Next observation
        """
        return self.get_state(self.charging_hub, self.env)


def convert_to_scalar(action_vector: np.ndarray) -> int:
    """
    Convert action vector to scalar for discrete actions.
    
    Args:
        action_vector: Vector of actions
        
    Returns:
        Scalar action value
    """
    action = 0
    for i in range(2):
        action += action_vector[i] * (5) ** (1 - i)
    return int(action)


def convert_to_vector(scalar_action: int, h: int = 1) -> np.ndarray:
    """
    Convert scalar action to vector for discrete actions.
    
    Args:
        scalar_action: Scalar action value
        h: Height parameter for conversion
        
    Returns:
        Vector of actions
    """
    action = np.zeros(2)
    j = 0
    
    for i in range(2):
        action[i] = int((scalar_action - scalar_action % (5 ** (h - j))) / (5 ** (h - j)))
        scalar_action = scalar_action % (5 ** (h - j))
        j += 1
    
    return action
