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
            self.underlying_env = PricingEnv(self.config, DQN=False, charging_hub=None, env=None)
        elif self.agent_type == AgentType.CHARGING:
            self.underlying_env = ChargingHubInvestmentEnv(self.config, charging_hub=None, env=None)
        elif self.agent_type == AgentType.STORAGE:
            self.underlying_env = StorageEnv(self.config, charging_hub=None, env=None)
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
        
        # Get new state with error handling
        try:
            next_state = self._get_state()
        except Exception as e:
            # If we can't get the state (e.g., simulation ended), return zeros and mark as terminated
            print(f"Could not get state: {e}. Marking episode as terminated.")
            next_state = np.zeros(self.observation_space.shape[0])
            terminated = True
            truncated = False
            self.done = True
            info = {
                "agent_type": self.agent_type.value,
                "episode_step": self.episode_step,
                "action": action,
                "reward": reward,
                "error": str(e)
            }
            return next_state, reward, terminated, truncated, info
        
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
        # Store action in the charging hub's pricing agent if it exists
        if hasattr(self.charging_hub, 'pricing_agent') and self.charging_hub.pricing_agent is not None:
            self.charging_hub.pricing_agent.action = action
        
        # Apply the pricing action to the pricing parameters for continuous pricing
        if hasattr(self.charging_hub, 'operator'):
            try:
                # Get the pricing mode from configuration
                from resources.configuration.configuration import Configuration
                config = Configuration.instance()
                pricing_mode = getattr(config, 'pricing_mode', 'Continuous')
                
                if pricing_mode == "Continuous":
                    # For continuous pricing, apply the action to pricing_parameters
                    if hasattr(self.charging_hub.operator, 'pricing_parameters'):
                        # Apply action to pricing parameters:
                        # pricing_parameters[0] = fixed term (p_0) - energy price component
                        # pricing_parameters[1] = rate-based term (alpha) - capacity price component
                        if len(action) >= 2:
                            self.charging_hub.operator.pricing_parameters[0] = action[0]  # Fixed term (p_0)
                            self.charging_hub.operator.pricing_parameters[1] = action[1]  # Rate-based term (alpha)
                        elif len(action) == 1:
                            self.charging_hub.operator.pricing_parameters[0] = action[0]  # Fixed term (p_0)
                    
                    # For continuous pricing, we don't use price_pairs directly
                    # Instead, vehicles calculate their own price using the price_function:
                    # price = p_0 + alpha * power^degree
                    print(f"Applied continuous pricing action {action} to pricing_parameters: {self.charging_hub.operator.pricing_parameters}")
                
                elif pricing_mode == "Discrete":
                    # For discrete pricing, apply to price_pairs
                    if hasattr(self.charging_hub.operator, 'price_pairs'):
                        from utilities.rl_environments.rl_pricing_env import convert_to_vector
                        if isinstance(action, (int, np.integer)):
                            vector_prices = convert_to_vector(action)
                            for i, price in enumerate(vector_prices):
                                if i < len(self.charging_hub.operator.price_pairs):
                                    self.charging_hub.operator.price_pairs[i, 1] = price
                        else:
                            # Action is already a vector
                            for i, price in enumerate(action):
                                if i < len(self.charging_hub.operator.price_pairs):
                                    self.charging_hub.operator.price_pairs[i, 1] = price
                        print(f"Applied discrete pricing action {action} to price_pairs: {self.charging_hub.operator.price_pairs}")
                
            except Exception as e:
                print(f"Failed to apply pricing action: {e}")
        
        # Advance the simulation by one planning interval (typically 60 minutes)
        planning_interval = getattr(self.charging_hub, 'planning_interval', 60)
        current_time = self.sim_env.now
        next_time = current_time + planning_interval
        
        # Run the simulation until the next time step
        try:
            self.sim_env.run(until=next_time)
        except Exception as e:
            # If simulation fails (e.g., data not available), mark as terminated
            print(f"Simulation ended at time {self.sim_env.now} due to data limits")
            return 0.0  # Return neutral reward
        
        # Choose reward calculation method based on configuration
        if self.config.use_comprehensive_rewards:
            # Use comprehensive reward calculation that consolidates simulation-based logic
            reward = self._calculate_comprehensive_reward()
        else:
            # Use original simulation-based reward calculation
            reward = self._calculate_simulation_based_reward()
        return reward
    
    def _apply_charging_action(self, action: np.ndarray) -> float:
        """Apply charging action and return reward."""
        # Store action in the charging hub's charging agent if it exists
        if hasattr(self.charging_hub, 'charging_agent') and self.charging_hub.charging_agent is not None:
            self.charging_hub.charging_agent.action = action
        
        # Advance the simulation by one planning interval (typically 60 minutes)
        planning_interval = getattr(self.charging_hub, 'planning_interval', 60)
        current_time = self.sim_env.now
        next_time = current_time + planning_interval
        
        # Run the simulation until the next time step
        try:
            self.sim_env.run(until=next_time)
        except Exception as e:
            # If simulation fails (e.g., data not available), mark as terminated
            print(f"Simulation ended at time {self.sim_env.now} due to data limits")
            return 0.0  # Return neutral reward
        
        # Choose reward calculation method based on configuration
        if self.config.use_comprehensive_rewards:
            # Use comprehensive reward calculation that consolidates simulation-based logic
            reward = self._calculate_comprehensive_reward()
        else:
            # Use original simulation-based reward calculation
            reward = self._calculate_simulation_based_reward()
        return reward
    
    def _apply_storage_action(self, action: np.ndarray) -> float:
        """Apply storage action and return reward."""
        # Store action in the charging hub's storage agent if it exists
        if hasattr(self.charging_hub, 'storage_agent') and self.charging_hub.storage_agent is not None:
            self.charging_hub.storage_agent.action = action
        
        # Advance the simulation by one planning interval (typically 60 minutes)
        planning_interval = getattr(self.charging_hub, 'planning_interval', 60)
        current_time = self.sim_env.now
        next_time = current_time + planning_interval
        
        # Run the simulation until the next time step
        try:
            self.sim_env.run(until=next_time)
        except Exception as e:
            # If simulation fails (e.g., data not available), mark as terminated
            print(f"Simulation ended at time {self.sim_env.now} due to data limits")
            return 0.0  # Return neutral reward
        
        # Choose reward calculation method based on configuration
        if self.config.use_comprehensive_rewards:
            # Use comprehensive reward calculation that consolidates simulation-based logic
            reward = self._calculate_comprehensive_reward()
        else:
            # Use original simulation-based reward calculation
            reward = self._calculate_simulation_based_reward()
        return reward
    
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

    def _calculate_comprehensive_reward(self) -> float:
        """
        Calculate comprehensive reward by moving simulation-based reward logic to the RL environment.
        
        This method consolidates reward calculation that was previously scattered across
        simulation components, making the RL environment more self-contained.
        
        Returns:
            float: Calculated reward value
        """
        if not self.charging_hub or not hasattr(self.charging_hub, 'operator'):
            return 0.0
        
        try:
            reward = 0.0
            operator = self.charging_hub.operator
            
            # 1. GRID USAGE AND PEAK PENALTIES
            current_grid_usage = max(self.charging_hub.grid.grid_usage) if self.charging_hub.grid.grid_usage else 0
            current_peak_threshold = operator.peak_threshold
            
            # Peak threshold violation penalty (from operator.reward_computing)
            if current_grid_usage > current_peak_threshold:
                peak_penalty = (current_grid_usage - current_peak_threshold) * operator.peak_cost
                reward -= peak_penalty
                # Update peak threshold (as done in simulation)
                operator.peak_threshold = current_grid_usage
            
            # 2. OBJECTIVE FUNCTION-BASED REWARD (from simulation model)
            # Calculate revenue and costs similar to simulation model
            total_revenue = 0.0
            total_energy_costs = self.charging_hub.grid.energy_costs if hasattr(self.charging_hub.grid, 'energy_costs') else 0.0
            
            # Calculate revenue from served vehicles
            requests = [r for r in operator.requests if r.ev == 1]
            for request in requests:
                if request.is_assigned and request.energy_requested > 0:
                    # Revenue from energy charged
                    energy_charged = min(request.energy_charged, request.energy_requested)
                    total_revenue += energy_charged * request.charging_price
                    
                    # Revenue from parking fees
                    total_revenue += request.park_duration * request.parking_fee
                    
                    # Penalty for missed energy (from simulation model)
                    energy_missed = max(request.energy_requested - request.energy_charged, 0)
                    if energy_missed > 0:
                        from resources.configuration.configuration import Configuration
                        missed_penalty = energy_missed * request.charging_price * Configuration.instance().energy_missed_penalty
                        total_revenue -= missed_penalty
            
            # Calculate operational costs
            operational_costs = total_energy_costs
            
            # Peak charges (if applicable)
            if hasattr(self.charging_hub, 'prices') and 'peak' in self.charging_hub.prices:
                peak_charge = max((current_grid_usage - operator.peak_threshold) * self.charging_hub.prices['peak'], 0)
                operational_costs += peak_charge
            
            # 3. OBJECTIVE FUNCTION CALCULATION (from simulation model)
            if hasattr(self.charging_hub, 'objective'):
                if self.charging_hub.objective == "min_costs":
                    # Calculate missed energy penalty
                    total_energy_missed = sum(
                        max(r.energy_requested - r.energy_charged, 0) 
                        for r in requests if r.energy_requested > 0
                    )
                    missed_penalty = total_energy_missed * getattr(self.charging_hub, 'penalty_for_missed_kWh', 1.0)
                    objective_value = missed_penalty + operational_costs
                elif self.charging_hub.objective == "max_profits":
                    objective_value = total_revenue - operational_costs
                else:
                    objective_value = total_revenue - operational_costs
            else:
                objective_value = total_revenue - operational_costs
            
            # 4. OBJECTIVE FUNCTION CHANGE REWARD (from operator.reward_computing)
            if hasattr(operator, 'objective'):
                objective_change = objective_value - operator.objective
                reward -= objective_change  # Negative because we want to minimize costs/maximize profits
                operator.objective = objective_value
            else:
                operator.objective = objective_value
            
            # 5. SERVICE LEVEL REWARD (from simulation model)
            served_requests = sum(1 for r in requests if r.energy_requested > 0 and r.energy_charged > 0)
            total_requests = sum(1 for r in requests if r.energy_requested > 0)
            if total_requests > 0:
                service_level = served_requests / total_requests
                service_reward = service_level * 0.5  # Reward for good service level
                reward += service_reward
            
            # 6. PRICING-SPECIFIC REWARDS (existing logic)
            if self.agent_type == AgentType.PRICING and hasattr(operator, 'pricing_parameters'):
                pricing_params = operator.pricing_parameters
                if len(pricing_params) >= 2:
                    base_price = pricing_params[0]
                    capacity_price = pricing_params[1]
                    
                    # Count rejected vehicles due to pricing
                    rejected_vehicles = sum(1 for r in requests 
                                          if r.ev == 1 and r.energy_requested == 0 and r.charging_price > 0)
                    
                    # Strong penalty for pricing that causes vehicle rejections
                    if rejected_vehicles > 0:
                        rejection_penalty = rejected_vehicles * 0.5
                        reward -= rejection_penalty
                    
                    # Penalty for extreme pricing
                    if base_price > 1.2 or base_price < 0.3:
                        reward -= 0.3
                    elif 0.4 <= base_price <= 1.0:
                        reward += 0.3
                    
                    if capacity_price > 0.3 or capacity_price < 0.01:
                        reward -= 0.2
                    elif 0.02 <= capacity_price <= 0.2:
                        reward += 0.2
            
            # 7. GRID EFFICIENCY REWARD (existing logic)
            if current_grid_usage <= current_peak_threshold:
                efficiency_ratio = current_grid_usage / current_peak_threshold
                if 0.7 <= efficiency_ratio <= 0.9:
                    reward += 0.3
                elif efficiency_ratio < 0.3:
                    reward -= 0.1
                elif efficiency_ratio > 0.95:
                    reward -= 0.2
            
            # 8. VEHICLE SERVICE REWARD (existing logic)
            active_vehicles = sum(1 for r in requests if r.mode in ["Connected", "Charging"])
            if active_vehicles > 0:
                service_reward = min(active_vehicles * 0.3, 2.0)
                reward += service_reward
            
            # 9. COMPLETION REWARD (existing logic)
            completed_vehicles = sum(1 for r in requests if r.mode in ["Fully_charged", "Left"])
            completion_reward = completed_vehicles * 0.1
            reward += completion_reward
            
            # 10. TIME-BASED REWARD (existing logic)
            time_factor = (self.sim_env.now % 1440) / 1440
            time_reward = 0.1 * time_factor
            reward += time_reward
            
            return reward
            
        except Exception as e:
            print(f"Comprehensive reward calculation failed: {e}")
            return 0.0

    def _calculate_simulation_based_reward(self) -> float:
        """
        Calculate reward using the original simulation-based approach.
        
        This method maintains backward compatibility with the existing
        simulation-based reward calculation logic.
        
        Returns:
            float: Calculated reward value
        """
        # This method contains the original reward calculation logic
        # that was previously in the pricing action method
        if hasattr(self.charging_hub, 'operator'):
            try:
                # Get current state metrics
                current_grid_usage = max(self.charging_hub.grid.grid_usage) if self.charging_hub.grid.grid_usage else 0
                current_peak_threshold = self.charging_hub.operator.peak_threshold
                
                # Count active vehicles (charging or connected)
                active_vehicles = sum(1 for request in self.charging_hub.operator.requests 
                                    if request.mode in ["Connected", "Charging"] and request.ev == 1)
                
                # Count completed vehicles (fully charged or left)
                completed_vehicles = sum(1 for request in self.charging_hub.operator.requests 
                                       if request.mode in ["Fully_charged", "Left"] and request.ev == 1)
                
                # Count vehicles that rejected charging due to high prices
                rejected_vehicles = sum(1 for request in self.charging_hub.operator.requests 
                                      if request.ev == 1 and request.energy_requested == 0 and request.charging_price > 0)
                
                # Calculate dynamic reward components
                reward = 0.0
                
                # 1. Grid usage penalty (negative reward for exceeding peak threshold)
                if current_grid_usage > current_peak_threshold:
                    penalty = (current_grid_usage - current_peak_threshold) * 0.1
                    reward -= penalty
                
                # 2. Dynamic service reward based on current pricing action
                if hasattr(self.charging_hub.operator, 'pricing_parameters'):
                    pricing_params = self.charging_hub.operator.pricing_parameters
                    if len(pricing_params) >= 2:
                        # Reward for optimal pricing (not too high, not too low)
                        base_price = pricing_params[0]
                        capacity_price = pricing_params[1]
                        
                        # Strong penalty for pricing that causes vehicle rejections
                        if rejected_vehicles > 0:
                            rejection_penalty = rejected_vehicles * 0.5  # Strong penalty per rejected vehicle
                            reward -= rejection_penalty
                        
                        # Penalty for extreme pricing
                        if base_price > 1.2 or base_price < 0.3:
                            reward -= 0.3
                        elif 0.4 <= base_price <= 1.0:
                            reward += 0.3  # Reward for reasonable pricing
                        
                        if capacity_price > 0.3 or capacity_price < 0.01:
                            reward -= 0.2
                        elif 0.02 <= capacity_price <= 0.2:
                            reward += 0.2  # Reward for reasonable capacity pricing
                
                # 3. Grid efficiency reward (varies based on usage)
                if current_grid_usage <= current_peak_threshold:
                    # Reward for efficient grid usage (closer to threshold = better)
                    efficiency_ratio = current_grid_usage / current_peak_threshold
                    if 0.7 <= efficiency_ratio <= 0.9:
                        reward += 0.3  # Sweet spot for efficiency
                    elif efficiency_ratio < 0.3:
                        reward -= 0.1  # Too low usage
                    elif efficiency_ratio > 0.95:
                        reward -= 0.2  # Too close to limit
                
                # 4. Vehicle service reward (varies based on demand)
                if active_vehicles > 0:
                    # Reward for serving vehicles, but with diminishing returns
                    service_reward = min(active_vehicles * 0.3, 2.0)  # Cap at 2.0
                    reward += service_reward
                
                # 5. Completion reward (small incremental reward)
                completion_reward = completed_vehicles * 0.1  # Smaller reward per completion
                reward += completion_reward
                
                # 6. Time-based reward variation (encourage progress)
                time_factor = (self.sim_env.now % 1440) / 1440  # Normalize to 0-1 over day
                time_reward = 0.1 * time_factor  # Small time-based reward
                reward += time_reward
                
                return reward
                
            except Exception as e:
                # If reward computation fails, return neutral reward
                print(f"Simulation-based reward computation failed: {e}")
                return 0.0
        else:
            # Fallback: return 0 reward
            return 0.0


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
