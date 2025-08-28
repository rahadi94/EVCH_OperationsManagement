from typing import Any, Optional
from simulation.operations.agents_controller import AgentsController
from utilities.rl_agents.adapters.pricing_adapter import PricingEnvAgentAdapter
from utilities.rl_agents.adapters.charging_adapter import ChargingEnvAgentAdapter
from utilities.rl_agents.adapters.storage_adapter import StorageEnvAgentAdapter
from utilities.rl_agents.adapters.gym_agent_adapter import (
    GymPricingAgentAdapter, GymChargingAgentAdapter, GymStorageAgentAdapter,
    create_gym_pricing_adapter, create_gym_charging_adapter, create_gym_storage_adapter
)
from utilities.rl_environments.rl_pricing_env import PricingEnv


def build_pricing_adapter(config: Any, policy: Any, dqn: bool = False) -> PricingEnvAgentAdapter:
    """
    Build a PricingEnv and wrap the given policy with the adapter.

    Args:
        config: pricing env configuration object (must have number_power_options, maximum_power, evaluation)
        policy: underlying RL policy with act(obs) and optional reset()/learn()
        dqn: whether to initialize env in DQN discrete mode

    Returns:
        PricingEnvAgentAdapter instance
    """
    env = PricingEnv(config, DQN=dqn)
    return PricingEnvAgentAdapter(env=env, policy=policy)


def build_charging_adapter(config: Any, policy: Any) -> ChargingEnvAgentAdapter:
    """
    Build a charging environment and wrap the given policy with the adapter.

    Args:
        config: charging env configuration object
        policy: underlying RL policy with act(obs) and optional reset()/learn()

    Returns:
        ChargingEnvAgentAdapter instance
    """
    from utilities.rl_environments.SC_env import ChargingHubInvestmentEnv
    env = ChargingHubInvestmentEnv(config)
    return ChargingEnvAgentAdapter(rl_agent=policy, charging_env=env)


def build_storage_adapter(config: Any, policy: Any) -> StorageEnvAgentAdapter:
    """
    Build a storage environment and wrap the given policy with the adapter.

    Args:
        config: storage env configuration object
        policy: underlying RL policy with act(obs) and optional reset()/learn()

    Returns:
        StorageEnvAgentAdapter instance
    """
    from utilities.rl_environments.SC_storage_env import StorageEnv
    env = StorageEnv(config)
    return StorageEnvAgentAdapter(rl_agent=policy, storage_env=env)


def build_agents_controller(
    pricing_agent: Optional[Any] = None,
    charging_agent: Optional[Any] = None,
    storage_agent: Optional[Any] = None,
    pricing_env: Optional[PricingEnv] = None,
    pricing_config: Optional[Any] = None,
    charging_config: Optional[Any] = None,
    storage_config: Optional[Any] = None
) -> AgentsController:
    """
    Build an AgentsController with the specified agents.

    Args:
        pricing_agent: RL agent for pricing (e.g., SAC, DQN)
        charging_agent: RL agent for charging (e.g., SAC, DQN)
        storage_agent: RL agent for storage (e.g., SAC, DQN)
        pricing_env: Pricing environment (optional, will be created if not provided)
        pricing_config: Configuration for pricing environment
        charging_config: Configuration for charging environment
        storage_config: Configuration for storage environment

    Returns:
        AgentsController instance
    """
    pricing_adapter = None
    charging_adapter = None
    storage_adapter = None
    
    if pricing_agent and pricing_config:
        if not pricing_env:
            pricing_env = PricingEnv(pricing_config)
        pricing_adapter = PricingEnvAgentAdapter(rl_agent=pricing_agent, pricing_env=pricing_env)
    
    if charging_agent and charging_config:
        charging_adapter = build_charging_adapter(charging_config, charging_agent)
    
    if storage_agent and storage_config:
        storage_adapter = build_storage_adapter(storage_config, storage_agent)
    
    return AgentsController(
        pricing=pricing_adapter,
        charging=charging_adapter,
        storage=storage_adapter
    )


# Gym-compatible agent factory functions
def build_gym_pricing_adapter(config_dict: Dict[str, Any], gym_agent: Any, **kwargs) -> GymPricingAgentAdapter:
    """
    Build a gym-compatible pricing agent adapter.
    
    Args:
        config_dict: Configuration dictionary for the environment
        gym_agent: Gym-compatible RL agent (e.g., Stable Baselines3 agent)
        **kwargs: Additional arguments for environment creation
        
    Returns:
        GymPricingAgentAdapter instance
    """
    return create_gym_pricing_adapter(config_dict, gym_agent, **kwargs)


def build_gym_charging_adapter(config_dict: Dict[str, Any], gym_agent: Any, **kwargs) -> GymChargingAgentAdapter:
    """
    Build a gym-compatible charging agent adapter.
    
    Args:
        config_dict: Configuration dictionary for the environment
        gym_agent: Gym-compatible RL agent (e.g., Stable Baselines3 agent)
        **kwargs: Additional arguments for environment creation
        
    Returns:
        GymChargingAgentAdapter instance
    """
    return create_gym_charging_adapter(config_dict, gym_agent, **kwargs)


def build_gym_storage_adapter(config_dict: Dict[str, Any], gym_agent: Any, **kwargs) -> GymStorageAgentAdapter:
    """
    Build a gym-compatible storage agent adapter.
    
    Args:
        config_dict: Configuration dictionary for the environment
        gym_agent: Gym-compatible RL agent (e.g., Stable Baselines3 agent)
        **kwargs: Additional arguments for environment creation
        
    Returns:
        GymStorageAgentAdapter instance
    """
    return create_gym_storage_adapter(config_dict, gym_agent, **kwargs)


def build_gym_agents_controller(
    pricing_agent: Optional[Any] = None,
    charging_agent: Optional[Any] = None,
    storage_agent: Optional[Any] = None,
    pricing_config: Optional[Dict[str, Any]] = None,
    charging_config: Optional[Dict[str, Any]] = None,
    storage_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AgentsController:
    """
    Build an AgentsController with gym-compatible agents.
    
    Args:
        pricing_agent: Gym-compatible RL agent for pricing
        charging_agent: Gym-compatible RL agent for charging
        storage_agent: Gym-compatible RL agent for storage
        pricing_config: Configuration for pricing environment
        charging_config: Configuration for charging environment
        storage_config: Configuration for storage environment
        **kwargs: Additional arguments for environment creation
        
    Returns:
        AgentsController instance with gym-compatible agents
    """
    pricing_adapter = None
    charging_adapter = None
    storage_adapter = None
    
    if pricing_agent and pricing_config:
        pricing_adapter = build_gym_pricing_adapter(pricing_config, pricing_agent, **kwargs)
    
    if charging_agent and charging_config:
        charging_adapter = build_gym_charging_adapter(charging_config, charging_agent, **kwargs)
    
    if storage_agent and storage_config:
        storage_adapter = build_gym_storage_adapter(storage_config, storage_agent, **kwargs)
    
    return AgentsController(
        pricing=pricing_adapter,
        charging=charging_adapter,
        storage=storage_adapter
    )


