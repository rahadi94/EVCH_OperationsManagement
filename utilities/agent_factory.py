"""
Agent Factory Module

This module handles the creation and configuration of different types of agents
for the EVCC simulation framework.
"""

from typing import Optional
from resources.configuration.configuration import Configuration
from utilities.rl_agents.interfaces import DecisionType, AgentType
from utilities.rl_agents.rule_based_agents import (
    RuleBasedPricingAgent, RuleBasedChargingAgent, RuleBasedStorageAgent
)
from utilities.rl_agents.algorithm_agents import (
    AlgorithmChargingAgent, AlgorithmRoutingAgent, AlgorithmStorageAgent
)
from utilities.rl_agents.agents.actor_critic_agents.SAC import SAC
from resources.configuration.SAC_configuration import pricing_config
from utilities.rl_environments.rl_pricing_env import PricingEnv


def is_agent_learnable(agent_type: str) -> bool:
    """
    Determine if an agent type is learnable (RL agent).
    
    Args:
        agent_type: String representation of agent type
        
    Returns:
        True if the agent is learnable (RL agent), False otherwise
    """
    learnable_types = ["RL_SAC", "RL_DQN", "RL_DDPG"]
    return agent_type.upper() in learnable_types


def create_agent(decision_type: str, agent_type: str, config: Configuration, 
                 algorithm: Optional[str] = None, strategy: Optional[str] = None):
    """
    Create an agent based on decision type and agent type.
    
    Args:
        decision_type: Type of decision (pricing, charging, storage, routing)
        agent_type: Type of agent (RL_SAC, RULE_BASED, HEURISTIC, etc.)
        config: Configuration instance
        algorithm: Algorithm name for heuristic agents
        strategy: Strategy name for agents that support different strategies
        
    Returns:
        Agent instance
        
    Raises:
        ValueError: If decision type or agent type is not supported
    """
    if decision_type == "pricing":
        return _create_pricing_agent(agent_type, config, strategy)
    elif decision_type == "charging":
        return _create_charging_agent(agent_type, config, algorithm, strategy)
    elif decision_type == "storage":
        return _create_storage_agent(agent_type, config, algorithm, strategy)
    elif decision_type == "routing":
        return _create_routing_agent(agent_type, config, algorithm, strategy)
    else:
        raise ValueError(f"Unsupported decision type: {decision_type}")


def _create_pricing_agent(agent_type: str, config: Configuration, strategy: Optional[str] = None):
    """Create a pricing agent."""
    if agent_type == "RL_SAC":
        return _create_sac_pricing_agent(config)
    elif agent_type == "HEURISTIC":
        strategy = strategy or "time_of_use"
        return RuleBasedPricingAgent(strategy=strategy)
    elif agent_type == "RULE_BASED":
        return RuleBasedPricingAgent(strategy=strategy or "time_of_use")
    else:
        raise ValueError(f"Unsupported agent type for pricing: {agent_type}")


def _create_charging_agent(agent_type: str, config: Configuration, algorithm: Optional[str] = None, 
                          strategy: Optional[str] = None):
    """Create a charging agent."""
    if agent_type == "HEURISTIC":
        return AlgorithmChargingAgent(algorithm=algorithm or "first_come_first_served")
    elif agent_type == "RULE_BASED":
        return RuleBasedChargingAgent(strategy=strategy or "first_come_first_served")
    else:
        raise ValueError(f"Unsupported agent type for charging: {agent_type}")


def _create_storage_agent(agent_type: str, config: Configuration, algorithm: Optional[str] = None, 
                         strategy: Optional[str] = None):
    """Create a storage agent."""
    if agent_type == "HEURISTIC":
        return AlgorithmStorageAgent(algorithm=algorithm or "peak_shaving")
    elif agent_type == "RULE_BASED":
        return RuleBasedStorageAgent(strategy=strategy or "peak_shaving")
    else:
        raise ValueError(f"Unsupported agent type for storage: {agent_type}")


def _create_routing_agent(agent_type: str, config: Configuration, algorithm: Optional[str] = None, 
                         strategy: Optional[str] = None):
    """Create a routing agent."""
    if agent_type == "HEURISTIC":
        return AlgorithmRoutingAgent(algorithm=algorithm or "lowest_occupancy_first")
    elif agent_type == "RULE_BASED":
        return AlgorithmRoutingAgent(algorithm=strategy or "lowest_occupancy_first")
    else:
        raise ValueError(f"Unsupported agent type for routing: {agent_type}")


def _create_sac_pricing_agent(config: Configuration):
    """Create and configure a SAC pricing agent."""
    # Configure pricing environment for RL agent
    pricing_config.number_chargers = config.facility_size
    pricing_config.maximum_power = 50
    pricing_config.maximum_grid_usage = 2000
    pricing_config.number_power_options = len(config.energy_prices)
    pricing_config.environment = PricingEnv(config=pricing_config, DQN=False)
    pricing_config.learnt_network = config.evaluation_after_training
    pricing_config.evaluation_after_training = config.evaluation_after_training
    
    return SAC(pricing_config)


def get_agent_configuration(config: Configuration) -> dict:
    """
    Get agent configuration from the main configuration.
    
    Args:
        config: Configuration instance
        
    Returns:
        Dictionary containing agent configurations
    """
    return {
        "pricing": {
            "agent_type": getattr(config, 'default_agent_types', {}).get("pricing"),
            "strategy": getattr(config, 'default_strategies', {}).get("pricing", "time_of_use")
        },
        "charging": {
            "agent_type": getattr(config, 'default_agent_types', {}).get("charging"),
            "algorithm": getattr(config, 'default_algorithms', {}).get("charging", "first_come_first_served"),
            "strategy": getattr(config, 'default_strategies', {}).get("charging", "first_come_first_served")
        },
        "storage": {
            "agent_type": getattr(config, 'default_agent_types', {}).get("storage"),
            "algorithm": getattr(config, 'default_algorithms', {}).get("storage", "peak_shaving"),
            "strategy": getattr(config, 'default_strategies', {}).get("storage", "peak_shaving")
        },
        "routing": {
            "agent_type": getattr(config, 'default_agent_types', {}).get("routing"),
            "algorithm": getattr(config, 'default_algorithms', {}).get("routing", "lowest_occupancy_first"),
            "strategy": getattr(config, 'default_strategies', {}).get("routing", "lowest_occupancy_first")
        }
    }
