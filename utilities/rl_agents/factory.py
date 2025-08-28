from typing import Any, Optional

from utilities.rl_agents.adapters.pricing_adapter import PricingEnvAgentAdapter
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


