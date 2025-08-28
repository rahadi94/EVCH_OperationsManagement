from typing import Any, Dict

from utilities.rl_agents.interfaces import PricingAgent as PricingAgentInterface
from utilities.rl_environments.rl_pricing_env import PricingEnv


class PricingEnvAgentAdapter(PricingAgentInterface):
    """
    Adapter that wires an underlying RL policy to the PricingEnv API.
    The underlying policy must implement: reset(), act(obs) -> action, learn(transition)
    """

    def __init__(self, env: PricingEnv, policy: Any):
        self.env = env
        self.policy = policy
        self._last_obs = None

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self._last_obs = self.env.reset()

    def update_state(self, context: Dict[str, Any]) -> None:
        charging_hub = context.get("charging_hub")
        sim_env = context.get("env")
        self._last_obs = self.env.get_state(charging_hub, sim_env)

    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if self._last_obs is None:
            self.update_state(context)
        if hasattr(self.policy, "act"):
            action = self.policy.act(self._last_obs)
        else:
            # Fallback: assume policy is callable
            action = self.policy(self._last_obs)
        return {"action": action}

    def learn(self, transition: Dict[str, Any]) -> None:
        if hasattr(self.policy, "learn"):
            self.policy.learn(transition)


