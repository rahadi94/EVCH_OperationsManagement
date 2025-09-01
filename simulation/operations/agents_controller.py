from typing import Optional, Dict, Any, List

from utilities.rl_agents.interfaces import PricingAgent, ChargingAgent, StorageAgent


class AgentsController:
    def __init__(self,
                 pricing: Optional[PricingAgent] = None,
                 charging: Optional[ChargingAgent] = None,
                 storage: Optional[StorageAgent] = None):
        self.pricing = pricing
        self.charging = charging
        self.storage = storage

    def reset_all(self) -> None:
        for agent in (self.pricing, self.charging, self.storage):
            if agent:
                agent.reset()

    def pricing_step(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.pricing:
            return None
        self.pricing.update_state(context)
        return self.pricing.select_action(context)

    def charging_step(self, vehicles: List[Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.charging:
            return None
        self.charging.update_state(context)
        return self.charging.select_action(vehicles, context)

    def storage_step(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.storage:
            return None
        self.storage.update_state(context)
        return self.storage.select_action(context)

    def learn_all(self, transition: Dict[str, Any]) -> None:
        for agent in (self.pricing, self.charging, self.storage):
            if agent:
                agent.learn(transition)


