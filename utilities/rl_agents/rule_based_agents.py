from typing import Any, Dict, List, Optional
import numpy as np
from datetime import datetime

from utilities.rl_agents.interfaces import (
    BaseAgent, 
    DecisionType, 
    AgentType,
    PricingAgent,
    ChargingAgent,
    StorageAgent,
    RoutingAgent,
    VehicleAssignmentAgent,
    ParkingAllocationAgent,
    GridManagementAgent,
    DemandForecastingAgent
)


class RuleBasedPricingAgent(PricingAgent):
    """
    Rule-based pricing agent that implements simple pricing strategies.
    
    This agent demonstrates how rule-based agents can be used alongside RL agents.
    It implements common pricing strategies like time-of-use, demand-based, and
    cost-plus pricing.
    """
    
    def __init__(self, strategy: str = "time_of_use"):
        self.strategy = strategy
        self.state = None
        self._agent_type = AgentType.RULE_BASED
        self._decision_type = DecisionType.PRICING
        
    @property
    def agent_type(self) -> AgentType:
        return self._agent_type
    
    @property
    def decision_type(self) -> DecisionType:
        return self._decision_type
    
    def reset(self) -> None:
        """Reset the agent state."""
        self.state = None
    
    def update_state(self, context: Dict[str, Any]) -> None:
        """Update agent state based on context."""
        self.state = context
    
    def get_state(self) -> Any:
        """Get current agent state."""
        return self.state
    
    def set_state(self, state: Any) -> None:
        """Set agent state."""
        self.state = state
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select pricing action based on rule-based strategy.
        
        Args:
            context: Dictionary containing pricing context
            
        Returns:
            Dictionary containing pricing decision
        """
        if self.strategy == "time_of_use":
            return self._time_of_use_pricing(context)
        elif self.strategy == "demand_based":
            return self._demand_based_pricing(context)
        elif self.strategy == "cost_plus":
            return self._cost_plus_pricing(context)
        else:
            return self._default_pricing(context)
    
    def _time_of_use_pricing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Time-of-use pricing strategy."""
        env = context.get("env")
        current_hour = int((env.now % 1440) / 60) if env else 12
        
        # Peak hours: 8-10 AM and 6-8 PM
        if current_hour in [8, 9, 18, 19]:
            energy_price = 0.25  # High price during peak
        elif current_hour in [10, 11, 12, 13, 14, 15, 16, 17]:
            energy_price = 0.15  # Medium price during day
        else:
            energy_price = 0.10  # Low price during off-peak
        
        parking_fee = 2.0  # Fixed parking fee
        
        return {
            "pricing_parameters": [energy_price, parking_fee],
            "energy_price": energy_price,
            "parking_fee": parking_fee,
            "confidence": 0.9,
            "strategy": "time_of_use",
            "reasoning": f"Peak hour pricing applied for hour {current_hour}"
        }
    
    def _demand_based_pricing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Demand-based pricing strategy."""
        charging_hub = context.get("charging_hub")
        current_demand = charging_hub.grid.current_load if charging_hub else 100
        max_capacity = charging_hub.grid.capacity if charging_hub else 500
        
        # Calculate demand ratio
        demand_ratio = current_demand / max_capacity if max_capacity > 0 else 0.2
        
        # Base price with demand multiplier
        base_price = 0.15
        if demand_ratio > 0.8:
            energy_price = base_price * 1.5  # High demand
        elif demand_ratio > 0.6:
            energy_price = base_price * 1.2  # Medium demand
        else:
            energy_price = base_price  # Low demand
        
        parking_fee = 2.0
        
        return {
            "pricing_parameters": [energy_price, parking_fee],
            "energy_price": energy_price,
            "parking_fee": parking_fee,
            "confidence": 0.85,
            "strategy": "demand_based",
            "reasoning": f"Demand ratio {demand_ratio:.2f} applied"
        }
    
    def _cost_plus_pricing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Cost-plus pricing strategy."""
        # Assume base electricity cost
        base_cost = 0.12
        markup = 0.25  # 25% markup
        
        energy_price = base_cost * (1 + markup)
        parking_fee = 2.0
        
        return {
            "pricing_parameters": [energy_price, parking_fee],
            "energy_price": energy_price,
            "parking_fee": parking_fee,
            "confidence": 0.95,
            "strategy": "cost_plus",
            "reasoning": f"Cost-plus pricing with {markup*100}% markup"
        }
    
    def _default_pricing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default pricing strategy."""
        energy_price = 0.15
        parking_fee = 2.0
        
        return {
            "pricing_parameters": [energy_price, parking_fee],
            "energy_price": energy_price,
            "parking_fee": parking_fee,
            "confidence": 0.8,
            "strategy": "default",
            "reasoning": "Default pricing applied"
        }
    
    def learn(self, transition: Optional[Dict[str, Any]] = None) -> None:
        """Rule-based agents don't learn from transitions."""
        pass


class RuleBasedChargingAgent(ChargingAgent):
    """
    Rule-based charging agent that implements simple charging strategies.
    
    This agent implements strategies like first-come-first-served, priority-based,
    and load-balancing charging.
    """
    
    def __init__(self, strategy: str = "first_come_first_served"):
        self.strategy = strategy
        self.state = None
        self._agent_type = AgentType.RULE_BASED
        self._decision_type = DecisionType.CHARGING
        
    @property
    def agent_type(self) -> AgentType:
        return self._agent_type
    
    @property
    def decision_type(self) -> DecisionType:
        return self._decision_type
    
    def reset(self) -> None:
        self.state = None
    
    def update_state(self, context: Dict[str, Any]) -> None:
        self.state = context
    
    def get_state(self) -> Any:
        return self.state
    
    def set_state(self, state: Any) -> None:
        self.state = state
    
    def select_action(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select charging action based on rule-based strategy.
        
        Args:
            vehicles: List of vehicles requiring charging
            context: Dictionary containing charging context
            
        Returns:
            Dictionary containing charging decision
        """
        if self.strategy == "first_come_first_served":
            return self._first_come_first_served(vehicles, context)
        elif self.strategy == "priority_based":
            return self._priority_based(vehicles, context)
        elif self.strategy == "load_balancing":
            return self._load_balancing(vehicles, context)
        else:
            return self._default_charging(vehicles, context)
    
    def _first_come_first_served(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """First-come-first-served charging strategy."""
        charging_actions = []
        priority_order = []
        
        # Sort vehicles by arrival time
        sorted_vehicles = sorted(vehicles, key=lambda v: v.arrival_period)
        
        for i, vehicle in enumerate(sorted_vehicles):
            # Assign equal power to all vehicles
            charging_power = 22.0  # Default charging power
            charging_actions.append(charging_power)
            priority_order.append(i)
        
        return {
            "charging_actions": charging_actions,
            "power_allocation": "equal",
            "priority_order": priority_order,
            "confidence": 0.9,
            "strategy": "first_come_first_served",
            "reasoning": f"FCFS strategy applied to {len(vehicles)} vehicles"
        }
    
    def _priority_based(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Priority-based charging strategy."""
        charging_actions = []
        priority_order = []
        
        # Sort vehicles by priority (energy deficit, departure time, etc.)
        def priority_key(vehicle):
            energy_deficit = vehicle.remaining_energy_deficit
            time_until_departure = vehicle.departure_period - context.get("env", {}).now
            return (energy_deficit, -time_until_departure)  # Higher deficit and earlier departure = higher priority
        
        sorted_vehicles = sorted(vehicles, key=priority_key, reverse=True)
        
        for i, vehicle in enumerate(sorted_vehicles):
            # Higher priority vehicles get more power
            if i < len(vehicles) // 3:
                charging_power = 50.0  # High priority
            elif i < 2 * len(vehicles) // 3:
                charging_power = 22.0  # Medium priority
            else:
                charging_power = 11.0  # Low priority
            
            charging_actions.append(charging_power)
            priority_order.append(i)
        
        return {
            "charging_actions": charging_actions,
            "power_allocation": "priority_based",
            "priority_order": priority_order,
            "confidence": 0.85,
            "strategy": "priority_based",
            "reasoning": f"Priority-based strategy applied to {len(vehicles)} vehicles"
        }
    
    def _load_balancing(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Load balancing charging strategy."""
        charging_actions = []
        priority_order = []
        
        # Calculate total available power
        charging_hub = context.get("charging_hub")
        available_power = charging_hub.grid.capacity if charging_hub else 500
        
        # Distribute power evenly among vehicles
        power_per_vehicle = available_power / len(vehicles) if vehicles else 0
        
        for i, vehicle in enumerate(vehicles):
            charging_actions.append(power_per_vehicle)
            priority_order.append(i)
        
        return {
            "charging_actions": charging_actions,
            "power_allocation": "load_balanced",
            "priority_order": priority_order,
            "confidence": 0.8,
            "strategy": "load_balancing",
            "reasoning": f"Load balancing with {power_per_vehicle:.1f} kW per vehicle"
        }
    
    def _default_charging(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Default charging strategy."""
        charging_actions = [22.0] * len(vehicles)  # Default power for all vehicles
        priority_order = list(range(len(vehicles)))
        
        return {
            "charging_actions": charging_actions,
            "power_allocation": "default",
            "priority_order": priority_order,
            "confidence": 0.7,
            "strategy": "default",
            "reasoning": f"Default charging strategy applied to {len(vehicles)} vehicles"
        }
    
    def learn(self, transition: Optional[Dict[str, Any]] = None) -> None:
        """Rule-based agents don't learn from transitions."""
        pass


class RuleBasedStorageAgent(StorageAgent):
    """
    Rule-based storage agent that implements simple storage strategies.
    
    This agent implements strategies like peak shaving, arbitrage, and
    grid support operations.
    """
    
    def __init__(self, strategy: str = "peak_shaving"):
        self.strategy = strategy
        self.state = None
        self._agent_type = AgentType.RULE_BASED
        self._decision_type = DecisionType.STORAGE
        
    @property
    def agent_type(self) -> AgentType:
        return self._agent_type
    
    @property
    def decision_type(self) -> DecisionType:
        return self._decision_type
    
    def reset(self) -> None:
        self.state = None
    
    def update_state(self, context: Dict[str, Any]) -> None:
        self.state = context
    
    def get_state(self) -> Any:
        return self.state
    
    def set_state(self, state: Any) -> None:
        self.state = state
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select storage action based on rule-based strategy.
        
        Args:
            context: Dictionary containing storage context
            
        Returns:
            Dictionary containing storage decision
        """
        if self.strategy == "peak_shaving":
            return self._peak_shaving(context)
        elif self.strategy == "arbitrage":
            return self._arbitrage(context)
        elif self.strategy == "grid_support":
            return self._grid_support(context)
        else:
            return self._default_storage(context)
    
    def _peak_shaving(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Peak shaving storage strategy."""
        charging_hub = context.get("charging_hub")
        current_load = charging_hub.grid.current_load if charging_hub else 100
        max_capacity = charging_hub.grid.capacity if charging_hub else 500
        storage_soc = charging_hub.electric_storage.soc if charging_hub else 0.5
        
        # Discharge if load is high and storage has capacity
        if current_load > max_capacity * 0.8 and storage_soc > 0.2:
            storage_action = -50.0  # Discharge
            strategy = "peak_shaving_discharge"
        elif current_load < max_capacity * 0.4 and storage_soc < 0.8:
            storage_action = 30.0   # Charge
            strategy = "peak_shaving_charge"
        else:
            storage_action = 0.0    # No action
            strategy = "peak_shaving_idle"
        
        return {
            "storage_action": storage_action,
            "power_level": abs(storage_action),
            "strategy": strategy,
            "confidence": 0.85,
            "reasoning": f"Peak shaving: load={current_load:.1f}, soc={storage_soc:.2f}"
        }
    
    def _arbitrage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Arbitrage storage strategy."""
        env = context.get("env")
        current_hour = int((env.now % 1440) / 60) if env else 12
        storage_soc = context.get("storage_soc", 0.5)
        
        # Charge during low-price hours (night), discharge during high-price hours (day)
        if 22 <= current_hour or current_hour <= 6:  # Night hours
            if storage_soc < 0.9:
                storage_action = 40.0  # Charge
                strategy = "arbitrage_charge"
            else:
                storage_action = 0.0   # Full
                strategy = "arbitrage_full"
        else:  # Day hours
            if storage_soc > 0.1:
                storage_action = -40.0  # Discharge
                strategy = "arbitrage_discharge"
            else:
                storage_action = 0.0    # Empty
                strategy = "arbitrage_empty"
        
        return {
            "storage_action": storage_action,
            "power_level": abs(storage_action),
            "strategy": strategy,
            "confidence": 0.8,
            "reasoning": f"Arbitrage: hour={current_hour}, soc={storage_soc:.2f}"
        }
    
    def _grid_support(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Grid support storage strategy."""
        charging_hub = context.get("charging_hub")
        grid_frequency = getattr(charging_hub.grid, 'frequency', 50.0) if charging_hub else 50.0
        storage_soc = charging_hub.electric_storage.soc if charging_hub else 0.5
        
        # Support grid frequency
        if grid_frequency < 49.8:  # Low frequency
            if storage_soc > 0.1:
                storage_action = -30.0  # Discharge to support
                strategy = "grid_support_discharge"
            else:
                storage_action = 0.0
                strategy = "grid_support_empty"
        elif grid_frequency > 50.2:  # High frequency
            if storage_soc < 0.9:
                storage_action = 30.0   # Charge to absorb
                strategy = "grid_support_charge"
            else:
                storage_action = 0.0
                strategy = "grid_support_full"
        else:  # Normal frequency
            storage_action = 0.0
            strategy = "grid_support_idle"
        
        return {
            "storage_action": storage_action,
            "power_level": abs(storage_action),
            "strategy": strategy,
            "confidence": 0.9,
            "reasoning": f"Grid support: frequency={grid_frequency:.1f}, soc={storage_soc:.2f}"
        }
    
    def _default_storage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default storage strategy."""
        storage_action = 0.0
        
        return {
            "storage_action": storage_action,
            "power_level": 0.0,
            "strategy": "default",
            "confidence": 0.7,
            "reasoning": "Default storage strategy applied"
        }
    
    def learn(self, transition: Optional[Dict[str, Any]] = None) -> None:
        """Rule-based agents don't learn from transitions."""
        pass
