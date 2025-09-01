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
    VehicleAssignmentAgent
)

# Import existing algorithms
from simulation.operations.ChargingAlgorithms import (
    uncontrolled as charging_uncontrolled,
    first_come_first_served,
    earliest_deadline_first,
    least_laxity_first,
    equal_sharing,
    online_myopic,
    online_multi_period,
    integrated_charging_storage
)

from simulation.operations.RoutingAlgorithms import (
    random_charger_assignment,
    lowest_occupancy_first_charger_assignment,
    fill_one_after_other_charger_assignment,
    lowest_utilization_first_charger_assignment,
    matching_supply_demand_level,
    assign_to_the_minimum_power
)

from simulation.operations.StorageAlgorithms import (
    uncontrolled as storage_uncontrolled,
    temporal_arbitrage,
    peak_shaving
)

from simulation.operations.IntegratedAlgorithms import (
    perfect_info_charging_routing,
    perfect_info_charging_routing_storage
)


class AlgorithmChargingAgent(ChargingAgent):
    """
    Agent that wraps existing charging algorithms.
    
    This agent provides a standardized interface to all the existing
    charging algorithms in the codebase.
    """
    
    def __init__(self, algorithm: str = "first_come_first_served"):
        self.algorithm = algorithm
        self.state = None
        self._agent_type = AgentType.HEURISTIC
        self._decision_type = DecisionType.CHARGING
        
        # Algorithm mapping
        self.algorithm_functions = {
            "uncontrolled": self._uncontrolled_charging,
            "first_come_first_served": self._first_come_first_served,
            "earliest_deadline_first": self._earliest_deadline_first,
            "least_laxity_first": self._least_laxity_first,
            "equal_sharing": self._equal_sharing,
            "online_myopic": self._online_myopic,
            "online_multi_period": self._online_multi_period,
            "integrated_storage": self._integrated_storage,
            "perfect_info": self._perfect_info,
            "perfect_info_with_storage": self._perfect_info_with_storage
        }
        
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
        Select charging action using the specified algorithm.
        
        Args:
            vehicles: List of vehicles requiring charging
            context: Dictionary containing charging context
            
        Returns:
            Dictionary containing charging decision
        """
        if self.algorithm not in self.algorithm_functions:
            raise ValueError(f"Unknown charging algorithm: {self.algorithm}")
        
        # Get algorithm function
        algo_func = self.algorithm_functions[self.algorithm]
        
        # Execute algorithm
        charging_actions = algo_func(vehicles, context)
        
        return {
            "charging_actions": charging_actions,
            "power_allocation": self.algorithm,
            "priority_order": list(range(len(vehicles))),
            "confidence": 0.9,
            "strategy": self.algorithm,
            "reasoning": f"Applied {self.algorithm} charging algorithm to {len(vehicles)} vehicles"
        }
    
    def _uncontrolled_charging(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Uncontrolled charging algorithm."""
        env = context.get("env")
        charging_stations = context.get("charging_stations", [])
        charging_capacity = context.get("charging_capacity", 500)
        free_grid_capacity = context.get("free_grid_capacity", 500)
        planning_period_length = context.get("planning_period_length", 15)
        
        # Execute algorithm
        charging_uncontrolled(
            env=env,
            connected_vehicles=vehicles,
            charging_stations=charging_stations,
            charging_capacity=charging_capacity,
            free_grid_capacity=free_grid_capacity,
            planning_period_length=planning_period_length
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _first_come_first_served(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """First come first served charging algorithm."""
        env = context.get("env")
        charging_stations = context.get("charging_stations", [])
        charging_capacity = context.get("charging_capacity", 500)
        free_grid_capacity = context.get("free_grid_capacity", 500)
        planning_period_length = context.get("planning_period_length", 15)
        
        # Execute algorithm
        first_come_first_served(
            env=env,
            connected_vehicles=vehicles,
            charging_stations=charging_stations,
            charging_capacity=charging_capacity,
            free_grid_capacity=free_grid_capacity,
            planning_period_length=planning_period_length
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _earliest_deadline_first(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Earliest deadline first charging algorithm."""
        env = context.get("env")
        charging_stations = context.get("charging_stations", [])
        charging_capacity = context.get("charging_capacity", 500)
        free_grid_capacity = context.get("free_grid_capacity", 500)
        planning_period_length = context.get("planning_period_length", 15)
        
        # Execute algorithm
        earliest_deadline_first(
            env=env,
            connected_vehicles=vehicles,
            charging_stations=charging_stations,
            charging_capacity=charging_capacity,
            free_grid_capacity=free_grid_capacity,
            planning_period_length=planning_period_length
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _least_laxity_first(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Least laxity first charging algorithm."""
        env = context.get("env")
        charging_stations = context.get("charging_stations", [])
        charging_capacity = context.get("charging_capacity", 500)
        free_grid_capacity = context.get("free_grid_capacity", 500)
        planning_period_length = context.get("planning_period_length", 15)
        
        # Execute algorithm
        least_laxity_first(
            env=env,
            connected_vehicles=vehicles,
            charging_stations=charging_stations,
            charging_capacity=charging_capacity,
            free_grid_capacity=free_grid_capacity,
            planning_period_length=planning_period_length
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _equal_sharing(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Equal sharing charging algorithm."""
        env = context.get("env")
        charging_stations = context.get("charging_stations", [])
        charging_capacity = context.get("charging_capacity", 500)
        free_grid_capacity = context.get("free_grid_capacity", 500)
        planning_period_length = context.get("planning_period_length", 15)
        
        # Execute algorithm
        equal_sharing(
            env=env,
            connected_vehicles=vehicles,
            charging_stations=charging_stations,
            charging_capacity=charging_capacity,
            free_grid_capacity=free_grid_capacity,
            planning_period_length=planning_period_length
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _online_myopic(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Online myopic charging algorithm."""
        env = context.get("env")
        charging_stations = context.get("charging_stations", [])
        electricity_cost = context.get("electricity_cost", [0.15] * 24)
        sim_time = context.get("sim_time", 1440)
        peak_load_history = context.get("peak_load_history", [])
        free_grid_capa_actual = context.get("free_grid_capa_actual", 500)
        free_grid_capa_predicted = context.get("free_grid_capa_predicted", 500)
        
        # Execute algorithm
        online_myopic(
            vehicles=vehicles,
            charging_stations=charging_stations,
            env=env,
            electricity_cost=electricity_cost,
            sim_time=sim_time,
            peak_load_history=peak_load_history,
            free_grid_capa_actual=free_grid_capa_actual,
            free_grid_capa_predicted=free_grid_capa_predicted
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _online_multi_period(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Online multi-period charging algorithm."""
        env = context.get("env")
        charging_stations = context.get("charging_stations", [])
        electricity_cost = context.get("electricity_cost", [0.15] * 24)
        sim_time = context.get("sim_time", 1440)
        peak_load_history = context.get("peak_load_history", [])
        free_grid_capa_actual = context.get("free_grid_capa_actual", 500)
        free_grid_capa_predicted = context.get("free_grid_capa_predicted", 500)
        
        # Execute algorithm
        online_multi_period(
            vehicles=vehicles,
            charging_stations=charging_stations,
            env=env,
            electricity_cost=electricity_cost,
            sim_time=sim_time,
            peak_load_history=peak_load_history,
            free_grid_capa_actual=free_grid_capa_actual,
            free_grid_capa_predicted=free_grid_capa_predicted
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _integrated_storage(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Integrated charging and storage algorithm."""
        storage = context.get("storage")
        charging_stations = context.get("charging_stations", [])
        env = context.get("env")
        electricity_cost = context.get("electricity_cost", [0.15] * 24)
        sim_time = context.get("sim_time", 1440)
        peak_load_history = context.get("peak_load_history", [])
        free_grid_capa_actual = context.get("free_grid_capa_actual", 500)
        free_grid_capa_predicted = context.get("free_grid_capa_predicted", 500)
        
        # Execute algorithm
        integrated_charging_storage(
            storage=storage,
            vehicles=vehicles,
            charging_stations=charging_stations,
            env=env,
            electricity_cost=electricity_cost,
            sim_time=sim_time,
            peak_load_history=peak_load_history,
            free_grid_capa_actual=free_grid_capa_actual,
            free_grid_capa_predicted=free_grid_capa_predicted
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _perfect_info(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Perfect information charging and routing algorithm."""
        charging_stations = context.get("charging_stations", [])
        env = context.get("env")
        grid_capacity = context.get("grid_capacity", 500)
        electricity_cost = context.get("electricity_cost", [0.15] * 24)
        sim_time = context.get("sim_time", 1440)
        baseload = context.get("baseload", 100)
        generation = context.get("generation")
        service_level = context.get("service_level", 1)
        time_range = context.get("time_range", 24)
        
        # Execute algorithm
        perfect_info_charging_routing(
            vehicles=vehicles,
            charging_stations=charging_stations,
            env=env,
            grid_capacity=grid_capacity,
            electricity_cost=electricity_cost,
            sim_time=sim_time,
            baseload=baseload,
            generation=generation,
            service_level=service_level,
            time_range=time_range
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def _perfect_info_with_storage(self, vehicles: List[Any], context: Dict[str, Any]) -> List[float]:
        """Perfect information charging, routing, and storage algorithm."""
        charging_stations = context.get("charging_stations", [])
        env = context.get("env")
        grid_capacity = context.get("grid_capacity", 500)
        electricity_cost = context.get("electricity_cost", [0.15] * 24)
        sim_time = context.get("sim_time", 1440)
        baseload = context.get("baseload", 100)
        storage = context.get("storage")
        service_level = context.get("service_level", 1)
        time_range = context.get("time_range", 24 * 5)
        
        # Execute algorithm
        perfect_info_charging_routing_storage(
            vehicles=vehicles,
            charging_stations=charging_stations,
            env=env,
            grid_capacity=grid_capacity,
            electricity_cost=electricity_cost,
            sim_time=sim_time,
            baseload=baseload,
            storage=storage,
            service_level=service_level,
            time_range=time_range
        )
        
        # Extract charging actions
        return [vehicle.charging_power for vehicle in vehicles]
    
    def learn(self, transition: Optional[Dict[str, Any]] = None) -> None:
        """Algorithm-based agents don't learn from transitions."""
        pass


class AlgorithmRoutingAgent(RoutingAgent):
    """
    Agent that wraps existing routing algorithms.
    
    This agent provides a standardized interface to all the existing
    routing algorithms in the codebase.
    """
    
    def __init__(self, algorithm: str = "lowest_occupancy_first"):
        self.algorithm = algorithm
        self.state = None
        self._agent_type = AgentType.HEURISTIC
        self._decision_type = DecisionType.ROUTING
        
        # Algorithm mapping
        self.algorithm_functions = {
            "random": self._random_routing,
            "lowest_occupancy_first": self._lowest_occupancy_first,
            "fill_one_after_other": self._fill_one_after_other,
            "lowest_utilization_first": self._lowest_utilization_first,
            "matching_supply_demand": self._matching_supply_demand,
            "minimum_power_requirement": self._minimum_power_requirement
        }
        
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
        Select routing action using the specified algorithm.
        
        Args:
            vehicles: List of vehicles requiring routing
            context: Dictionary containing routing context
            
        Returns:
            Dictionary containing routing decision
        """
        if self.algorithm not in self.algorithm_functions:
            raise ValueError(f"Unknown routing algorithm: {self.algorithm}")
        
        # Get algorithm function
        algo_func = self.algorithm_functions[self.algorithm]
        
        # Execute algorithm for each vehicle
        routing_assignments = []
        for vehicle in vehicles:
            assignment = algo_func(vehicle, context)
            routing_assignments.append(assignment)
        
        return {
            "routing_assignments": routing_assignments,
            "queue_order": list(range(len(vehicles))),
            "wait_times": [0] * len(vehicles),  # Placeholder
            "confidence": 0.9,
            "strategy": self.algorithm,
            "reasoning": f"Applied {self.algorithm} routing algorithm to {len(vehicles)} vehicles"
        }
    
    def _random_routing(self, vehicle: Any, context: Dict[str, Any]) -> Any:
        """Random routing algorithm."""
        charging_stations = context.get("charging_stations", [])
        number_of_connectors = context.get("number_of_connectors", 1)
        demand_threshold = context.get("demand_threshold", 1)
        duration_threshold = context.get("duration_threshold", 100 * 60)
        
        return random_charger_assignment(
            charging_stations=charging_stations,
            number_of_connectors=number_of_connectors,
            request=vehicle,
            demand_threshold=demand_threshold,
            duration_threshold=duration_threshold
        )
    
    def _lowest_occupancy_first(self, vehicle: Any, context: Dict[str, Any]) -> Any:
        """Lowest occupancy first routing algorithm."""
        charging_stations = context.get("charging_stations", [])
        number_of_connectors = context.get("number_of_connectors", 1)
        demand_threshold = context.get("demand_threshold", 1)
        duration_threshold = context.get("duration_threshold", 100 * 60)
        
        return lowest_occupancy_first_charger_assignment(
            charging_stations=charging_stations,
            number_of_connectors=number_of_connectors,
            request=vehicle,
            demand_threshold=demand_threshold,
            duration_threshold=duration_threshold
        )
    
    def _fill_one_after_other(self, vehicle: Any, context: Dict[str, Any]) -> Any:
        """Fill one after other routing algorithm."""
        charging_stations = context.get("charging_stations", [])
        number_of_connectors = context.get("number_of_connectors", 1)
        demand_threshold = context.get("demand_threshold", 1)
        duration_threshold = context.get("duration_threshold", 24 * 60)
        
        return fill_one_after_other_charger_assignment(
            charging_stations=charging_stations,
            number_of_connectors=number_of_connectors,
            request=vehicle,
            demand_threshold=demand_threshold,
            duration_threshold=duration_threshold
        )
    
    def _lowest_utilization_first(self, vehicle: Any, context: Dict[str, Any]) -> Any:
        """Lowest utilization first routing algorithm."""
        charging_stations = context.get("charging_stations", [])
        number_of_connectors = context.get("number_of_connectors", 1)
        demand_threshold = context.get("demand_threshold", 1)
        duration_threshold = context.get("duration_threshold", 100 * 60)
        
        return lowest_utilization_first_charger_assignment(
            charging_stations=charging_stations,
            number_of_connectors=number_of_connectors,
            request=vehicle,
            demand_threshold=demand_threshold,
            duration_threshold=duration_threshold
        )
    
    def _matching_supply_demand(self, vehicle: Any, context: Dict[str, Any]) -> Any:
        """Matching supply demand routing algorithm."""
        charging_stations = context.get("charging_stations", [])
        number_of_connectors = context.get("number_of_connectors", 1)
        demand_threshold = context.get("demand_threshold", 1)
        duration_threshold = context.get("duration_threshold", 100 * 60)
        
        return matching_supply_demand_level(
            charging_stations=charging_stations,
            number_of_connectors=number_of_connectors,
            request=vehicle,
            demand_threshold=demand_threshold,
            duration_threshold=duration_threshold
        )
    
    def _minimum_power_requirement(self, vehicle: Any, context: Dict[str, Any]) -> Any:
        """Minimum power requirement routing algorithm."""
        charging_stations = context.get("charging_stations", [])
        number_of_connectors = context.get("number_of_connectors", 1)
        demand_threshold = context.get("demand_threshold", 1)
        duration_threshold = context.get("duration_threshold", 100 * 60)
        
        return assign_to_the_minimum_power(
            charging_stations=charging_stations,
            number_of_connectors=number_of_connectors,
            request=vehicle,
            demand_threshold=demand_threshold,
            duration_threshold=duration_threshold
        )
    
    def learn(self, transition: Optional[Dict[str, Any]] = None) -> None:
        """Algorithm-based agents don't learn from transitions."""
        pass


class AlgorithmStorageAgent(StorageAgent):
    """
    Agent that wraps existing storage algorithms.
    
    This agent provides a standardized interface to all the existing
    storage algorithms in the codebase.
    """
    
    def __init__(self, algorithm: str = "peak_shaving"):
        self.algorithm = algorithm
        self.state = None
        self._agent_type = AgentType.HEURISTIC
        self._decision_type = DecisionType.STORAGE
        
        # Algorithm mapping
        self.algorithm_functions = {
            "uncontrolled": self._uncontrolled_storage,
            "temporal_arbitrage": self._temporal_arbitrage,
            "peak_shaving": self._peak_shaving
        }
        
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
        Select storage action using the specified algorithm.
        
        Args:
            context: Dictionary containing storage context
            
        Returns:
            Dictionary containing storage decision
        """
        if self.algorithm not in self.algorithm_functions:
            raise ValueError(f"Unknown storage algorithm: {self.algorithm}")
        
        # Get algorithm function
        algo_func = self.algorithm_functions[self.algorithm]
        
        # Execute algorithm
        storage_action = algo_func(context)
        
        return {
            "storage_action": storage_action,
            "power_level": abs(storage_action),
            "strategy": self.algorithm,
            "confidence": 0.9,
            "reasoning": f"Applied {self.algorithm} storage algorithm"
        }
    
    def _uncontrolled_storage(self, context: Dict[str, Any]) -> float:
        """Uncontrolled storage algorithm."""
        env = context.get("env")
        storage_object = context.get("storage_object")
        
        # Execute algorithm
        storage_uncontrolled(env=env, storage_object=storage_object)
        
        # Extract storage action
        if storage_object.charge_yn == 1:
            return storage_object.charging_power
        elif storage_object.discharge_yn == 1:
            return -storage_object.discharging_power
        else:
            return 0.0
    
    def _temporal_arbitrage(self, context: Dict[str, Any]) -> float:
        """Temporal arbitrage storage algorithm."""
        env = context.get("env")
        storage_object = context.get("storage_object")
        planning_interval = context.get("planning_interval", 15)
        electricity_tariff = context.get("electricity_tariff", [0.15] * 24)
        free_grid_capacity = context.get("free_grid_capacity", 500)
        ev_charging_load = context.get("ev_charging_load", 100)
        
        # Execute algorithm
        temporal_arbitrage(
            env=env,
            storage_object=storage_object,
            planning_interval=planning_interval,
            electricity_tariff=electricity_tariff,
            free_grid_capacity=free_grid_capacity,
            ev_charging_load=ev_charging_load
        )
        
        # Extract storage action
        if storage_object.charge_yn == 1:
            return storage_object.charging_power
        elif storage_object.discharge_yn == 1:
            return -storage_object.discharging_power
        else:
            return 0.0
    
    def _peak_shaving(self, context: Dict[str, Any]) -> float:
        """Peak shaving storage algorithm."""
        env = context.get("env")
        storage_object = context.get("storage_object")
        planning_interval = context.get("planning_interval", 15)
        electricity_tariff = context.get("electricity_tariff", [0.15] * 24)
        free_grid_capacity = context.get("free_grid_capacity", 500)
        ev_charging_load = context.get("ev_charging_load", 100)
        
        # Execute algorithm
        peak_shaving(
            env=env,
            storage_object=storage_object,
            planning_interval=planning_interval,
            electricity_tariff=electricity_tariff,
            free_grid_capacity=free_grid_capacity,
            ev_charging_load=ev_charging_load
        )
        
        # Extract storage action
        if storage_object.charge_yn == 1:
            return storage_object.charging_power
        elif storage_object.discharge_yn == 1:
            return -storage_object.discharging_power
        else:
            return 0.0
    
    def learn(self, transition: Optional[Dict[str, Any]] = None) -> None:
        """Algorithm-based agents don't learn from transitions."""
        pass
