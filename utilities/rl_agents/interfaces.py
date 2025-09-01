from typing import Protocol, Any, Dict, List, Optional, Union
from enum import Enum


class DecisionType(Enum):
    """Types of decisions that can be made by agents"""
    PRICING = "pricing"
    CHARGING = "charging"
    STORAGE = "storage"
    ROUTING = "routing"
    VEHICLE_ASSIGNMENT = "vehicle_assignment"
    PARKING_ALLOCATION = "parking_allocation"
    GRID_MANAGEMENT = "grid_management"
    DEMAND_FORECASTING = "demand_forecasting"


class AgentType(Enum):
    """Types of agents that can make decisions"""
    RL_SAC = "rl_sac"
    RL_DQN = "rl_dqn"
    RL_DDPG = "rl_ddpg"
    RULE_BASED = "rule_based"
    HEURISTIC = "heuristic"
    OPTIMIZATION = "optimization"
    ML_MODEL = "ml_model"


class BaseAgent(Protocol):
    """
    Base interface for all decision-making agents.
    
    This protocol defines the standard interface that all agents must implement,
    regardless of whether they are RL agents, rule-based agents, or other types.
    """
    
    @property
    def agent_type(self) -> AgentType: ...
    
    @property
    def decision_type(self) -> DecisionType: ...
    
    def reset(self) -> None: ...
    
    def update_state(self, context: Dict[str, Any]) -> None: ...
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]: ...
    
    def learn(self, transition: Optional[Dict[str, Any]] = None) -> None: ...
    
    def get_state(self) -> Any: ...
    
    def set_state(self, state: Any) -> None: ...


class PricingAgent(BaseAgent):
    """
    Interface for pricing decision agents.
    
    Pricing agents make decisions about:
    - Energy prices
    - Parking fees
    - Dynamic pricing strategies
    - Price optimization
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.PRICING
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select pricing action based on current context.
        
        Args:
            context: Dictionary containing:
                - eval_ep: Whether this is evaluation mode
                - pricing_mode: "Discrete", "Continuous", "ToU"
                - charging_hub: Current charging hub state
                - env: Simulation environment
                - current_demand: Current energy demand
                - grid_capacity: Available grid capacity
                
        Returns:
            Dictionary containing:
                - pricing_parameters: List of pricing parameters
                - energy_price: Energy price per kWh
                - parking_fee: Parking fee per hour
                - confidence: Confidence in the decision (0-1)
        """
        ...


class ChargingAgent(BaseAgent):
    """
    Interface for charging decision agents.
    
    Charging agents make decisions about:
    - Charging power allocation
    - Charging schedules
    - Priority assignment
    - Load balancing
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.CHARGING
    
    def select_action(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select charging action based on vehicles and context.
        
        Args:
            vehicles: List of vehicles requiring charging decisions
            context: Dictionary containing:
                - eval_ep: Whether this is evaluation mode
                - charging_hub: Current charging hub state
                - env: Simulation environment
                - available_power: Available charging power
                - grid_constraints: Grid capacity constraints
                
        Returns:
            Dictionary containing:
                - charging_actions: List of charging actions per vehicle
                - power_allocation: Power allocation strategy
                - priority_order: Vehicle priority ordering
                - confidence: Confidence in the decision (0-1)
        """
        ...


class StorageAgent(BaseAgent):
    """
    Interface for storage decision agents.
    
    Storage agents make decisions about:
    - Energy storage charging/discharging
    - Storage scheduling
    - Peak shaving strategies
    - Grid support operations
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.STORAGE
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select storage action based on current context.
        
        Args:
            context: Dictionary containing:
                - eval_ep: Whether this is evaluation mode
                - charging_hub: Current charging hub state
                - env: Simulation environment
                - storage_soc: Current state of charge
                - grid_demand: Current grid demand
                - pv_generation: Current PV generation
                
        Returns:
            Dictionary containing:
                - storage_action: Charging/discharging action
                - power_level: Power level for storage operation
                - strategy: Storage strategy (peak_shaving, arbitrage, etc.)
                - confidence: Confidence in the decision (0-1)
        """
        ...


class RoutingAgent(BaseAgent):
    """
    Interface for routing decision agents.
    
    Routing agents make decisions about:
    - Vehicle routing to charging stations
    - Parking space allocation
    - Queue management
    - Resource assignment
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.ROUTING
    
    def select_action(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select routing action based on vehicles and context.
        
        Args:
            vehicles: List of vehicles requiring routing decisions
            context: Dictionary containing:
                - eval_ep: Whether this is evaluation mode
                - charging_hub: Current charging hub state
                - env: Simulation environment
                - available_spaces: Available parking/charging spaces
                - queue_status: Current queue status
                
        Returns:
            Dictionary containing:
                - routing_assignments: Vehicle to space assignments
                - queue_order: Queue ordering
                - wait_times: Estimated wait times
                - confidence: Confidence in the decision (0-1)
        """
        ...


class VehicleAssignmentAgent(BaseAgent):
    """
    Interface for vehicle assignment decision agents.
    
    Vehicle assignment agents make decisions about:
    - Which charging station to assign vehicles to
    - Charging connector allocation
    - Priority-based assignments
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.VEHICLE_ASSIGNMENT
    
    def select_action(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select vehicle assignment action.
        
        Args:
            vehicles: List of vehicles to assign
            context: Dictionary containing assignment context
            
        Returns:
            Dictionary containing assignment decisions
        """
        ...


class ParkingAllocationAgent(BaseAgent):
    """
    Interface for parking allocation decision agents.
    
    Parking allocation agents make decisions about:
    - Parking space allocation
    - Parking duration optimization
    - Space utilization
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.PARKING_ALLOCATION
    
    def select_action(self, vehicles: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select parking allocation action.
        
        Args:
            vehicles: List of vehicles requiring parking
            context: Dictionary containing parking context
            
        Returns:
            Dictionary containing parking allocation decisions
        """
        ...


class GridManagementAgent(BaseAgent):
    """
    Interface for grid management decision agents.
    
    Grid management agents make decisions about:
    - Grid capacity management
    - Load balancing
    - Grid stability
    - Peak demand management
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.GRID_MANAGEMENT
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select grid management action.
        
        Args:
            context: Dictionary containing grid management context
            
        Returns:
            Dictionary containing grid management decisions
        """
        ...


class DemandForecastingAgent(BaseAgent):
    """
    Interface for demand forecasting agents.
    
    Demand forecasting agents make decisions about:
    - Energy demand prediction
    - Load forecasting
    - Demand patterns analysis
    """
    
    @property
    def decision_type(self) -> DecisionType:
        return DecisionType.DEMAND_FORECASTING
    
    def select_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select demand forecasting action.
        
        Args:
            context: Dictionary containing forecasting context
            
        Returns:
            Dictionary containing demand forecasts
        """
        ...


