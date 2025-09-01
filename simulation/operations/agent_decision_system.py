from typing import Any, Dict, List, Optional, Union, Type
import logging
from dataclasses import dataclass
from datetime import datetime
import uuid

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
from simulation.operations.decision_request_system import (
    DecisionRequestSystem, 
    DecisionRequest, 
    DecisionResponse,
    RequestStatus,
    decision_system
)

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    """Represents a decision made by an agent"""
    decision_id: str
    agent_type: AgentType
    decision_type: DecisionType
    context: Dict[str, Any]
    action: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]


class AgentDecisionSystem:
    """
    Centralized system for managing all agent-based decisions.
    
    This system ensures that ALL decisions in the EV charging operations
    are made by agents (RL agents, rule-based agents, etc.) rather than
    being hardcoded in the business logic.
    
    Key principles:
    1. No decisions in business logic - all decisions go through agents
    2. Standardized interface for all agents
    3. Comprehensive tracking and monitoring
    4. Fallback mechanisms for reliability
    5. Support for multiple agent types (RL, rule-based, ML, etc.)
    """
    
    def __init__(self):
        self.agents: Dict[DecisionType, BaseAgent] = {}
        self.decision_history: List[AgentDecision] = []
        self.agent_registry: Dict[str, Type[BaseAgent]] = {}
        self.decision_callbacks: Dict[str, callable] = {}
        
    def register_agent(self, decision_type: DecisionType, agent: BaseAgent) -> None:
        """
        Register an agent for a specific decision type.
        
        Args:
            decision_type: The type of decision this agent can make
            agent: The agent instance
        """
        self.agents[decision_type] = agent
        logger.info(f"Registered {agent.__class__.__name__} for {decision_type.value} decisions")
        
    def register_agent_class(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register an agent class for dynamic instantiation.
        
        Args:
            name: Name identifier for the agent class
            agent_class: The agent class to register
        """
        self.agent_registry[name] = agent_class
        logger.info(f"Registered agent class {name}: {agent_class.__name__}")
        
    def make_decision(
        self, 
        decision_type: DecisionType, 
        context: Dict[str, Any],
        vehicles: Optional[List[Any]] = None,
        priority: int = 1,
        timeout_seconds: float = 30.0
    ) -> AgentDecision:
        """
        Make a decision using the appropriate agent.
        
        This is the main entry point for all decisions in the system.
        Every decision request goes through this method.
        
        Args:
            decision_type: Type of decision needed
            context: Context information for the decision
            vehicles: List of vehicles (for vehicle-related decisions)
            priority: Decision priority (1-10, higher = more important)
            timeout_seconds: Timeout for the decision
            
        Returns:
            AgentDecision object containing the decision result
            
        Raises:
            ValueError: If no agent is registered for the decision type
        """
        if decision_type not in self.agents:
            raise ValueError(f"No agent registered for decision type: {decision_type.value}")
            
        agent = self.agents[decision_type]
        decision_id = str(uuid.uuid4())
        
        # Create decision request for tracking
        request_id = decision_system.create_request(
            agent_type=decision_type,
            state=agent.get_state(),
            context=context,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata={
                "agent_type": agent.agent_type.value,
                "decision_id": decision_id
            }
        )
        
        try:
            # Update agent state
            agent.update_state(context)
            
            # Make decision based on agent type
            if decision_type in [DecisionType.CHARGING, DecisionType.ROUTING, 
                               DecisionType.VEHICLE_ASSIGNMENT, DecisionType.PARKING_ALLOCATION]:
                # Vehicle-related decisions
                if vehicles is None:
                    vehicles = []
                action_result = agent.select_action(vehicles, context)
            else:
                # Non-vehicle decisions
                action_result = agent.select_action(context)
            
            # Process the request
            response = decision_system.process_request(request_id)
            
            # Create decision record
            decision = AgentDecision(
                decision_id=decision_id,
                agent_type=agent.agent_type,
                decision_type=decision_type,
                context=context,
                action=action_result,
                confidence=action_result.get("confidence", 0.5),
                timestamp=datetime.now(),
                metadata={
                    "request_id": request_id,
                    "agent_class": agent.__class__.__name__,
                    "vehicles_count": len(vehicles) if vehicles else 0
                }
            )
            
            # Store decision in history
            self.decision_history.append(decision)
            
            # Call any registered callbacks
            if decision_type.value in self.decision_callbacks:
                self.decision_callbacks[decision_type.value](decision)
            
            logger.info(f"Decision made: {decision_type.value} by {agent.__class__.__name__}")
            return decision
            
        except Exception as e:
            logger.error(f"Error making {decision_type.value} decision: {e}")
            # Mark request as failed
            if request_id in decision_system.requests:
                decision_system.requests[request_id].status = RequestStatus.FAILED
                decision_system.requests[request_id].metadata["error"] = str(e)
            raise
    
    def make_pricing_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Make a pricing decision using the pricing agent."""
        return self.make_decision(DecisionType.PRICING, context)
    
    def make_charging_decision(self, vehicles: List[Any], context: Dict[str, Any]) -> AgentDecision:
        """Make a charging decision using the charging agent."""
        return self.make_decision(DecisionType.CHARGING, context, vehicles)
    
    def make_storage_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Make a storage decision using the storage agent."""
        return self.make_decision(DecisionType.STORAGE, context)
    
    def make_routing_decision(self, vehicles: List[Any], context: Dict[str, Any]) -> AgentDecision:
        """Make a routing decision using the routing agent."""
        return self.make_decision(DecisionType.ROUTING, context, vehicles)
    
    def make_vehicle_assignment_decision(self, vehicles: List[Any], context: Dict[str, Any]) -> AgentDecision:
        """Make a vehicle assignment decision using the vehicle assignment agent."""
        return self.make_decision(DecisionType.VEHICLE_ASSIGNMENT, context, vehicles)
    
    def make_parking_allocation_decision(self, vehicles: List[Any], context: Dict[str, Any]) -> AgentDecision:
        """Make a parking allocation decision using the parking allocation agent."""
        return self.make_decision(DecisionType.PARKING_ALLOCATION, context, vehicles)
    
    def make_grid_management_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Make a grid management decision using the grid management agent."""
        return self.make_decision(DecisionType.GRID_MANAGEMENT, context)
    
    def make_demand_forecasting_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """Make a demand forecasting decision using the demand forecasting agent."""
        return self.make_decision(DecisionType.DEMAND_FORECASTING, context)
    
    def register_decision_callback(self, decision_type: DecisionType, callback: callable) -> None:
        """
        Register a callback to be called when a decision is made.
        
        Args:
            decision_type: The decision type to monitor
            callback: Function to call with the decision result
        """
        self.decision_callbacks[decision_type.value] = callback
        logger.info(f"Registered callback for {decision_type.value} decisions")
    
    def get_decision_history(self, decision_type: Optional[DecisionType] = None) -> List[AgentDecision]:
        """
        Get decision history, optionally filtered by decision type.
        
        Args:
            decision_type: Optional filter for specific decision type
            
        Returns:
            List of decisions
        """
        if decision_type:
            return [d for d in self.decision_history if d.decision_type == decision_type]
        return self.decision_history.copy()
    
    def get_agent_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for all agents.
        
        Returns:
            Dictionary containing performance statistics
        """
        stats = {}
        
        for decision_type, agent in self.agents.items():
            decisions = self.get_decision_history(decision_type)
            
            if decisions:
                avg_confidence = sum(d.confidence for d in decisions) / len(decisions)
                success_rate = len([d for d in decisions if d.confidence > 0.5]) / len(decisions)
                
                stats[decision_type.value] = {
                    "agent_type": agent.agent_type.value,
                    "agent_class": agent.__class__.__name__,
                    "total_decisions": len(decisions),
                    "average_confidence": avg_confidence,
                    "success_rate": success_rate,
                    "last_decision": decisions[-1].timestamp if decisions else None
                }
            else:
                stats[decision_type.value] = {
                    "agent_type": agent.agent_type.value,
                    "agent_class": agent.__class__.__name__,
                    "total_decisions": 0,
                    "average_confidence": 0.0,
                    "success_rate": 0.0,
                    "last_decision": None
                }
        
        return stats
    
    def reset_agents(self) -> None:
        """Reset all registered agents."""
        for agent in self.agents.values():
            agent.reset()
        logger.info("All agents reset")
    
    def cleanup_old_decisions(self, max_age_hours: float = 24.0) -> None:
        """
        Clean up old decisions from history.
        
        Args:
            max_age_hours: Maximum age of decisions to keep
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        original_count = len(self.decision_history)
        self.decision_history = [
            d for d in self.decision_history
            if d.timestamp.timestamp() > cutoff_time
        ]
        
        removed_count = original_count - len(self.decision_history)
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old decisions")


# Global instance for easy access
agent_decision_system = AgentDecisionSystem()
