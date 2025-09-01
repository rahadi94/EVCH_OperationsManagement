from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import uuid
from datetime import datetime


class DecisionType(Enum):
    """Types of decisions that RL agents can make"""
    PRICING = "pricing"
    CHARGING = "charging"
    STORAGE = "storage"
    ROUTING = "routing"


class RequestStatus(Enum):
    """Status of a decision request"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class DecisionRequest:
    """Represents a decision request for an RL agent"""
    request_id: str
    agent_type: DecisionType
    state: Any
    context: Dict[str, Any]
    timestamp: datetime
    status: RequestStatus
    priority: int = 1
    timeout_seconds: float = 30.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class DecisionResponse:
    """Represents a response to a decision request"""
    request_id: str
    action: Any
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DecisionRequestSystem:
    """
    Centralized system for managing decision requests from RL agents.
    
    This system provides a standardized way for RL agents to request decisions,
    track request status, and handle responses. It supports:
    - Request queuing and prioritization
    - Timeout handling
    - Request tracking and logging
    - Integration with existing RL agent infrastructure
    """
    
    def __init__(self):
        self.requests: Dict[str, DecisionRequest] = {}
        self.responses: Dict[str, DecisionResponse] = {}
        self.request_history: List[DecisionRequest] = []
        self.agent_handlers: Dict[DecisionType, Any] = {}
        self.request_callbacks: Dict[str, callable] = {}
        
    def register_agent_handler(self, decision_type: DecisionType, handler: Any) -> None:
        """
        Register an agent handler for a specific decision type.
        
        Args:
            decision_type: The type of decision this handler can process
            handler: The agent object that can make decisions
        """
        self.agent_handlers[decision_type] = handler
        
    def create_request(
        self,
        agent_type: DecisionType,
        state: Any,
        context: Dict[str, Any],
        priority: int = 1,
        timeout_seconds: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new decision request.
        
        Args:
            agent_type: Type of decision needed
            state: Current state for the agent
            context: Additional context information
            priority: Request priority (higher = more important)
            timeout_seconds: Timeout for the request
            metadata: Additional metadata
            
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        
        request = DecisionRequest(
            request_id=request_id,
            agent_type=agent_type,
            state=state,
            context=context,
            timestamp=datetime.now(),
            status=RequestStatus.PENDING,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata=metadata or {}
        )
        
        self.requests[request_id] = request
        return request_id
    
    def process_request(self, request_id: str) -> Optional[DecisionResponse]:
        """
        Process a decision request using the appropriate agent.
        
        Args:
            request_id: ID of the request to process
            
        Returns:
            Decision response if successful, None if failed
        """
        if request_id not in self.requests:
            raise ValueError(f"Request {request_id} not found")
            
        request = self.requests[request_id]
        
        # Check if agent handler exists
        if request.agent_type not in self.agent_handlers:
            request.status = RequestStatus.FAILED
            request.metadata["error"] = f"No handler registered for {request.agent_type}"
            return None
            
        # Check timeout
        if self._is_request_timed_out(request):
            request.status = RequestStatus.TIMEOUT
            return None
            
        # Update status
        request.status = RequestStatus.PROCESSING
        
        try:
            # Get the appropriate agent handler
            agent = self.agent_handlers[request.agent_type]
            
            # Process the request based on agent type
            response = self._process_with_agent(agent, request)
            
            if response:
                request.status = RequestStatus.COMPLETED
                self.responses[request_id] = response
            else:
                request.status = RequestStatus.FAILED
                
            return response
            
        except Exception as e:
            request.status = RequestStatus.FAILED
            request.metadata["error"] = str(e)
            return None
    
    def _process_with_agent(self, agent: Any, request: DecisionRequest) -> Optional[DecisionResponse]:
        """
        Process request with the specific agent type.
        
        Args:
            agent: The agent to use for decision making
            request: The decision request
            
        Returns:
            Decision response
        """
        try:
            if request.agent_type == DecisionType.PRICING:
                return self._process_pricing_request(agent, request)
            elif request.agent_type == DecisionType.CHARGING:
                return self._process_charging_request(agent, request)
            elif request.agent_type == DecisionType.STORAGE:
                return self._process_storage_request(agent, request)
            elif request.agent_type == DecisionType.ROUTING:
                return self._process_routing_request(agent, request)
            else:
                raise ValueError(f"Unknown decision type: {request.agent_type}")
                
        except Exception as e:
            request.metadata["processing_error"] = str(e)
            return None
    
    def _process_pricing_request(self, agent: Any, request: DecisionRequest) -> DecisionResponse:
        """Process pricing decision request"""
        # Set agent state
        agent.state = request.state
        
        # Get evaluation flag from context
        eval_ep = request.context.get("eval_ep", False)
        
        # Get action from agent
        if hasattr(agent, "pick_action"):
            # Handle different pick_action signatures
            import inspect
            sig = inspect.signature(agent.pick_action)
            if len(sig.parameters) > 1:  # Method expects eval_ep parameter
                action = agent.pick_action(eval_ep)
            else:  # Method doesn't expect eval_ep parameter
                action = agent.pick_action()
        else:
            raise ValueError("Agent does not have pick_action method")
            
        return DecisionResponse(
            request_id=request.request_id,
            action=action,
            metadata={"agent_type": "pricing", "eval_ep": eval_ep}
        )
    
    def _process_charging_request(self, agent: Any, request: DecisionRequest) -> DecisionResponse:
        """Process charging decision request"""
        # Set agent state
        agent.state = request.state
        
        # Get evaluation flag from context
        eval_ep = request.context.get("eval_ep", False)
        
        # Get action from agent
        if hasattr(agent, "pick_action"):
            # Handle different pick_action signatures
            import inspect
            sig = inspect.signature(agent.pick_action)
            if len(sig.parameters) > 1:  # Method expects eval_ep parameter
                action = agent.pick_action(eval_ep)
            else:  # Method doesn't expect eval_ep parameter
                action = agent.pick_action()
        else:
            raise ValueError("Agent does not have pick_action method")
            
        # Rescale action if needed
        if hasattr(agent, "rescale_action"):
            action = agent.rescale_action(action)
            
        return DecisionResponse(
            request_id=request.request_id,
            action=action,
            metadata={"agent_type": "charging", "eval_ep": eval_ep}
        )
    
    def _process_storage_request(self, agent: Any, request: DecisionRequest) -> DecisionResponse:
        """Process storage decision request"""
        # Set agent state
        agent.state = request.state
        
        # Get evaluation flag from context
        eval_ep = request.context.get("eval_ep", False)
        charging_hub = request.context.get("charging_hub", None)
        
        # Get action from agent
        if hasattr(agent, "pick_action"):
            # Handle different pick_action signatures
            import inspect
            sig = inspect.signature(agent.pick_action)
            if len(sig.parameters) > 2:  # Method expects eval_ep and charging_hub parameters
                action = agent.pick_action(eval_ep, charging_hub)
            elif len(sig.parameters) > 1:  # Method expects eval_ep parameter
                action = agent.pick_action(eval_ep)
            else:  # Method doesn't expect eval_ep parameter
                action = agent.pick_action()
        else:
            raise ValueError("Agent does not have pick_action method")
            
        return DecisionResponse(
            request_id=request.request_id,
            action=action,
            metadata={"agent_type": "storage", "eval_ep": eval_ep}
        )
    
    def _process_routing_request(self, agent: Any, request: DecisionRequest) -> DecisionResponse:
        """Process routing decision request"""
        # For routing, we might need different logic
        # This is a placeholder for future implementation
        raise NotImplementedError("Routing decisions not yet implemented")
    
    def get_response(self, request_id: str) -> Optional[DecisionResponse]:
        """Get the response for a request"""
        return self.responses.get(request_id)
    
    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """Get the status of a request"""
        if request_id in self.requests:
            return self.requests[request_id].status
        return None
    
    def _is_request_timed_out(self, request: DecisionRequest) -> bool:
        """Check if a request has timed out"""
        elapsed = (datetime.now() - request.timestamp).total_seconds()
        return elapsed > request.timeout_seconds
    
    def cleanup_old_requests(self, max_age_hours: float = 24.0) -> None:
        """Clean up old requests and responses"""
        current_time = datetime.now()
        cutoff_time = current_time.timestamp() - (max_age_hours * 3600)
        
        # Move old requests to history
        old_requests = [
            req for req in self.requests.values()
            if req.timestamp.timestamp() < cutoff_time
        ]
        
        for req in old_requests:
            self.request_history.append(req)
            del self.requests[req.request_id]
            if req.request_id in self.responses:
                del self.responses[req.request_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the request system"""
        total_requests = len(self.requests) + len(self.request_history)
        completed = len([r for r in self.requests.values() if r.status == RequestStatus.COMPLETED])
        failed = len([r for r in self.requests.values() if r.status == RequestStatus.FAILED])
        pending = len([r for r in self.requests.values() if r.status == RequestStatus.PENDING])
        
        return {
            "total_requests": total_requests,
            "active_requests": len(self.requests),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "success_rate": completed / total_requests if total_requests > 0 else 0.0
        }


# Global instance for easy access
decision_system = DecisionRequestSystem()
