from typing import Any, Dict, Optional, Callable
from functools import wraps
import logging
from .decision_request_system import (
    DecisionRequestSystem, 
    DecisionType, 
    decision_system
)

logger = logging.getLogger(__name__)


def require_decision_request(decision_type: DecisionType, timeout_seconds: float = 30.0):
    """
    Decorator that automatically creates a decision request when an RL agent method is called.
    
    This decorator can be applied to methods like pick_action() to ensure that every
    decision is tracked through the request system.
    
    Args:
        decision_type: The type of decision being made
        timeout_seconds: Timeout for the request
        
    Example:
        @require_decision_request(DecisionType.PRICING)
        def pick_action(self, eval_ep=False):
            # Original pick_action implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create context from method arguments
            context = {
                "method_name": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "eval_ep": kwargs.get("eval_ep", False)
            }
            
            # Get current state from the agent
            state = getattr(self, "state", None)
            
            # Create decision request
            request_id = decision_system.create_request(
                agent_type=decision_type,
                state=state,
                context=context,
                timeout_seconds=timeout_seconds,
                metadata={
                    "agent_class": self.__class__.__name__,
                    "method": func.__name__
                }
            )
            
            logger.info(f"Created decision request {request_id} for {decision_type.value}")
            
            try:
                # Process the request immediately
                response = decision_system.process_request(request_id)
                
                if response:
                    logger.info(f"Decision request {request_id} completed successfully")
                    return response.action
                else:
                    logger.warning(f"Decision request {request_id} failed, falling back to direct call")
                    # Fallback to original method
                    return func(self, *args, **kwargs)
                    
            except Exception as e:
                logger.error(f"Error processing decision request {request_id}: {e}")
                # Fallback to original method
                return func(self, *args, **kwargs)
                
        return wrapper
    return decorator


def track_decision(decision_type: DecisionType):
    """
    Decorator that tracks decisions without requiring the request system.
    
    This is a lighter-weight decorator that just logs decisions without
    going through the full request system.
    
    Args:
        decision_type: The type of decision being made
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Log the decision attempt
            logger.info(f"Making {decision_type.value} decision via {func.__name__}")
            
            # Call the original method
            result = func(self, *args, **kwargs)
            
            # Log the decision result
            logger.info(f"Completed {decision_type.value} decision: {result}")
            
            return result
        return wrapper
    return decorator


class DecisionRequestMixin:
    """
    Mixin class that provides decision request functionality to RL agents.
    
    This mixin can be added to RL agent classes to provide standardized
    decision request capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._decision_type = None
        self._last_request_id = None
        
    def set_decision_type(self, decision_type: DecisionType) -> None:
        """Set the decision type for this agent"""
        self._decision_type = decision_type
        
    def make_decision_request(
        self, 
        state: Any, 
        context: Optional[Dict[str, Any]] = None,
        priority: int = 1,
        timeout_seconds: float = 30.0
    ) -> Optional[Any]:
        """
        Make a decision request through the decision system.
        
        Args:
            state: Current state for the agent
            context: Additional context
            priority: Request priority
            timeout_seconds: Request timeout
            
        Returns:
            The decision/action if successful, None if failed
        """
        if not self._decision_type:
            raise ValueError("Decision type not set for this agent")
            
        if context is None:
            context = {}
            
        # Create request
        request_id = decision_system.create_request(
            agent_type=self._decision_type,
            state=state,
            context=context,
            priority=priority,
            timeout_seconds=timeout_seconds,
            metadata={
                "agent_class": self.__class__.__name__,
                "agent_id": id(self)
            }
        )
        
        self._last_request_id = request_id
        
        # Process request
        response = decision_system.process_request(request_id)
        
        if response:
            return response.action
        else:
            return None
            
    def get_last_request_status(self) -> Optional[str]:
        """Get the status of the last request made by this agent"""
        if self._last_request_id:
            status = decision_system.get_request_status(self._last_request_id)
            return status.value if status else None
        return None


def register_agent_with_system(agent: Any, decision_type: DecisionType) -> None:
    """
    Register an agent with the decision request system.
    
    Args:
        agent: The RL agent to register
        decision_type: The type of decisions this agent can make
    """
    decision_system.register_agent_handler(decision_type, agent)
    logger.info(f"Registered {agent.__class__.__name__} for {decision_type.value} decisions")


def auto_register_agents(operator_instance: Any) -> None:
    """
    Automatically register all agents from an operator instance with the decision system.
    
    Args:
        operator_instance: The operator instance containing agents
    """
    from .decision_request_system import DecisionType
    
    # Register pricing agent
    if hasattr(operator_instance, "pricing_agent") and operator_instance.pricing_agent:
        register_agent_with_system(operator_instance.pricing_agent, DecisionType.PRICING)
        
    # Register charging agent
    if hasattr(operator_instance, "charging_agent") and operator_instance.charging_agent:
        register_agent_with_system(operator_instance.charging_agent, DecisionType.CHARGING)
        
    # Register storage agent
    if hasattr(operator_instance, "storage_agent") and operator_instance.storage_agent:
        register_agent_with_system(operator_instance.storage_agent, DecisionType.STORAGE)
        
    logger.info("Auto-registered agents with decision request system")
