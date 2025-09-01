from typing import List, Dict, Any, Optional
from simulation.operations.agents_controller import AgentsController
from simulation.config_facade import ConfigFacade
from .decision_request_system import DecisionType, decision_system
from .decision_decorators import auto_register_agents


class StorageService:
    """
    Service class for managing storage-related RL agent operations.
    
    Encapsulates all storage agent logic that was previously in the Operator class,
    providing a clean separation of concerns and standardized interface for RL agents.
    """
    
    def __init__(self, operator_instance: Any, agents_controller: Optional[AgentsController] = None, config_facade: Optional[ConfigFacade] = None):
        """
        Initialize the StorageService.
        
        Args:
            operator_instance: Reference to the main operator instance
            agents_controller: Controller for managing RL agents
            config_facade: Facade for accessing configuration values
        """
        self.op = operator_instance
        self.agents_controller = agents_controller
        self.config = config_facade or ConfigFacade()
        
        # Register agents with the decision request system
        auto_register_agents(operator_instance)
    
    def take_storage_action(self) -> None:
        """
        Take storage action using the RL storage agent.
        """
        if self.agents_controller and self.agents_controller.storage:
            # Use the controller to get storage action
            context = {
                "charging_hub": self.op.charging_hub,
                "env": self.op.env
            }
            action_result = self.agents_controller.storage_step(context)
            if action_result:
                self.op.storage_agent.action = action_result.get("storage_action")
        else:
            # Use decision request system for storage decisions
            action = self._get_storage_decision_via_request()
            self.op.storage_agent.action = action
    
    def conduct_storage_action(self, given_storage_action: Optional[List[float]] = None) -> None:
        """
        Execute the storage action by applying it to the electric storage system.
        
        Args:
            given_storage_action: Optional storage action to use instead of agent's action
        """
        if given_storage_action:
            storage_power = given_storage_action[0]
        else:
            storage_power = self.op.storage_agent.action[0]
            
        if storage_power >= 0:
            self.op.charging_hub.electric_storage.charge_yn = 1
            self.op.charging_hub.electric_storage.charging_power = storage_power
            self.op.charging_hub.electric_storage.discharge_yn = 0
            self.op.charging_hub.electric_storage.discharging_power = 0
        elif storage_power < 0:
            self.op.charging_hub.electric_storage.charge_yn = 0
            self.op.charging_hub.electric_storage.charging_power = 0
            self.op.charging_hub.electric_storage.discharge_yn = 1
            self.op.charging_hub.electric_storage.discharging_power = -storage_power
            
        self.op.check_storage(given_storage_action=given_storage_action)
    
    def update_storage_agent(self) -> None:
        """
        Update the storage agent with new state and experience.
        """
        eval_ep = self.op.storage_agent.do_evaluation_iterations
        action = self.op.storage_agent.descale_action(
            self.op.storage_agent.action, self.op.charging_hub
        )
        self.op.storage_agent.conduct_action(action, self.op.charging_hub, self.op.env, eval_ep=eval_ep)
        if self.op.storage_agent.time_for_critic_and_actor_to_learn():
            for _ in range(
                self.op.storage_agent.hyperparameters[
                    "learning_updates_per_learning_session"
                ]
            ):
                self.op.storage_agent.learn()
        mask = (
            False
            if self.op.storage_agent.episode_step_number_val
            >= self.op.storage_agent.environment.MAX_EPISODE_STEPS
            else self.op.storage_agent.done
        )
        # if not eval_ep:

        self.op.storage_agent.save_experience(
            experience=(
                self.op.storage_agent.state,
                action,
                self.op.storage_agent.reward,
                self.op.storage_agent.next_state,
                mask,
            )
        )
        self.op.storage_agent.global_step_number += 1
        self.op.storage_agent.step_counter += 1

    def _get_storage_decision_via_request(self) -> Any:
        """
        Get storage decision through the decision request system.
        
        Returns:
            The storage action/decision
        """
        # Get current state from environment
        storage_state = self.op.charging_hub.storage_agent.environment.get_state(
            self.op.charging_hub, self.op.env
        )
        self.op.storage_agent.state = storage_state

        eval_ep = self.op.storage_agent.do_evaluation_iterations
        self.op.storage_agent.episode_step_number_val = 0
        
        # Create context for the decision request
        context = {
            "eval_ep": eval_ep,
            "charging_hub": self.op.charging_hub,
            "env": self.op.env
        }
        
        # Create and process decision request
        request_id = decision_system.create_request(
            agent_type=DecisionType.STORAGE,
            state=self.op.storage_agent.state,
            context=context,
            metadata={
                "agent_name": getattr(self.op.storage_agent, "agent_name", "Unknown")
            }
        )
        
        # Process the request
        response = decision_system.process_request(request_id)
        
        if response:
            return response.action
        else:
            # Fallback to direct agent call if request system fails
            # Handle different pick_action signatures
            import inspect
            sig = inspect.signature(self.op.storage_agent.pick_action)
            if len(sig.parameters) > 2:  # Method expects eval_ep and charging_hub parameters
                return self.op.storage_agent.pick_action(eval_ep, self.op.charging_hub)
            elif len(sig.parameters) > 1:  # Method expects only eval_ep parameter
                return self.op.storage_agent.pick_action(eval_ep)
            else:  # Method doesn't expect eval_ep parameter
                return self.op.storage_agent.pick_action()
