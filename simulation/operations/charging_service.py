from typing import List, Dict, Any, Optional
from simulation.operations.agents_controller import AgentsController
from simulation.config_facade import ConfigFacade
from .decision_request_system import DecisionType, decision_system
from .decision_decorators import auto_register_agents


class ChargingService:
    """
    Service class for managing charging-related RL agent operations.
    
    Encapsulates all charging agent logic that was previously in the Operator class,
    providing a clean separation of concerns and standardized interface for RL agents.
    """
    
    def __init__(self, operator_instance: Any, agents_controller: Optional[AgentsController] = None, config_facade: Optional[ConfigFacade] = None):
        """
        Initialize the ChargingService.
        
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
    
    def take_learning_charging_actions(self, charging_strategy: str) -> None:
        """
        Execute learning-based charging actions using RL agents.
        
        Args:
            charging_strategy: The charging strategy to use
        """
        if self.op.charging_hub.dynamic_charging:
            self.op.update_vehicles_status()
            self.take_charging_action()
            self.conduct_charging_action()

            if self.op.storage_agent:
                self.op.get_exp_free_grid_capacity()
                # Storage actions are handled by StorageService
                if hasattr(self.op, 'storage_service'):
                    self.op.storage_service.take_storage_action()
                    self.op.storage_service.conduct_storage_action()
    
    def update_learning_charging_agent(self, charging_strategy: str) -> None:
        """
        Update the learning charging agent.
        
        Args:
            charging_strategy: The charging strategy to use
        """
        if self.op.charging_hub.dynamic_charging:
            self.update_charging_agent()
    
    def take_charging_action(self) -> None:
        """
        Take charging action using the RL charging agent.
        """
        if not self.op.charging_agent:
            return
            
        # Check if this is an RL agent with environment or a rule-based/algorithm agent
        if hasattr(self.op.charging_agent, 'environment'):
            # RL agent with environment - use direct RL agent logic
            charging_state = self.op.charging_agent.environment.get_state(self.op.charging_hub, self.op.env)
            self.op.charging_agent.state = charging_state
            eval_ep = self.op.charging_agent.do_evaluation_iterations

            # Handle different pick_action signatures
            import inspect
            sig = inspect.signature(self.op.charging_agent.pick_action)
            if len(sig.parameters) > 1:  # Method expects eval_ep parameter
                action = self.op.charging_agent.pick_action(eval_ep)
            else:  # Method doesn't expect eval_ep parameter
                action = self.op.charging_agent.pick_action()
                
            self.op.charging_agent.action = action
            
        else:
            # Rule-based/algorithm agent - use decision request system
            action = self._get_charging_decision_via_request()
            self.op.charging_agent.action = action
    
    def conduct_charging_action(self) -> None:
        """
        Execute the charging action by applying it to vehicles and chargers.
        """
        action = self.op.charging_agent.action
        
        # Check if this is an RL agent with environment
        if hasattr(self.op.charging_agent, 'environment'):
            # For RL agents, rescale the action from normalized range to actual power values
            if hasattr(self.op.charging_agent, 'rescale_action'):
                action = self.op.charging_agent.rescale_action(action)
            
            # For RL agents, use the environment's checked_action method to ensure feasibility
            if hasattr(self.op.charging_agent.environment, 'checked_action'):
                action = self.op.charging_agent.environment.checked_action(action)
            
            # Apply storage action (action[0])
            if len(action) > 0:
                storage_power = action[0]
                if storage_power >= 0:
                    self.op.charging_hub.electric_storage.charge_yn = 1
                    self.op.charging_hub.electric_storage.charging_power = storage_power
                elif storage_power < 0:
                    self.op.charging_hub.electric_storage.discharge_yn = 1
                    self.op.charging_hub.electric_storage.discharging_power = -storage_power
            
            # Apply charging actions (action[1:])
            action_index = 1  # Start from 1 because action[0] is for storage
            for charger in self.op.charging_hub.chargers:
                for connector_idx in range(charger.number_of_connectors):
                    if action_index >= len(action):
                        break  # Prevent index error if action list is shorter than expected

                    charging_power = action[action_index]
                    if charging_power > 0:
                        charging_vehicles = charger.charging_vehicles
                        if connector_idx < len(charging_vehicles):
                            vehicle = charging_vehicles[connector_idx]
                            vehicle.charging_power = charging_power
                    action_index += 1
        else:
            # For non-RL agents, use the original logic
            action_index = 1  # Start from 1 because action[0] is reserved
            for charger in self.op.charging_hub.chargers:
                for connector_idx in range(charger.number_of_connectors):
                    if action_index >= len(action):
                        break

                    charging_power = action[action_index]
                    if charging_power > 0:
                        charging_vehicles = charger.charging_vehicles
                        if connector_idx < len(charging_vehicles):
                            vehicle = charging_vehicles[connector_idx]
                            vehicle.charging_power = charging_power
                    action_index += 1

        self.op.charging_hub.reward["profit"] = self.op.reward_computing()
        self.op.charging_hub.grid.reset_reward()
    
    def update_charging_agent(self) -> None:
        """
        Update the charging agent with new state and experience.
        """
        if not self.op.charging_agent:
            return
            
        # Check if this is an RL agent with environment
        if hasattr(self.op.charging_agent, 'environment'):
            # Initialize episode step number if not already set
            if not hasattr(self.op.charging_agent, 'episode_step_number_val'):
                self.op.charging_agent.episode_step_number_val = 0
            
            # RL agent update logic
            self.op.update_vehicles_status()
            self.op.charging_hub.reward["missed"] = self.op.reward_computing()

            eval_ep = self.op.charging_agent.do_evaluation_iterations
            self.op.charging_agent.conduct_action(self.op.charging_agent.action)
            if self.op.charging_agent.time_for_critic_and_actor_to_learn():
                if not eval_ep:
                    for _ in range(
                        self.op.charging_agent.hyperparameters[
                            "learning_updates_per_learning_session"
                        ]
                    ):
                        self.op.charging_agent.learn()
            mask = (
                False
                if self.op.charging_agent.episode_step_number_val
                >= self.op.charging_agent.environment._max_episode_steps
                else self.op.charging_agent.done
            )
            # if not eval_ep:
            action = self.op.charging_agent.descale_action(self.op.charging_agent.action)
            self.op.charging_agent.save_experience(
                experience=(
                    self.op.charging_agent.state,
                    action,
                    self.op.charging_agent.reward,
                    self.op.charging_agent.next_state,
                    mask,
                )
            )
            self.op.charging_agent.global_step_number += 1
            self.op.charging_agent.step_counter += 1
        else:
            # Non-RL agent update (minimal)
            if hasattr(self.op.charging_agent, 'agent_name'):
                print(f"Updated {self.op.charging_agent.__class__.__name__} (no learning required)")
            else:
                print(f"Updated {self.op.charging_agent.__class__.__name__} (no learning required)")

    def _get_charging_decision_via_request(self) -> Any:
        """
        Get charging decision through the decision request system.
        
        Returns:
            The charging action/decision
        """
        # Get current state from environment
        state = self.op.charging_hub.charging_agent.environment.get_state(
            self.op.charging_hub, self.op.env
        )
        self.op.charging_agent.state = state

        eval_ep = self.op.charging_agent.do_evaluation_iterations
        self.op.charging_agent.episode_step_number_val = 0
        
        # Create context for the decision request
        context = {
            "eval_ep": eval_ep,
            "charging_hub": self.op.charging_hub,
            "env": self.op.env,
            "vehicles": self.op.requests
        }
        
        # Create and process decision request
        request_id = decision_system.create_request(
            agent_type=DecisionType.CHARGING,
            state=self.op.charging_agent.state,
            context=context,
            metadata={
                "agent_name": getattr(self.op.charging_agent, "agent_name", "Unknown")
            }
        )
        
        # Process the request
        response = decision_system.process_request(request_id)
        
        if response:
            # Rescale action if needed
            if hasattr(self.op.charging_agent, "rescale_action"):
                return self.op.charging_agent.rescale_action(response.action)
            else:
                return response.action
        else:
            # Fallback to direct agent call if request system fails
            # Handle different pick_action signatures
            import inspect
            sig = inspect.signature(self.op.charging_agent.pick_action)
            if len(sig.parameters) > 1:  # Method expects eval_ep parameter
                action = self.op.charging_agent.pick_action(eval_ep)
            else:  # Method doesn't expect eval_ep parameter
                action = self.op.charging_agent.pick_action()
                
            if hasattr(self.op.charging_agent, "rescale_action"):
                return self.op.charging_agent.rescale_action(action)
            else:
                return action
