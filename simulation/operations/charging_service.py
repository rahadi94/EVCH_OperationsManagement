from typing import List, Dict, Any, Optional
from simulation.operations.agents_controller import AgentsController
from simulation.config_facade import ConfigFacade


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
    
    def take_learning_charging_actions(self, charging_strategy: str) -> None:
        """
        Execute learning-based charging actions using RL agents.
        
        Args:
            charging_strategy: The charging strategy to use
        """
        if charging_strategy == "dynamic":
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
        if charging_strategy == "dynamic":
            self.update_charging_agent()
    
    def take_charging_action(self) -> None:
        """
        Take charging action using the RL charging agent.
        """
        if self.agents_controller and self.agents_controller.charging:
            # Use the controller to get charging action
            context = {
                "charging_hub": self.op.charging_hub,
                "env": self.op.env
            }
            action_result = self.agents_controller.charging_step(
                vehicles=self.op.requests, 
                context=context
            )
            if action_result:
                self.op.charging_agent.action = action_result.get("charging_action")
        else:
            # Fallback to direct agent access (legacy behavior)
            state = self.op.charging_hub.charging_agent.environment.get_state(
                self.op.charging_hub, self.op.env
            )
            self.op.charging_agent.state = state

            eval_ep = self.op.charging_agent.do_evaluation_iterations
            self.op.charging_agent.episode_step_number_val = 0
            action = self.op.charging_agent.pick_action(eval_ep, self.op.charging_hub)
            self.op.charging_agent.action = self.op.charging_agent.rescale_action(action)
    
    def conduct_charging_action(self) -> None:
        """
        Execute the charging action by applying it to vehicles and chargers.
        """
        action = self.op.charging_agent.action
        action_index = 1  # Start from 1 because action[0] is reserved (possibly for pricing or metadata)

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

        self.op.check_charging_power()
        self.op.charging_hub.grid.reset_reward()
    
    def update_charging_agent(self) -> None:
        """
        Update the charging agent with new state and experience.
        """
        self.op.update_vehicles_status()
        self.op.charging_hub.reward["missed"] = self.op.reward_computing()

        eval_ep = self.op.charging_agent.do_evaluation_iterations
        self.op.charging_agent.conduct_action(self.op.charging_agent.action, self.op.charging_hub, self.op.env)
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
            >= self.op.charging_agent.environment.MAX_EPISODE_STEPS
            else self.op.charging_agent.done
        )
        # if not eval_ep:
        action = self.op.charging_agent.descale_action(self.op.charging_agent.action, self.op.charging_hub)
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
