from typing import Any, Dict
import pandas as pd
from resources.configuration.configuration import Configuration
from utilities.rl_environments.rl_pricing_env import convert_to_vector
from .decision_request_system import DecisionType, decision_system
from .decision_decorators import auto_register_agents


class PricingService:
    """
    Encapsulates all pricing-related behavior away from Operator.
    Accesses Operator state via the provided reference.
    """

    def __init__(self, operator: Any, agents_controller: Any | None = None):
        self.op = operator
        self.agents_controller = agents_controller
        
        # Register agents with the decision request system
        auto_register_agents(operator)

    # Public APIs used by Operator
    def take_dynamic_pricing_actions(self) -> None:
        self.op.get_exp_free_grid_capacity()
        self.op.update_vehicles_status()

        # Prefer external agent when available
        used_agent = False
        if getattr(self, "agents_controller", None) and getattr(self.agents_controller, "pricing", None):
            context: Dict[str, Any] = {"charging_hub": self.op.charging_hub, "env": self.op.env}
            action_dict = self.agents_controller.pricing_step(context)
            if action_dict and "action" in action_dict:
                action = action_dict["action"]
                try:
                    import numpy as _np
                    if _np.isscalar(action):
                        action = convert_to_vector(int(action), h=1)
                    action_list = action.tolist() if hasattr(action, "tolist") else list(action)
                except Exception:
                    action_list = [action]

                if not hasattr(self.op, "pricing_parameters") or self.op.pricing_parameters is None:
                    self.op.pricing_parameters = [0.0, 0.0]
                if len(action_list) >= 1:
                    self.op.pricing_parameters[0] = action_list[0]
                if len(action_list) >= 2:
                    if len(self.op.pricing_parameters) < 2:
                        self.op.pricing_parameters.append(0.0)
                    self.op.pricing_parameters[1] = action_list[1]
                used_agent = True

        if not used_agent:
            # Fallback to previous behavior
            self.take_pricing_action()

        self._update_dynamic_price_history()

    def take_static_pricing_action(self) -> None:
        self.op.get_exp_free_grid_capacity()
        self.op.update_vehicles_status()
        self._update_pricing_parameters()
        self._update_static_price_history()

    def take_pricing_action(self) -> None:
        # Get current state from environment
        pricing_state = self.op.pricing_agent.environment.get_state(self.op.charging_hub, self.op.env)
        self.op.pricing_agent.state = pricing_state
        eval_ep = self.op.pricing_agent.do_evaluation_iterations

        pricing_mode = Configuration.instance().pricing_mode
        agent_name = self.op.pricing_agent.agent_name

        if pricing_mode == "Discrete":
            if agent_name == "DQN":
                # Use decision request system for DQN pricing
                action = self._get_pricing_decision_via_request(eval_ep=False)
                self.op.pricing_agent.action = action
                
                if len(self.op.price_pairs[:, 1]) > 1:
                    vector_prices = convert_to_vector(self.op.pricing_agent.action)
                else:
                    vector_prices = [self.op.pricing_agent.action]
                final_pricing = self.op.pricing_agent.environment.get_final_prices_DQN(vector_prices)
                for i, price in enumerate(final_pricing):
                    self.op.price_pairs[i, 1] = price

            elif agent_name == "SAC":
                # Use decision request system for SAC pricing
                action = self._get_pricing_decision_via_request(eval_ep)
                self.op.pricing_agent.action = action
                
                rescaled_actions = self.op.pricing_agent.environment.rescale_action(self.op.pricing_agent.action)
                number_of_power_options = len(self.op.price_pairs[:, 1])
                final_pricing = rescaled_actions[:number_of_power_options]
                self.op.price_pairs[0, 1] = final_pricing[0]
                self.op.price_pairs[1, 1] = min(final_pricing[1], 1.5)

        elif pricing_mode == "Continuous":
            # Use decision request system for continuous pricing
            action = self._get_pricing_decision_via_request(eval_ep)
            self.op.pricing_agent.action = action
            
            rescaled_actions = self.op.pricing_agent.environment.rescale_action(self.op.pricing_agent.action)

            config = Configuration.instance()
            if not config.dynamic_fix_term_pricing and config.capacity_pricing:
                self.op.pricing_parameters[1] = rescaled_actions[0]

            elif config.dynamic_fix_term_pricing and not config.capacity_pricing:
                self.op.pricing_parameters[0] = rescaled_actions[0]
                if config.dynamic_parking_fee:
                    self.op.parking_fee = rescaled_actions[1]

            elif config.dynamic_fix_term_pricing and config.capacity_pricing:
                self.op.pricing_parameters[0] = rescaled_actions[0]
                self.op.pricing_parameters[1] = rescaled_actions[1]

            if config.limiting_grid_capa:
                self.op.grid_capa = rescaled_actions[1]

            if config.dynamic_storage_scheduling:
                self.op.storage_agent.action = [rescaled_actions[1]]

            # Use storage service instead of direct call
            if hasattr(self.op, 'storage_service'):
                self.op.storage_service.conduct_storage_action(given_storage_action=[rescaled_actions[1]])
            else:
                # Fallback for backward compatibility
                self.op.conduct_storage_action(given_storage_action=[rescaled_actions[1]])

        # Reset reward at the end
        self.op.charging_hub.grid.reset_reward()

    def update_pricing_agent(self) -> None:
        self.op.update_vehicles_status()

        if not self.op.charging_agent:
            self.op.charging_hub.reward["profit"] = self.op.reward_computing()

        agent = self.op.pricing_agent
        agent_name = agent.agent_name
        # config = agent.config  # not used here directly

        if agent_name == "SAC":
            agent.conduct_action(agent.action)
            eval_ep = agent.do_evaluation_iterations

            if agent.time_for_critic_and_actor_to_learn() and not eval_ep:
                for _ in range(agent.hyperparameters["learning_updates_per_learning_session"]):
                    agent.learn()

            mask = False if agent.global_step_number >= agent.environment.MAX_EPISODE_STEPS else agent.done

            agent.save_experience(
                experience=(
                    agent.state,
                    agent.action,
                    agent.reward,
                    agent.next_state,
                    mask,
                )
            )

        elif agent_name == "DQN":
            agent.conduct_action(agent.action)

            if agent.time_for_q_network_to_learn():
                for _ in range(agent.hyperparameters["learning_iterations"]):
                    agent.learn()

            agent.save_experience(
                experience=(
                    agent.state,
                    agent.action,
                    agent.reward,
                    agent.next_state,
                    False,
                )
            )

        agent.global_step_number += 1

    def get_current_pricing_data(self):
        from dataclasses import dataclass

        @dataclass
        class PricingData:
            energy_price: float
            parking_price: float
            pricing_mode: str
            price_history: pd.DataFrame

        params = getattr(self.op, "pricing_parameters", [0.0])
        return PricingData(
            energy_price=params[0] if len(params) > 0 else 0.0,
            parking_price=self.op.parking_fee,
            pricing_mode=self.op.pricing_mode,
            price_history=self.op.price_history,
        )

    def _get_pricing_decision_via_request(self, eval_ep: bool = False) -> Any:
        """
        Get pricing decision through the decision request system.
        
        Args:
            eval_ep: Whether this is an evaluation episode
            
        Returns:
            The pricing action/decision
        """
        # Create context for the decision request
        context = {
            "eval_ep": eval_ep,
            "pricing_mode": Configuration.instance().pricing_mode,
            "agent_name": self.op.pricing_agent.agent_name,
            "charging_hub": self.op.charging_hub,
            "env": self.op.env
        }
        
        # Create and process decision request
        request_id = decision_system.create_request(
            agent_type=DecisionType.PRICING,
            state=self.op.pricing_agent.state,
            context=context,
            metadata={
                "pricing_mode": context["pricing_mode"],
                "agent_name": context["agent_name"]
            }
        )
        
        # Process the request
        response = decision_system.process_request(request_id)
        
        if response:
            return response.action
        else:
            # Fallback to direct agent call if request system fails
            return self.op.pricing_agent.pick_action(eval_ep)

    # Internal helpers (ported from Operator)
    def _update_dynamic_price_history(self) -> None:
        if self.op.pricing_mode == "Discrete":
            self._add_discrete_price_to_history()
        elif self.op.pricing_mode == "Continuous":
            self._add_continuous_price_to_history()

    def _update_static_price_history(self) -> None:
        if self.op.pricing_mode == "Discrete":
            self._add_discrete_price_to_history()
        elif self.op.pricing_mode in ["Continuous", "ToU"]:
            self._add_continuous_price_to_history()

    def _update_pricing_parameters(self) -> None:
        if self.op.pricing_mode == "ToU":
            self._update_tou_pricing()
        elif self.op.pricing_mode == "perfect_info":
            self._update_perfect_info_pricing()

    def _update_tou_pricing(self) -> None:
        hour = self._get_current_hour()
        max_price = Configuration.instance().max_price_ToU
        self.op.pricing_parameters[0] = (
            self.op.electricity_tariff[hour] / max(self.op.electricity_tariff) * max_price
        )

    def _update_perfect_info_pricing(self) -> None:
        hour = self._get_current_hour()
        config = Configuration.instance()
        if config.dynamic_fix_term_pricing:
            self.op.pricing_parameters[1] = self.op.price_schedules[1][hour]
            self.op.pricing_parameters[0] = self.op.price_schedules[0][hour]
        else:
            self.op.pricing_parameters[1] = self.op.price_schedules[hour]

    def _get_current_hour(self) -> int:
        return int((self.op.env.now % 1440) / 60)

    def _add_discrete_price_to_history(self) -> None:
        self.op.price_history = pd.concat([
            self.op.price_history,
            pd.DataFrame(self.op.price_pairs[:, 1]).transpose(),
        ])

    def _add_continuous_price_to_history(self) -> None:
        self.op.price_history = pd.concat([
            self.op.price_history,
            pd.DataFrame([
                self.op.pricing_parameters[0],
                self.op.pricing_parameters[1]
            ]).transpose(),
        ])


