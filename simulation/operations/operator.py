import simpy
from resources.configuration.configuration import Configuration
import simulation.operations.ChargingAlgorithms as charge_algos
import simulation.operations.RoutingAlgorithms as route_algos
import simulation.operations.IntegratedAlgorithms as integrate_algos
import simulation.operations.StorageAlgorithms as store_algos
from resources.logging.log import lg
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from simulation.operations.NonLinearAlgorithms import nonlinear_pricing
from simulation.operations.Operator_utils import compute_free_grid_capacity
from simulation.operations.pricing_service import PricingService
from utilities.rl_environments.rl_pricing_env import convert_to_vector


@dataclass
class GridCapacityData:
    """Data class for grid capacity information."""
    free_grid_capa_actual: List[float]
    free_grid_capa_predicted: List[float]
    free_grid_capa_without_storage: float
    base_load_list: List[float]
    generation_list: List[float]


@dataclass
class PricingData:
    """Data class for pricing information."""
    energy_price: float
    parking_price: float
    pricing_mode: str
    price_history: pd.DataFrame


@dataclass
class ChargingRequest:
    """Data class for charging request information."""
    request_id: str
    energy_requested: float
    park_duration: int
    ev: bool
    mode: Optional[str]
    charging_power: float
    assigned_charger: Optional[Any]


class Operator:
    """
    Main operator class for managing EV charging operations.
    
    Handles routing, charging, storage, and pricing decisions for electric vehicle
    charging infrastructure.
    """

    def __init__(
        self,
        env: simpy.Environment,
        requests: List[Any],
        chargers: List[Any],
        routing_strategy: str,
        charging_strategy: str,
        storage_strategy: str,
        charging_capa: float,
        grid_capa: float,
        sim_time: int,
        electricity_tariff: float,
        connector_num: int,
        parking_spots: int,
        baseload: Any,
        max_facility_baseload: float,
        non_dispatchable_generator: Any,
        electric_storage: Any,
        num_lookback_periods: int,
        planning_interval: int,
        optimization_period_length: int,
        num_lookahead_planning_periods: int,
        service_level: float,
        charging_hub: Any,
        minimum_served_demand: float,
        agents_controller: Optional[Any] = None,
    ):
        """
        Initialize the Operator with simulation parameters and strategies.
        
        Args:
            env: SimPy environment for discrete event simulation
            requests: List of charging requests
            chargers: List of charging stations
            routing_strategy: Strategy for routing vehicles to chargers
            charging_strategy: Strategy for charging optimization
            storage_strategy: Strategy for energy storage management
            charging_capa: Charging capacity limit
            grid_capa: Grid capacity limit
            sim_time: Total simulation time
            electricity_tariff: Electricity price
            connector_num: Number of connectors
            parking_spots: Number of parking spots
            baseload: Base load data
            max_facility_baseload: Maximum facility base load
            non_dispatchable_generator: Renewable energy generator
            electric_storage: Energy storage system
            num_lookback_periods: Number of historical periods for prediction
            planning_interval: Planning interval duration
            optimization_period_length: Length of optimization period
            num_lookahead_planning_periods: Number of future periods to plan
            service_level: Target service level
            charging_hub: Charging hub object
            minimum_served_demand: Minimum demand to serve
        """
        self._init_simulation_environment(env, sim_time)
        self._init_planning_parameters(
            num_lookback_periods, 
            planning_interval, 
            optimization_period_length, 
            num_lookahead_planning_periods
        )
        self._init_strategies(routing_strategy, charging_strategy, storage_strategy)
        self._init_infrastructure(
            requests, chargers, charging_capa, grid_capa, 
            connector_num, parking_spots, charging_hub
        )
        self._init_energy_systems(
            baseload, max_facility_baseload, 
            non_dispatchable_generator, electric_storage
        )
        self._init_pricing_configuration(
            electricity_tariff, service_level, minimum_served_demand
        )
        self._init_agents_and_events()
        self._init_capacity_tracking()
        # Optional RL agents controller (pricing/charging/storage)
        self.agents_controller = agents_controller
        # Pricing service composition
        self.pricing_service = PricingService(operator=self, agents_controller=self.agents_controller)
        
        # Initialize based on configuration
        self._initialize_strategy_dependent_behavior()

    def _init_simulation_environment(self, env: simpy.Environment, sim_time: int) -> None:
        """Initialize simulation environment and timing parameters."""
        if env is None:
            raise ValueError("SimPy environment cannot be None")
        if sim_time <= 0:
            raise ValueError("simulation time must be positive")
            
        self.env = env
        self.sim_time = sim_time

    def _init_planning_parameters(
        self, 
        num_lookback_periods: int,
        planning_interval: int,
        optimization_period_length: int,
        num_lookahead_planning_periods: int
    ) -> None:
        """Initialize planning and optimization parameters."""
        # Validate parameters
        if num_lookback_periods < 0:
            raise ValueError("Number of lookback periods cannot be negative")
        if planning_interval <= 0:
            raise ValueError("Planning interval must be positive")
        if optimization_period_length <= 0:
            raise ValueError("Optimization period length must be positive")
        if num_lookahead_planning_periods <= 0:
            raise ValueError("Number of lookahead planning periods must be positive")
            
        self.num_lookback_periods = num_lookback_periods
        self.planning_interval = planning_interval
        self.optimization_period_length = optimization_period_length
        self.num_lookahead_planning_periods = num_lookahead_planning_periods
        
        # Load configuration thresholds
        config = Configuration.instance()
        self.demand_threshold = config.demand_threshold
        self.duration_threshold = config.duration_threshold

    def _init_strategies(
        self, 
        routing_strategy: str, 
        charging_strategy: str, 
        storage_strategy: str
    ) -> None:
        """Initialize operational strategies."""
        self.routing_strategy = routing_strategy
        self.charging_strategy = charging_strategy
        self.storage_strategy = storage_strategy

    def _init_infrastructure(
        self,
        requests: List[Any],
        chargers: List[Any],
        charging_capa: float,
        grid_capa: float,
        connector_num: int,
        parking_spots: int,
        charging_hub: Any
    ) -> None:
        """Initialize charging infrastructure components."""
        self.requests = requests
        self.chargers = chargers
        self.charging_capa = charging_capa
        self.grid_capa = grid_capa
        self.connector_num = connector_num
        self.parking_spots = parking_spots
        self.charging_hub = charging_hub
        self.storage_object = charging_hub.electric_storage

    def _init_energy_systems(
        self,
        baseload: Any,
        max_facility_baseload: float,
        non_dispatchable_generator: Any,
        electric_storage: Any
    ) -> None:
        """Initialize energy systems and load tracking."""
        self.baseload = baseload
        self.non_dispatchable_generator = non_dispatchable_generator
        self.electric_storage = electric_storage
        
        # Initialize peak load history
        self.peak_load_history = [int(max_facility_baseload)]
        self.peak_load_history_inc_storage = [int(max_facility_baseload)]

    def _init_pricing_configuration(
        self,
        electricity_tariff: float,
        service_level: float,
        minimum_served_demand: float
    ) -> None:
        """Initialize pricing and service configuration."""
        self.electricity_tariff = electricity_tariff
        self.service_level = service_level
        self.minimum_served_demand = minimum_served_demand
        
        # Load pricing configuration
        config = Configuration.instance()
        self.peak_threshold = config.peak_threshold
        self.peak_cost = config.peak_cost
        self.B2G = config.B2G
        self.price_pairs = config.energy_prices
        self.multiple_power = config.multiple_power
        self.parking_fee = config.parking_price
        self.pricing_parameters = config.price_parameters
        self.pricing_mode = config.pricing_mode
        
        # Initialize pricing state
        self.price_history = pd.DataFrame()
        self.energy_reward = 0
        self.objective = 0
        self.generation_min = 0

    def _init_agents_and_events(self) -> None:
        """Initialize agents and simulation events."""
        self.charging_agent = None
        self.storage_agent = None
        self.pricing_agent = None
        self.agent_name = "PGMM"
        
        # Initialize simulation events
        self.arrival_event = self.env.event()
        self.routing_decision_event = self.env.event()
        self.organizer = simpy.Resource(self.env, capacity=1)

    def _init_capacity_tracking(self) -> None:
        """Initialize capacity tracking variables."""
        self.free_grid_capa_actual = 0
        self.free_grid_capa_predicted = 0
        self.free_battery_load_capa = 0
        self.free_grid_capa_without_storage = 0

    def _initialize_strategy_dependent_behavior(self) -> None:
        """Initialize behavior based on selected strategies."""
        if self.pricing_mode == "perfect_info":
            self.solve_pricing_with_perfect_info()
        
        if self.charging_strategy == "average_power":
            self._set_average_power_charging()

    def _set_average_power_charging(self) -> None:
        """Set average power charging for all requests."""
        for request in self.requests:
            request.charging_power = request.energy_requested / request.park_duration * 60

    ##########################################
    # BELOW FUNCTIONS DERIVE DECISIONS

    # have this update automatically each planning_period
    def get_exp_free_grid_capacity(self) -> GridCapacityData:
        """
        Compute and update expected free grid capacity and related time series.

        Returns:
            GridCapacityData: Structured data containing all grid capacity information
            
        Notes:
        - Reads simulation context (time, planning window) and calls the shared utility.
        - Sets instance attributes used by scheduling and pricing routines.
        - Returns structured data for better type safety.
        """
        results = compute_free_grid_capacity(self)

        # Create structured data object
        grid_data = GridCapacityData(
            free_grid_capa_actual=results.get("free_grid_capa_actual", []),
            free_grid_capa_predicted=results.get("free_grid_capa_predicted", []),
            free_grid_capa_without_storage=results.get("free_grid_capa_without_storage", 0.0),
            base_load_list=results.get("base_load_list", []),
            generation_list=results.get("generation_list", [])
        )

        # Update instance attributes for backward compatibility
        self.free_grid_capa_actual = grid_data.free_grid_capa_actual
        self.base_load_list = grid_data.base_load_list
        self.generation_list = grid_data.generation_list
        self.free_grid_capa_without_storage = grid_data.free_grid_capa_without_storage
        self.free_grid_capa_predicted = grid_data.free_grid_capa_predicted

        return grid_data



    def get_available_battery_load(self):
        """
        Updates available battery capacity for planning period as minimum of peak discharge rate and
        load that discharges SoC over planning period.
        :return:
        """
        if self.storage_object.max_energy_stored_kWh > 0:
            return
        max_remaining_discharge = (
            self.electric_storage.SoC - self.electric_storage.min_energy_stored_kWh
        )

        battery_max = min(
            self.electric_storage.kW_discharge_peak,
            (max_remaining_discharge * 60 / self.planning_interval),
        )

        self.free_battery_load_capa = battery_max

    # ============================================================================
    # ROUTING METHODS
    # ============================================================================

    def get_routing_instructions(self, request: Any) -> Any:
        """
        Route new arrivals to charging stations.
        
        This is called on a discrete event basis (per arrival) as opposed to 
        continuous optimization approaches.
        
        Args:
            request: Charging request to route
            
        Returns:
            Assigned charger for the request
        """
        return self._route_request_by_strategy(request)

    def _route_request_by_strategy(self, request: Any) -> Any:
        """Route request using the configured routing strategy."""
        if self.routing_strategy in self._ROUTING_STRATEGIES:
            return self._ROUTING_STRATEGIES[self.routing_strategy](request)
        
        # Fallback for perfect info strategies
        if self.routing_strategy in ["perfect_info", "perfect_info_with_storage"]:
            return request.assigned_charger
            
        raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

    @property
    def _ROUTING_STRATEGIES(self) -> Dict[str, callable]:
        """Get routing strategy functions."""
        return {
            "random": self._route_random,
            "lowest_occupancy_first": self._route_lowest_occupancy_first,
            "fill_one_after_other": self._route_fill_one_after_other,
            "lowest_utilization_first": self._route_lowest_utilization_first,
            "matching_supply_demand": self._route_matching_supply_demand,
            "minimum_power_requirement": self._route_minimum_power_requirement,
        }

    def _route_random(self, request: Any) -> Any:
        """Route using random charger assignment."""
        return route_algos.random_charger_assignment(
            charging_stations=self.chargers,
            number_of_connectors=self.connector_num,
            request=request,
            demand_threshold=self.demand_threshold,
            duration_threshold=self.duration_threshold,
        )

    def _route_lowest_occupancy_first(self, request: Any) -> Any:
        """Route using lowest occupancy first strategy."""
        return route_algos.lowest_occupancy_first_charger_assignment(
            charging_stations=self.chargers,
            number_of_connectors=self.connector_num,
            request=request,
            demand_threshold=self.demand_threshold,
            duration_threshold=self.duration_threshold,
        )

    def _route_fill_one_after_other(self, request: Any) -> Any:
        """Route using fill one after other strategy."""
        return route_algos.fill_one_after_other_charger_assignment(
            charging_stations=self.chargers,
            number_of_connectors=self.connector_num,
            request=request,
            demand_threshold=self.demand_threshold,
            duration_threshold=self.duration_threshold,
        )

    def _route_lowest_utilization_first(self, request: Any) -> Any:
        """Route using lowest utilization first strategy."""
        return route_algos.lowest_utilization_first_charger_assignment(
            charging_stations=self.chargers,
            number_of_connectors=self.connector_num,
            request=request,
            demand_threshold=self.demand_threshold,
            duration_threshold=self.duration_threshold,
        )

    def _route_matching_supply_demand(self, request: Any) -> Any:
        """Route using matching supply demand strategy."""
        return route_algos.matching_supply_demand_level(
            charging_stations=self.chargers,
            number_of_connectors=self.connector_num,
            request=request,
            demand_threshold=self.demand_threshold,
            duration_threshold=self.duration_threshold,
        )

    def _route_minimum_power_requirement(self, request: Any) -> Any:
        """Route using minimum power requirement strategy."""
        return route_algos.assign_to_the_minimum_power(
            charging_stations=self.chargers,
            number_of_connectors=self.connector_num,
            request=request,
            demand_threshold=self.demand_threshold,
            duration_threshold=self.duration_threshold,
        )

    # ============================================================================
    # CHARGING METHODS
    # ============================================================================

    def schedule_charging_and_routing_perfect_info(self, strategy: str) -> None:
        """
        Schedule charging and routing using perfect information strategies.
        
        Args:
            strategy: Charging strategy to use ("perfect_info" or "perfect_info_with_storage")
        """
        self.get_exp_free_grid_capacity()
        connected_vehicles = self._get_connected_vehicles()

        if strategy == "perfect_info":
            self._schedule_perfect_info_charging(connected_vehicles)
        elif strategy == "perfect_info_with_storage" and connected_vehicles:
            self._schedule_perfect_info_charging_with_storage(connected_vehicles)

    def _get_connected_vehicles(self) -> List[Any]:
        """Get list of connected vehicles that are EVs and not yet assigned a mode."""
        return [x for x in self.requests if x.mode is None and x.ev == 1]

    def _schedule_perfect_info_charging(self, connected_vehicles: List[Any]) -> None:
        """Schedule charging using perfect information without storage."""
        integrate_algos.perfect_info_charging_routing(
            vehicles=connected_vehicles,
            charging_stations=self.chargers,
            env=self.env,
            grid_capacity=self.free_grid_capa_actual,
            electricity_cost=self.electricity_tariff,
            baseload=self.base_load_list,
            sim_time=self.sim_time,
            generation=self.generation_list,
        )

    def _schedule_perfect_info_charging_with_storage(self, connected_vehicles: List[Any]) -> None:
        """Schedule charging using perfect information with storage integration."""
        integrate_algos.perfect_info_charging_routing_storage(
            vehicles=connected_vehicles,
            charging_stations=self.chargers,
            env=self.env,
            grid_capacity=self.free_grid_capa_actual,
            electricity_cost=self.electricity_tariff,
            sim_time=self.sim_time,
            storage=self.electric_storage,
            baseload=self.base_load_list,
        )

    def apply_charging_routing_storage_perfect_info(self, charging_strategy: str) -> None:
        """
        Apply charging and storage schedules based on perfect information.
        
        Args:
            charging_strategy: Strategy to use for charging and storage
        """
        hour = self._get_current_hour(charging_strategy)
        self._apply_vehicle_charging_schedules(hour)
        
        if self._should_apply_storage_schedule(charging_strategy):
            self._apply_storage_schedule(hour)

    def _get_current_hour(self, charging_strategy: str) -> int:
        """Get current hour based on charging strategy."""
        if charging_strategy == "perfect_info_with_storage":
            return int((self.env.now % 1440) / 60)
        return int(self.env.now / 60)

    def _apply_vehicle_charging_schedules(self, hour: int) -> None:
        """Apply charging schedules to all EV requests."""
        for request in self.requests:
            if request.ev == 1:
                request.charging_power = request.charge_schedule[hour]

    def _should_apply_storage_schedule(self, charging_strategy: str) -> bool:
        """Check if storage schedule should be applied."""
        return (charging_strategy == "perfect_info_with_storage" and 
                self.storage_object.max_capacity_kWh > 0)

    def _apply_storage_schedule(self, hour: int) -> None:
        """Apply storage charging/discharging schedule."""
        storage_power = self.electric_storage.charge_schedule[hour]
        
        if storage_power >= 0:
            self._set_storage_charging(storage_power)
        else:
            self._set_storage_discharging(storage_power)

    def _set_storage_charging(self, power: float) -> None:
        """Set storage to charging mode."""
        self.electric_storage.charge_yn = 1
        self.electric_storage.discharge_yn = 0
        self.electric_storage.discharging_power = 0
        self.electric_storage.charging_power = power

    def _set_storage_discharging(self, power: float) -> None:
        """Set storage to discharging mode."""
        self.electric_storage.charge_yn = 0
        self.electric_storage.discharge_yn = 1
        self.electric_storage.discharging_power = power
        self.electric_storage.charging_power = 0

    # ============================================================================
    # PRICING METHODS
    # ============================================================================

    def take_dynamic_pricing_actions(self) -> None:
        """Execute dynamic pricing actions and update price history."""
        self.pricing_service.take_dynamic_pricing_actions()

    def take_static_pricing_action(self) -> None:
        """Execute static pricing actions and update price history."""
        self.pricing_service.take_static_pricing_action()

    def _update_dynamic_price_history(self) -> None:
        # Backward-compat shim; delegate to service
        self.pricing_service._update_dynamic_price_history()

    def _update_static_price_history(self) -> None:
        self.pricing_service._update_static_price_history()

    def _update_pricing_parameters(self) -> None:
        self.pricing_service._update_pricing_parameters()

    def _update_tou_pricing(self) -> None:
        self.pricing_service._update_tou_pricing()

    def _update_perfect_info_pricing(self) -> None:
        self.pricing_service._update_perfect_info_pricing()

    def _get_current_hour(self) -> int:
        return self.pricing_service._get_current_hour()

    def _add_discrete_price_to_history(self) -> None:
        self.pricing_service._add_discrete_price_to_history()

    def _add_continuous_price_to_history(self) -> None:
        self.pricing_service._add_continuous_price_to_history()

    def get_current_pricing_data(self) -> PricingData:
        return self.pricing_service.get_current_pricing_data()

    # ============================================================================
    # CHARGING ACTION METHODS
    # ============================================================================

    def take_non_learning_charging_actions(self, charging_strategy: str, connected_vehicles: List[Any]) -> None:
        """
        Execute non-learning charging actions based on strategy.
        
        Args:
            charging_strategy: Strategy to use for charging
            connected_vehicles: List of connected vehicles
        """
        self._execute_basic_charging_strategies(charging_strategy, connected_vehicles)
        self._execute_advanced_charging_strategies(charging_strategy, connected_vehicles)

    def _execute_basic_charging_strategies(self, charging_strategy: str, connected_vehicles: List[Any]) -> None:
        """Execute basic charging strategies that don't require foresight."""
        basic_strategies = {
            "uncontrolled": self._execute_uncontrolled_charging,
            "average_power": self._execute_average_power_charging,
            "first_come_first_served": self._execute_fcfs_charging,
            "earliest_deadline_first": self._execute_edf_charging,
            "least_laxity_first": self._execute_llf_charging,
            "equal_sharing": self._execute_equal_sharing_charging,
            "online_myopic": self._execute_online_myopic_charging,
        }
        
        if charging_strategy in basic_strategies:
            basic_strategies[charging_strategy](connected_vehicles)

    def _execute_advanced_charging_strategies(self, charging_strategy: str, connected_vehicles: List[Any]) -> None:
        """Execute advanced charging strategies that require foresight."""
        if charging_strategy == "online_multi_period":
            self._execute_online_multi_period_charging(connected_vehicles)
        elif charging_strategy == "integrated_storage":
            self._execute_integrated_storage_charging(connected_vehicles)

    def _execute_uncontrolled_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute uncontrolled charging strategy."""
        charge_algos.uncontrolled(
            env=self.env,
            connected_vehicles=connected_vehicles,
            charging_stations=self.chargers,
            charging_capacity=self.charging_capa,
            free_grid_capacity=self.free_grid_capa_actual,
            planning_period_length=self.planning_interval,
        )

    def _execute_average_power_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute average power charging strategy."""
        charge_algos.average_power(
            env=self.env,
            connected_vehicles=connected_vehicles,
            charging_stations=self.chargers,
            charging_capacity=self.charging_capa,
            free_grid_capacity=self.free_grid_capa_actual,
            planning_period_length=self.planning_interval,
        )

    def _execute_fcfs_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute first-come-first-served charging strategy."""
        charge_algos.first_come_first_served(
            env=self.env,
            connected_vehicles=connected_vehicles,
            charging_stations=self.chargers,
            charging_capacity=self.charging_capa,
            free_grid_capacity=self.free_grid_capa_actual,
            planning_period_length=self.planning_interval,
        )

    def _execute_edf_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute earliest deadline first charging strategy."""
        charge_algos.earliest_deadline_first(
            env=self.env,
            connected_vehicles=connected_vehicles,
            charging_stations=self.chargers,
            charging_capacity=self.charging_capa,
            free_grid_capacity=self.free_grid_capa_actual,
            planning_period_length=self.planning_interval,
        )

    def _execute_llf_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute least laxity first charging strategy."""
        charge_algos.least_laxity_first(
            env=self.env,
            connected_vehicles=connected_vehicles,
            charging_stations=self.chargers,
            charging_capacity=self.charging_capa,
            free_grid_capacity=self.free_grid_capa_actual,
            planning_period_length=self.planning_interval,
        )

    def _execute_equal_sharing_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute equal sharing charging strategy."""
        charge_algos.equal_sharing(
            env=self.env,
            connected_vehicles=connected_vehicles,
            charging_stations=self.chargers,
            charging_capacity=self.charging_capa,
            free_grid_capacity=self.free_grid_capa_actual,
            planning_period_length=self.planning_interval,
        )

    def _execute_online_myopic_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute online myopic charging strategy."""
        charge_algos.online_myopic(
            vehicles=connected_vehicles,
            charging_stations=self.chargers,
            env=self.env,
            grid_capacity=self.free_grid_capa_actual,
            optimization_period_length=self.optimization_period_length,
            alpha=0,
        )

    def _execute_online_multi_period_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute online multi-period charging strategy."""
        self._update_peak_threshold()
        
        if connected_vehicles:
            charge_algos.online_multi_period(
                vehicles=connected_vehicles,
                charging_stations=self.chargers,
                env=self.env,
                free_grid_capa_actual=self.free_grid_capa_actual,
                free_grid_capa_predicted=self.free_grid_capa_predicted,
                peak_load_history=self.peak_load_history,
                electricity_cost=self.electricity_tariff,
                sim_time=self.sim_time,
                service_level=self.service_level,
                optimization_period_length=self.optimization_period_length,
                num_lookahead_planning_periods=4,
                flex_margin=0.5,
                peak_threshold=self.peak_threshold,
            )

    def _execute_integrated_storage_charging(self, connected_vehicles: List[Any]) -> None:
        """Execute integrated storage charging strategy."""
        if connected_vehicles:
            charge_algos.integrated_charging_storage(
                storage=self.electric_storage,
                vehicles=connected_vehicles,
                charging_stations=self.chargers,
                env=self.env,
                free_grid_capa_actual=self.free_grid_capa_actual,
                free_grid_capa_predicted=self.free_grid_capa_predicted,
                peak_load_history=self.peak_load_history,
                electricity_cost=self.electricity_tariff,
                sim_time=self.sim_time,
                service_level=self.service_level,
                optimization_period_length=self.optimization_period_length,
                num_lookahead_planning_periods=12,
                flex_margin=0.5,
            )

    def _update_peak_threshold(self) -> None:
        """Update peak threshold based on current grid usage."""
        current_peak = max(self.charging_hub.grid.grid_usage)
        if current_peak > self.peak_threshold:
            self.peak_threshold = current_peak
    def take_learning_charging_actions(self, charging_strategy):
        if charging_strategy == "dynamic":
            self.update_vehicles_status()
            self.take_charging_action()
            self.conduct_charging_action()

            if self.storage_agent:
                self.get_exp_free_grid_capacity()
                self.take_storage_action()
                self.conduct_storage_action()

    def update_learning_charging_and_pricing_agents(self, charging_strategy):
        if charging_strategy == "dynamic":
            self.update_charging_agent()
            if self.storage_agent:
                self.update_storage_agent()
        if self.charging_hub.dynamic_pricing:
            self.pricing_service.update_pricing_agent()

    def get_charging_schedules_and_prices(self, charging_strategy, mode):
        """
        Periodically updates charging schedule based on selected strategy. Decides which vehicle charges and how much!
        This is on a discrete time basis
        :param scheduling_mode: simulation mode (discrete-time or discrete-event)
        :param charging_strategy:
        :param planning_period_length: length of period (in unit sim time). Schedule is re-computed every n(=period_length) time steps
        :return: n/a
        """
        # first_scheduling = False
        if charging_strategy in ["perfect_info", "perfect_info_with_storage"]:
            self.schedule_charging_and_routing_perfect_info(charging_strategy)
        while True:
            self.get_exp_free_grid_capacity()
            self.get_available_battery_load()
            connected_vehicles = [x for x in self.requests if x.mode == "Connected"]
            if charging_strategy in ["perfect_info", "perfect_info_with_storage"]:
                self.apply_charging_routing_storage_perfect_info(charging_strategy)

            if self.charging_hub.dynamic_pricing:
                self.take_dynamic_pricing_actions()

            else:
                self.take_static_pricing_action()

            self.take_non_learning_charging_actions(charging_strategy, connected_vehicles)
            self.take_learning_charging_actions(connected_vehicles)

            # update peak load history
            self.update_peak_load_history()

            # yield until next planning period
            if mode == "discrete_time":
                yield self.env.timeout(self.planning_interval)
            if mode == "discrete_event":
                yield self.arrival_event

            self.update_learning_charging_and_pricing_agents(charging_strategy)

    def take_charging_action(self):
        state = self.charging_hub.charging_agent.environment.get_state(
            self.charging_hub, self.env
        )
        self.charging_agent.state = state

        eval_ep = self.charging_agent.do_evaluation_iterations
        self.charging_agent.episode_step_number_val = 0
        # while not self.done:
        action = self.charging_agent.pick_action(eval_ep, self.charging_hub)
        self.charging_agent.action = self.charging_agent.rescale_action(action)

    def take_pricing_action(self):
        # Get current state from environment
        pricing_state = self.pricing_agent.environment.get_state(self.charging_hub, self.env)
        self.pricing_agent.state = pricing_state
        eval_ep = self.pricing_agent.do_evaluation_iterations

        pricing_mode = Configuration.instance().pricing_mode
        agent_name = self.pricing_agent.agent_name

        if pricing_mode == "Discrete":
            if agent_name == "DQN":
                self.pricing_agent.action = self.pricing_agent.pick_action()
                if len(self.price_pairs[:, 1]) > 1:
                    vector_prices = convert_to_vector(self.pricing_agent.action)
                else:
                    vector_prices = [self.pricing_agent.action]
                final_pricing = self.pricing_agent.environment.get_final_prices_DQN(vector_prices)
                for i, price in enumerate(final_pricing):
                    self.price_pairs[i, 1] = price

            elif agent_name == "SAC":
                self.pricing_agent.action = self.pricing_agent.pick_action(eval_ep, self.charging_hub)
                rescaled_actions = self.pricing_agent.environment.rescale_action(self.pricing_agent.action)
                number_of_power_options = len(self.price_pairs[:, 1])
                final_pricing = rescaled_actions[:number_of_power_options]
                self.price_pairs[0, 1] = final_pricing[0]
                self.price_pairs[1, 1] = min(final_pricing[1], 1.5)

                # Optional: handle grid capacity and storage
                # if Configuration.instance().limiting_grid_capa:
                #     self.grid_capa = rescaled_actions[number_of_power_options]
                # if len(rescaled_actions) >= number_of_power_options + 2:
                #     self.storage_agent.action = [rescaled_actions[number_of_power_options + 1]]
                #     self.conduct_storage_action()

        elif pricing_mode == "Continuous":
            self.pricing_agent.action = self.pricing_agent.pick_action(eval_ep, self.charging_hub)
            rescaled_actions = self.pricing_agent.environment.rescale_action(self.pricing_agent.action)

            config = Configuration.instance()
            if not config.dynamic_fix_term_pricing and config.capacity_pricing:
                self.pricing_parameters[1] = rescaled_actions[0]

            elif config.dynamic_fix_term_pricing and not config.capacity_pricing:
                self.pricing_parameters[0] = rescaled_actions[0]
                if config.dynamic_parking_fee:
                    self.parking_fee = rescaled_actions[1]

            elif config.dynamic_fix_term_pricing and config.capacity_pricing:
                self.pricing_parameters[0] = rescaled_actions[0]
                self.pricing_parameters[1] = rescaled_actions[1]

            if config.limiting_grid_capa:
                self.grid_capa = rescaled_actions[1]

            if config.dynamic_storage_scheduling:
                self.storage_agent.action = [rescaled_actions[1]]

            self.conduct_storage_action(given_storage_action=[rescaled_actions[1]])

        # Reset reward at the end
        self.charging_hub.grid.reset_reward() #TODO: it does not belong to the grid object

    def take_storage_action(self):
        storage_state = self.charging_hub.storage_agent.environment.get_state(
            self.charging_hub, self.env
        )
        self.storage_agent.state = storage_state

        eval_ep = self.storage_agent.do_evaluation_iterations
        self.storage_agent.episode_step_number_val = 0
        # while not self.done:
        self.storage_agent.action = self.storage_agent.pick_action(
            eval_ep, self.charging_hub
        )

    def get_battery_max_min(self):
        bound_1 = (
            (self.storage_object.max_energy_stored_kWh - self.storage_object.SoC)
            * 60
            / self.charging_hub.planning_interval
        )
        bound_2 = self.charging_hub.operator.free_grid_capa_actual[0]
        bound_3 = self.electric_storage.kW_charge_peak
        charging_bound = min(bound_1, bound_2, bound_3)
        self.charging_hub.max_battery_charging_rate = charging_bound
        hub_generation_kW, hub_demand_kW, max_grid_capa = (
            self.get_hub_generation_kW(),
            self.get_hub_load_kW(),
            self.grid_capa,
        )
        bound_1 = -(hub_demand_kW - hub_generation_kW)
        bound_2 = -(self.storage_object.SoC) * 60 / self.charging_hub.planning_interval
        bound_3 = -self.electric_storage.kW_discharge_peak
        if not self.B2G:
            discharging_bound = max(bound_2, bound_3)
        else:
            discharging_bound = max(bound_1, bound_2, bound_3)
        self.charging_hub.max_battery_discharging_rate = discharging_bound
        self.storage_agent.action_range = [discharging_bound, charging_bound]

    def check_storage(self, given_storage_action=None):
        if given_storage_action:
            storage_power = given_storage_action[0]
        else:
            storage_power = self.storage_agent.action[0]
        if storage_power >= 0:
            if (
                self.storage_object.SoC
                + storage_power / 60 * self.charging_hub.planning_interval
                > self.storage_object.max_energy_stored_kWh
            ):
                storage_power = (
                    (
                        self.storage_object.max_energy_stored_kWh
                        - self.storage_object.SoC
                    )
                    * 60
                    / self.charging_hub.planning_interval
                )
            storage_power = min(
                storage_power, self.charging_hub.operator.free_grid_capa_without_storage
            )

            self.charging_hub.electric_storage.charge_yn = 1
            self.charging_hub.electric_storage.charging_power = storage_power
            self.charging_hub.electric_storage.discharge_yn = 0
            self.charging_hub.electric_storage.discharging_power = 0
        hub_generation_kW, hub_demand_kW, max_grid_capa = (
            self.get_hub_generation_kW(),
            self.get_hub_load_kW(),
            self.grid_capa,
        )
        if storage_power < 0:
            if not self.B2G:
                if storage_power + hub_demand_kW - hub_generation_kW < 0:
                    storage_power = -(hub_demand_kW - hub_generation_kW)
            if (
                self.storage_object.SoC
                + (storage_power / 60 * self.charging_hub.planning_interval)
                < 0
            ):
                storage_power = -max(
                    (self.storage_object.SoC)
                    * 60
                    / self.charging_hub.planning_interval,
                    0,
                )
            if self.storage_object.SoC <= 0:
                storage_power = 0

            self.charging_hub.electric_storage.charge_yn = 0
            self.charging_hub.electric_storage.charging_power = 0
            self.charging_hub.electric_storage.discharge_yn = 1
            self.charging_hub.electric_storage.discharging_power = -storage_power
        if given_storage_action:
            raw_storage_power = given_storage_action[0]
        else:
            raw_storage_power = self.storage_agent.action[0]

        self.charging_hub.reward["feasibility_storage"] += abs(
             raw_storage_power - storage_power
        )

    def check_storage(self, given_storage_action=None):
        # Determine the raw storage power action
        raw_storage_power = (
            given_storage_action[0] if given_storage_action else self.storage_agent.action[0]
        )
        storage_power = raw_storage_power

        interval_factor = self.charging_hub.planning_interval / 60
        soc = self.storage_object.SoC
        max_energy = self.storage_object.max_energy_stored_kWh
        grid_capacity = self.charging_hub.operator.free_grid_capa_without_storage

        # Handle charging
        if storage_power >= 0:
            projected_soc = soc + storage_power * interval_factor
            if projected_soc > max_energy:
                storage_power = (max_energy - soc) / interval_factor
            storage_power = min(storage_power, grid_capacity)

            self.charging_hub.electric_storage.charge_yn = 1
            self.charging_hub.electric_storage.charging_power = storage_power
            self.charging_hub.electric_storage.discharge_yn = 0
            self.charging_hub.electric_storage.discharging_power = 0

        # Handle discharging
        else:
            hub_generation_kW = self.get_hub_generation_kW()
            hub_demand_kW = self.get_hub_load_kW()

            if not self.B2G and (storage_power + hub_demand_kW - hub_generation_kW < 0):
                storage_power = -(hub_demand_kW - hub_generation_kW)

            projected_soc = soc + storage_power * interval_factor
            if projected_soc < 0:
                storage_power = -max(soc / interval_factor, 0)

            if soc <= 0:
                storage_power = 0

            self.charging_hub.electric_storage.charge_yn = 0
            self.charging_hub.electric_storage.charging_power = 0
            self.charging_hub.electric_storage.discharge_yn = 1
            self.charging_hub.electric_storage.discharging_power = -storage_power

        # Track feasibility deviation
        self.charging_hub.reward["feasibility_storage"] += abs(raw_storage_power - storage_power)

    def conduct_storage_action(self, given_storage_action=None):
        if given_storage_action:
            storage_power = given_storage_action[0]
        else:
            storage_power = self.storage_agent.action[0]
        if storage_power >= 0:
            self.charging_hub.electric_storage.charge_yn = 1
            self.charging_hub.electric_storage.charging_power = storage_power
            self.charging_hub.electric_storage.discharge_yn = 0
            self.charging_hub.electric_storage.discharging_power = 0
        elif storage_power < 0:
            self.charging_hub.electric_storage.charge_yn = 0
            self.charging_hub.electric_storage.charging_power = 0
            self.charging_hub.electric_storage.discharge_yn = 1
            self.charging_hub.electric_storage.discharging_power = -storage_power
        self.check_storage(given_storage_action=given_storage_action)

    def conduct_charging_action(self):
        action = self.charging_agent.action
        action_index = 1  # Start from 1 because action[0] is reserved (possibly for pricing or metadata)

        for charger in self.charging_hub.chargers:
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

        self.check_charging_power()
        self.charging_hub.grid.reset_reward()

    def update_pricing_agent(self):
        # Delegated to PricingService for backward compatibility
        self.pricing_service.update_pricing_agent()

    def update_storage_agent(self):

        eval_ep = self.storage_agent.do_evaluation_iterations
        action = self.storage_agent.descale_action(
            self.storage_agent.action, self.charging_hub
        )
        self.storage_agent.conduct_action(action, self.charging_hub, self.env, eval_ep=eval_ep)
        if self.storage_agent.time_for_critic_and_actor_to_learn():
            for _ in range(
                self.storage_agent.hyperparameters[
                    "learning_updates_per_learning_session"
                ]
            ):
                self.storage_agent.learn()
        mask = (
            False
            if self.storage_agent.episode_step_number_val
            >= self.storage_agent.environment._max_episode_steps
            else self.storage_agent.done
        )
        # if not eval_ep:

        self.storage_agent.save_experience(
            experience=(
                self.storage_agent.state,
                action,
                self.storage_agent.reward,
                self.storage_agent.next_state,
                mask,
            )
        )
        self.storage_agent.global_step_number += 1
        self.storage_agent.step_counter += 1

    def update_charging_agent(self):
        self.update_vehicles_status()
        self.charging_hub.reward["missed"] = self.reward_computing()

        eval_ep = self.charging_agent.do_evaluation_iterations
        self.charging_agent.conduct_action(self.charging_agent.action, self.charging_hub, self.env)
        if self.charging_agent.time_for_critic_and_actor_to_learn():
            if not eval_ep:
                for _ in range(
                    self.charging_agent.hyperparameters[
                        "learning_updates_per_learning_session"
                    ]
                ):
                    self.charging_agent.learn()
        mask = (
            False
            if self.charging_agent.episode_step_number_val
            >= self.charging_agent.environment._max_episode_steps
            else self.charging_agent.done
        )
        # if not eval_ep:
        action = self.charging_agent.descale_action(self.charging_agent.action, self.charging_hub)
        self.charging_agent.save_experience(
            experience=(
                self.charging_agent.state,
                action,
                self.charging_agent.reward,
                self.charging_agent.next_state,
                mask,
            )
        )
        self.charging_agent.global_step_number += 1
        self.charging_agent.step_counter += 1

    def get_storage_schedule(self, storage_strategy, mode):
        """
        Get schedule for battery ops (receives charge schedules, PV schedules and tariff info, etc., optimizes accordingly)
        :return:
        """

        # if storage_strategy == 'testing':
        while True:
            # get charging load
            t = self.env.now
            ev_charging_load = sum(
                [x.charging_power for x in self.requests if x.mode == "Connected"]
            )  # previously defined by charging algo
            # Here we use actuals but since is highly predictable it should be fine
            max_base_load = max(
                self.baseload.loc[t : t + self.planning_interval - 1][
                    "load_kw_rescaled"
                ]
            )
            min_PV_generation = min(
                self.non_dispatchable_generator.generation_profile_actual.loc[
                    t : t + self.planning_interval - 1
                ]["pv_generation"]
            )

            if storage_strategy == "uncontrolled":
                store_algos.uncontrolled(
                    env=self.env, storage_object=self.electric_storage
                )

            if storage_strategy == "temporal_arbitrage":
                store_algos.temporal_arbitrage(
                    env=self.env,
                    storage_object=self.electric_storage,
                    planning_interval=self.planning_interval,
                    electricity_tariff=self.electricity_tariff,
                    free_grid_capacity=self.free_grid_capa_without_storage,
                    ev_charging_load=ev_charging_load,
                )

            if storage_strategy == "peak_shaving":
                store_algos.peak_shaving(
                    env=self.env,
                    storage_object=self.electric_storage,
                    planning_interval=self.planning_interval,
                    free_grid_capacity=self.free_grid_capa_actual,
                    ev_charging_load=ev_charging_load,
                    max_base_load=max_base_load,
                    min_PV_generation=min_PV_generation,
                    peak_history_inc_storage=self.peak_load_history_inc_storage,
                )

            # update load history
            self.update_peak_load_history_inc_storage()

            # yield until next planning period
            if mode == "discrete_time":
                yield self.env.timeout(self.planning_interval)
            if mode == "discrete_event":
                yield self.arrival_event

    def update_peak_load_history(self):
        """
        Calculates peak load in planning period and appends to peak_load history
        :param self:
        :return:
        """
        # LOAD SOURCES AND SINKS

        t = self.env.now
        charging_load = sum(
            [x.charging_power for x in self.requests if x.mode == "Connected"]
        )
        baseload_max = max(
            self.baseload.loc[t : t + self.planning_interval - 1]["load_kw_rescaled"]
        )
        generation_min = min(
            self.non_dispatchable_generator.generation_profile_actual.loc[
                t : t + self.planning_interval - 1
            ]["pv_generation"]
        )

        planning_window_peak_load = charging_load + baseload_max - generation_min
        self.peak_load_history.append(planning_window_peak_load)

    def update_peak_load_history_inc_storage(self):
        """
        Calculates peak load in planning period and appends to peak_load history
        :param self:
        :return:
        """
        # LOAD SOURCES AND SINKS

        t = self.env.now
        charging_load = sum(
            [x.charging_power for x in self.requests if x.mode == "Connected"]
        )
        baseload_max = max(
            self.baseload.loc[t : t + self.planning_interval - 1]["load_kw_rescaled"]
        )
        generation_min = min(
            self.non_dispatchable_generator.generation_profile_actual.loc[
                t : t + self.planning_interval - 1
            ]["pv_generation"]
        )
        battery_charge = (
            self.electric_storage.charge_yn * self.electric_storage.charging_power
        )
        battery_discharge = (
            self.electric_storage.discharge_yn * self.electric_storage.discharging_power
        )

        planning_window_peak_load = (
            charging_load
            + baseload_max
            + battery_charge
            - generation_min
            - battery_discharge
        )
        self.peak_load_history_inc_storage.append(planning_window_peak_load)

        # print("Peak load planning window (post charging)",planning_window_peak_load)

    def get_hub_load_kW(self):
        """
        Retrieves total load in current period
        :param self:
        :return:
        """
        # TODO: GET FORECASTS FOR t+n

        t = self.env.now
        charging_load = sum(
            [x.charging_power for x in self.requests if x.mode == "Connected"]
        )
        baseload = self.baseload.loc[t]["load_kw_rescaled"]

        return charging_load + baseload

    def get_hub_generation_kW(self):
        """
        Retrieves total generation (i.e., PV) supply in current period
        :param self:
        :return:
        """
        # TODO: GET FORECASTS FOR t+n

        t = self.env.now

        generation_current_period = (
            self.non_dispatchable_generator.generation_profile_actual.loc[t][
                "pv_generation"
            ]
        )

        return generation_current_period

    ##########################################
    # BELOW FUNCTIONS EXECUTE DECISIONS

    def request_queueing(self):
        while True:
            not_arrived_requests = [x for x in self.requests if x.mode is None]
            if len(not_arrived_requests) > 0:
                request = not_arrived_requests[0]
                interarrival_time = request.arrival_period - self.env.now
                yield self.env.timeout(interarrival_time)
                requests = [
                    x for x in not_arrived_requests if x.arrival_period <= self.env.now
                ]
                for request in requests:
                    request.mode = "Arrived"
                    if self.multiple_power:
                        request.adjust_request_demand_based_on_pricing(
                            self.price_pairs, self.pricing_parameters, self.parking_fee
                        )
                    self.env.process(self.assign_parking_charging_resources(request))
                    for charger in self.charging_hub.chargers:
                        charger.status_update()
                    yield self.env.timeout(0.01)
                    self.env.process(self.request_process(request))
            else:
                return

    # SERVING A REQUEST --> We need to talk about this section as I am not quite sure about it.
    def assign_parking_charging_resources(self, request):
        """
        Executing the process of charging for EVs and parking for Non-EVs
        :param duration_threshold:
        :param demand_threshold:
        :param request: object of request
        """
        lg.info(f"Request {request.id} arrived at {self.env.now}"
                , extra={"clazz": self.__class__.__name__, "oid": ""})
        request.mode = "Arrived"

        # get charger for request
        charging_station = self.get_routing_instructions(request=request)

        if charging_station:
            with charging_station.connectors.request() as charging_req:
                charging_station.in_queue_vehicles.append(request)
                yield charging_req
                charging_station.in_queue_vehicles.remove(request)
                with self.parking_spots.request() as parking_req:
                    yield parking_req
                    charging_station.connected_vehicles.append(request)
                    # charging_req = charging_station.connectors.request()
                    # parking_req = self.parking_spots.request()
                    # yield charging_req and parking_req
                    request.assigned_charger = charging_station
                    request.is_assigned = True
                    request.mode = "Assigned"
                    lg.info(
                        f"Request {request.id} (EV={request.ev}; "
                        f"requested charge = {request.energy_requested} kW) assigned to charging station "
                        f"{request.assigned_charger.id}"
                        , extra={"clazz": self.__class__.__name__, "oid": ""}
                    )
                    self.arrival_event.succeed()
                    self.arrival_event = self.env.event()
                    request.mode = "Connected"
                    request.assigned_time = self.env.now
                    request.waiting_time = (
                        request.assigned_time - request.arrival_period
                    )
                    lg.info(
                        f"Request {request.id} (EV={request.ev}; "
                        f"requested charge = {request.energy_requested} kW) got connected to charging station "
                        f"{request.assigned_charger.id}"
                        , extra={"clazz": self.__class__.__name__, "oid": ""}
                    )
                    yield request.event_departure
                    # charging_station.connectors.release(charging_req)
                    # charging_req.cancel()
                    # self.parking_spots.release(parking_req)
                    # parking_req.cancel()
                    request.mode = "Left"
                    # request.assigned_charger = None
                    lg.info(
                        f"Request {request.id} got {request.energy_charged} with requested energy"
                        f" {request.energy_requested}", extra={"clazz": self.__class__.__name__, "oid": ""}
                    )
                    charging_station.connected_vehicles.remove(request)

        else:
            request.is_assigned = False
            # lg.info('No charger assigned charger')

            with self.parking_spots.request() as req:
                yield req
                # lg.info(f'Request {request.id} starts parking')
                request.mode = "Parking"
                yield request.event_departure
                request.mode = "Left"

    def storage_process(self):
        """
        Executing the charging and discharging schedule of the storage
        :param request: object of request
        """
        while True:
            hub_generation_kW, hub_demand_kW, max_grid_capa = (
                self.get_hub_generation_kW(),
                self.get_hub_load_kW(),
                self.grid_capa,
            )

            self.electric_storage.deploy(
                B2G=self.B2G,
                hub_demand_kW=hub_demand_kW,
                hub_generation_kW=hub_generation_kW,
                max_grid_capa=max_grid_capa,
            )
            # print(self.electric_storage.charging_power, self.electric_storage.discharging_power, self.electric_storage.SoC)/
            yield self.env.timeout(1)

    # TODO: Physical execution of above decisions to be added to infrastrcuture objects
    def request_process(self, request):
        """
        Tracking the events of arrival, stop-charging and departure of each request (i.e., each vehicle)
        :param request: object of request
        """
        while True:
            # if request.arrival_period == self.env.now:
            # request.event_arrival.succeed()
            # request.event_arrival = self.env.event()
            # self.env.process(self.charging_parking_task(request))
            if Configuration.instance().remove_low_request_EVs:
                if request.energy_requested == 0:
                    request.mode = "Left"
                    request.event_departure.succeed()
                    return
            if request.ev == 1:
                if request.mode == "Connected":
                    # print(f'{request.id} is charging with power {request.charging_power}')
                    request.energy_charged += (
                        request.charging_power / 60
                    )  # sim unit time is minutes so need to divide by 60
                    request.calculate_profit_reward(
                        self.charging_hub.penalty_for_missed_kWh,
                        self.electricity_tariff,
                    )
                    if request.charging_power < 0:
                        lg.warning(
                            f"charging power of {request.id} is negative{request.charging_power}"
                            , extra={"clazz": self.__class__.__name__, "oid": ""}
                        )
                if (
                    request.mode == "Connected"
                    and request.energy_charged >= request.energy_requested
                ):
                    request.event_stop_charging.succeed()
                    request.event_stop_charging = self.env.event()
                    request.stop_charging_time = self.env.now
                    request.mode = "Fully_charged"
                    request.charging_power = 0
                    lg.info(f"Request {request.id} stopped charging at {self.env.now}"
                            , extra={"clazz": self.__class__.__name__, "oid": ""})
                    lg.info(
                        f"Request {request.id} got {request.energy_charged} with requested energy"
                        f" {request.energy_requested}"
                        , extra={"clazz": self.__class__.__name__, "oid": ""}
                    )
                    if request.energy_charged < 0:
                        lg.info(f"request.energy_charged is negative for {request.id}"
                                , extra={"clazz": self.__class__.__name__, "oid": ""})
            if request.departure_period <= self.env.now:
                lg.info(f"Request {request.id} left at {self.env.now}"
                        , extra={"clazz": self.__class__.__name__, "oid": ""})
                # self.charging_hub.reward['missed'] += self.request_reward_computing(request)
                if request.energy_charged < 0:
                    lg.info(f"request.energy_charged is negative for {request.id}"
                            , extra={"clazz": self.__class__.__name__, "oid": ""})
                request.event_departure.succeed()
                return
            elif self.env.now == self.sim_time - 1:
                lg.info(f"Request {request.id} left at {self.env.now}"
                        , extra={"clazz": self.__class__.__name__, "oid": ""})
                # self.charging_hub.reward['missed'] += self.request_reward_computing(request)
                if request.energy_charged < 0:
                    lg.info(f"request.energy_charged is negative for {request.id}"
                            , extra={"clazz": self.__class__.__name__, "oid": ""})
                request.mode = "Left"
                request.event_departure.succeed()
                return
            yield self.env.timeout(1)

    ############################################################################
    # MONITOR EV CHARGING
    def request_monitoring(self, request):
        """
        Monitoring the state of charge and the mode of each request every time step
        """
        request.info["SOC"] = []
        request.info["mode"] = []
        request.info["charging_power"] = []
        while True:
            request.info["SOC"].append(request.energy_charged)
            request.info["mode"].append(request.mode)
            request.info["charging_power"].append(request.charging_power)
            yield self.env.timeout(1)

    def reward_computing(self):
        reward = 0

        if max(self.charging_hub.grid.grid_usage) > self.peak_threshold:
            reward += (
                max(self.charging_hub.grid.grid_usage) - self.peak_threshold
            ) * self.peak_cost
            self.peak_threshold = max(self.charging_hub.grid.grid_usage)
        new_objective = self.charging_hub.update_objective_function(self.peak_threshold)
        reward -= new_objective - self.objective
        self.objective = new_objective
        # print(reward, self.objective, self.peak_threshold)
        return reward

    def update_vehicles_status(self):
        for request in [
            request
            for request in self.requests
            if request.mode in ["Connected", "Fully_charged"]
        ]:
            request.update_status()

    def solve_pricing_with_perfect_info(self):
        first_pricing = False
        if first_pricing == False:
            self.get_exp_free_grid_capacity()
            connected_vehicles = [
                x
                for x in self.requests
                if x.mode is None and x.ev == 1 and x.energy_requested > 0
            ]
            # integrate_algos.perfect_info_pricing_charging_routing(vehicles=connected_vehicles,
            #                                               charging_stations=self.chargers, env=self.env,
            #                                               grid_capacity=self.free_grid_capa_actual,
            #                                               electricity_cost=self.electricity_tariff,
            #                                               baseload=self.base_load_list,
            #                                               sim_time=self.sim_time,
            #                                               generation=self.generation_list)
            solution = nonlinear_pricing(
                vehicles_list=connected_vehicles,
                electricity_cost=self.electricity_tariff,
                PV=self.generation_list,
                base_load=self.base_load_list,
                sim_time=self.sim_time,
            )
            if Configuration.instance().dynamic_fix_term_pricing:

                self.price_schedules = [solution["p_0"], solution["alpha"]]
            else:
                self.price_schedules = solution["alpha"]
            first_scheduling = True
            # hour = int((self.env.now) / 60)
            # for request in self.requests:
            #     if request.ev == 1:
            #         request.charging_power = request.charge_schedule[hour]
