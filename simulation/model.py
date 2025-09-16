from resources.configuration.configuration import Configuration
from resources.logging.log import lg
from simulation.infrastructure.grid import GridCapacity
from simulation.infrastructure.parking_lot import ParkingLot
from simulation.operations.operator import Operator
from simulation.enums.vehicle_status import VehicleStatus
import pandas as pd
from simulation.infrastructure.ev_charger import EVCharger
from simulation.infrastructure.electric_generator import NonDispatchableGenerator
from simulation.infrastructure.electric_storage import ElectricStorage
import utilities.visualization as viz
import utilities.sim_input_processing as prep
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# NOTE: unit sim time is defined as 1 minute real time!
from simulation.preferences.vehicle import Vehicle


@dataclass
class SimulationConfig:
    """Configuration data for simulation parameters."""
    base_path: str
    raw_output_save_path: str
    visuals_save_path: str
    cache_path: str
    post_fix: str
    sim_season: str
    sim_start_date: str
    day_types: List[str]
    sim_duration: int
    facility_list: List[str]
    ev_share: float
    demand_gen_approach: str
    geography: str
    limit_requests_to_capa: bool
    year: int


@dataclass
class InfrastructureConfig:
    """Configuration data for infrastructure parameters."""
    parking_capa: int
    grid_capa: float
    transformer_num: int
    charging_capa: Union[float, Dict[str, float]]
    min_facility_baseload: float
    max_facility_baseload: float
    installed_capa_PV: float
    installed_storage: float
    charging_num: Union[int, Dict[str, int]]
    connector_num: int
    chargers_type: str


@dataclass
class OperationsConfig:
    """Configuration data for operations parameters."""
    planning_interval: int
    optimization_period_length: int
    lookahead: int
    lookback: int
    routing_algo: str
    charging_algo: str
    storage_algo: str
    scheduling_mode: str
    service_level: float
    minimum_served_demand: float
    penalty_for_missed_kWh: float
    planning: bool
    objective: str


class EVCC_Sim_Model:
    """
    Main simulation model for Electric Vehicle Charging Center (EVCC).
    
    This class implements the singleton pattern and manages the entire simulation
    including infrastructure, vehicles, and operations.
    """
    
    # Singleton instance
    __instance = None

    @staticmethod
    def instance() -> "EVCC_Sim_Model":
        """Get the singleton instance of the simulation model."""
        if EVCC_Sim_Model.__instance is None:
            raise Exception("World was not initialized")
        return EVCC_Sim_Model.__instance

    @staticmethod
    def init(
        env: Any,
        base_path: str,
        raw_output_save_path: str,
        visuals_save_path: str,
        cache_path: str,
        post_fix: str,
        sim_season: str,
        sim_start_date: str,
        day_types: List[str],
        sim_duration: int,
        facility_list: List[str],
        ev_share: float,
        demand_gen_approach: str,
        geography: str,
        limit_requests_to_capa: bool,
        parking_capa: int,
        grid_capa: float,
        transformer_num: int,
        charging_capa: Union[float, Dict[str, float]],
        min_facility_baseload: float,
        max_facility_baseload: float,
        installed_capa_PV: float,
        installed_storage: float,
        charging_num: Union[int, Dict[str, int]],
        connector_num: int,
        electricity_tariff: List[float],
        prices: Dict[str, float],
        year: int,
        planning_interval: int,
        optimization_period_length: int,
        lookahead: int,
        lookback: int,
        routing_algo: str,
        charging_algo: str,
        storage_algo: str,
        scheduling_mode: str,
        service_level: float,
        minimum_served_demand: float,
        penalty_for_missed_kWh: float,
        planning: bool,
        objective: str,
        charging_agent: Optional[Any] = None,
        storage_agent: Optional[Any] = None,
        pricing_agent: Optional[Any] = None,
        chargers_type: str = "single",
        reset: bool = False,
    ) -> "EVCC_Sim_Model":
        """
        Initialize the singleton instance of the simulation model.
        
        Args:
            env: SimPy environment
            base_path: Base path for data files
            raw_output_save_path: Path for raw output data
            visuals_save_path: Path for visualization outputs
            cache_path: Path for cached data
            post_fix: Postfix for file naming
            sim_season: simulation season
            sim_start_date: simulation start date
            day_types: List of day types
            sim_duration: simulation duration in days
            facility_list: List of facility IDs
            ev_share: Electric vehicle share
            demand_gen_approach: Demand generation approach
            geography: Geographic region
            limit_requests_to_capa: Whether to limit requests to capacity
            parking_capa: Parking capacity
            grid_capa: Grid capacity
            transformer_num: Number of transformers
            charging_capa: Charging capacity
            min_facility_baseload: Minimum facility base load
            max_facility_baseload: Maximum facility base load
            installed_capa_PV: Installed PV capacity
            installed_storage: Installed storage capacity
            charging_num: Number of chargers
            connector_num: Number of connectors
            electricity_tariff: Electricity tariff
            prices: Price configuration
            year: simulation year
            planning_interval: Planning interval
            optimization_period_length: Optimization period length
            lookahead: Lookahead periods
            lookback: Lookback periods
            routing_algo: Routing algorithm
            charging_algo: Charging algorithm
            storage_algo: Storage algorithm
            scheduling_mode: Scheduling mode
            service_level: Service level
            minimum_served_demand: Minimum served demand
            penalty_for_missed_kWh: Penalty for missed kWh
            planning: Whether planning is enabled
            objective: Objective function
            charging_agent: Charging agent
            storage_agent: Storage agent
            pricing_agent: Pricing agent
            chargers_type: Type of chargers
            reset: Whether to reset the singleton instance
            
        Returns:
            EVCC_Sim_Model: The singleton instance
        """
        if EVCC_Sim_Model.__instance is None or reset is True:
            EVCC_Sim_Model(
                env=env,
                base_path=base_path,
                raw_output_save_path=raw_output_save_path,
                visuals_save_path=visuals_save_path,
                cache_path=cache_path,
                post_fix=post_fix,
                sim_season=sim_season,
                sim_start_date=sim_start_date,
                day_types=day_types,
                sim_duration=sim_duration,
                facility_list=facility_list,
                ev_share=ev_share,
                demand_gen_approach=demand_gen_approach,
                geography=geography,
                limit_requests_to_capa=limit_requests_to_capa,
                parking_capa=parking_capa,
                grid_capa=grid_capa,
                transformer_num=transformer_num,
                charging_capa=charging_capa,
                min_facility_baseload=min_facility_baseload,
                max_facility_baseload=max_facility_baseload,
                installed_capa_PV=installed_capa_PV,
                installed_storage=installed_storage,
                charging_num=charging_num,
                connector_num=connector_num,
                electricity_tariff=electricity_tariff,
                prices=prices,
                year=year,
                planning_interval=planning_interval,
                optimization_period_length=optimization_period_length,
                lookahead=lookahead,
                lookback=lookback,
                routing_algo=routing_algo,
                charging_algo=charging_algo,
                storage_algo=storage_algo,
                scheduling_mode=scheduling_mode,
                service_level=service_level,
                minimum_served_demand=minimum_served_demand,
                penalty_for_missed_kWh=penalty_for_missed_kWh,
                planning=planning,
                objective=objective,
                charging_agent=charging_agent,
                storage_agent=storage_agent,
                pricing_agent=pricing_agent,
                chargers_type=chargers_type,
            )
        return EVCC_Sim_Model.__instance

    def __init__(
        self,
        env: Any,
        base_path: str,
        raw_output_save_path: str,
        visuals_save_path: str,
        cache_path: str,
        post_fix: str,
        sim_season: str,
        sim_start_date: str,
        day_types: List[str],
        sim_duration: int,
        facility_list: List[str],
        ev_share: float,
        demand_gen_approach: str,
        geography: str,
        limit_requests_to_capa: bool,
        parking_capa: int,
        grid_capa: float,
        transformer_num: int,
        charging_capa: Union[float, Dict[str, float]],
        min_facility_baseload: float,
        max_facility_baseload: float,
        installed_capa_PV: float,
        installed_storage: float,
        charging_num: Union[int, Dict[str, int]],
        connector_num: int,
        electricity_tariff: List[float],
        prices: Dict[str, float],
        year: int,
        planning_interval: int,
        optimization_period_length: int,
        lookahead: int,
        lookback: int,
        routing_algo: str,
        charging_algo: str,
        storage_algo: str,
        scheduling_mode: str,
        service_level: float,
        minimum_served_demand: float,
        penalty_for_missed_kWh: float,
        planning: bool,
        objective: str,
        charging_agent: Optional[Any] = None,
        storage_agent: Optional[Any] = None,
        pricing_agent: Optional[Any] = None,
        chargers_type: str = "single",
        config: Optional[Any] = None,
    ):
        """
        Initialize the EVCC simulation model.
        
        Args:
            base_path: Base path for data files
            raw_output_save_path: Path for raw output data
            visuals_save_path: Path for visualization outputs
            cache_path: Path for cached data
            post_fix: Postfix for file naming
            env: SimPy environment
            sim_season: simulation season
            sim_start_date: simulation start date
            day_types: List of day types
            sim_duration: simulation duration in days
            facility_list: List of facility IDs
            ev_share: Electric vehicle share
            demand_gen_approach: Demand generation approach
            geography: Geographic region
            limit_requests_to_capa: Whether to limit requests to capacity
            parking_capa: Parking capacity
            grid_capa: Grid capacity
            transformer_num: Number of transformers
            charging_capa: Charging capacity
            min_facility_baseload: Minimum facility base load
            max_facility_baseload: Maximum facility base load
            installed_capa_PV: Installed PV capacity
            installed_storage: Installed storage capacity
            charging_num: Number of chargers
            connector_num: Number of connectors
            electricity_tariff: Electricity tariff
            prices: Price configuration
            year: simulation year
            planning_interval: Planning interval
            optimization_period_length: Optimization period length
            lookahead: Lookahead periods
            lookback: Lookback periods
            routing_algo: Routing algorithm
            charging_algo: Charging algorithm
            storage_algo: Storage algorithm
            scheduling_mode: Scheduling mode
            service_level: Service level
            minimum_served_demand: Minimum served demand
            penalty_for_missed_kWh: Penalty for missed kWh
            planning: Whether planning is enabled
            objective: Objective function
            charging_agent: Charging agent
            storage_agent: Storage agent
            pricing_agent: Pricing agent
            chargers_type: Type of chargers
        """
        # Set singleton instance
        EVCC_Sim_Model.__instance = self
        
        # Store configuration
        self.config = config
        
        # Initialize configuration objects
        self._init_simulation_config(
            base_path, raw_output_save_path, visuals_save_path, cache_path, post_fix,
            sim_season, sim_start_date, day_types, sim_duration, facility_list,
            ev_share, demand_gen_approach, geography, limit_requests_to_capa, year
        )
        
        self._init_infrastructure_config(
            parking_capa, grid_capa, transformer_num, charging_capa,
            min_facility_baseload, max_facility_baseload, installed_capa_PV,
            installed_storage, charging_num, connector_num, chargers_type
        )
        
        self._init_operations_config(
            planning_interval, optimization_period_length, lookahead, lookback,
            routing_algo, charging_algo, storage_algo, scheduling_mode,
            service_level, minimum_served_demand, penalty_for_missed_kWh,
            planning, objective
        )
        
        # Initialize simulation environment
        self._init_simulation_environment(env, electricity_tariff, prices)
        
        # Initialize infrastructure and data
        self._init_infrastructure_and_data()
        
        # Initialize operations
        self._init_operations(charging_agent, storage_agent, pricing_agent)

    def _init_simulation_config(
        self,
        base_path: str,
        raw_output_save_path: str,
        visuals_save_path: str,
        cache_path: str,
        post_fix: str,
        sim_season: str,
        sim_start_date: str,
        day_types: List[str],
        sim_duration: int,
        facility_list: List[str],
        ev_share: float,
        demand_gen_approach: str,
        geography: str,
        limit_requests_to_capa: bool,
        year: int
    ) -> None:
        """Initialize simulation configuration parameters."""
        self.sim_config = SimulationConfig(
            base_path=base_path,
            raw_output_save_path=raw_output_save_path,
            visuals_save_path=visuals_save_path,
            cache_path=cache_path,
            post_fix=post_fix,
            sim_season=sim_season,
            sim_start_date=sim_start_date,
            day_types=day_types,
            sim_duration=sim_duration,
            facility_list=facility_list,
            ev_share=ev_share,
            demand_gen_approach=demand_gen_approach,
            geography=geography,
            limit_requests_to_capa=limit_requests_to_capa,
            year=year
        )
        
        # Set instance attributes for backward compatibility
        self.base_path = base_path
        self.raw_output_save_path = raw_output_save_path
        self.post_fix = post_fix
        self.visuals_save_path = visuals_save_path
        self.cache_path = cache_path
        self.sim_season = sim_season
        self.sim_start_date = sim_start_date
        self.day_types = day_types
        self.sim_duration = sim_duration
        self.facility_list = facility_list
        self.ev_share = ev_share
        self.geography = geography
        self.demand_gen_approach = demand_gen_approach
        self.limit_requests_to_capa = limit_requests_to_capa
        self.year = year

    def _init_infrastructure_config(
        self,
        parking_capa: int,
        grid_capa: float,
        transformer_num: int,
        charging_capa: Union[float, Dict[str, float]],
        min_facility_baseload: float,
        max_facility_baseload: float,
        installed_capa_PV: float,
        installed_storage: float,
        charging_num: Union[int, Dict[str, int]],
        connector_num: int,
        chargers_type: str
    ) -> None:
        """Initialize infrastructure configuration parameters."""
        self.infrastructure_config = InfrastructureConfig(
            parking_capa=parking_capa,
            grid_capa=grid_capa,
            transformer_num=transformer_num,
            charging_capa=charging_capa,
            min_facility_baseload=min_facility_baseload,
            max_facility_baseload=max_facility_baseload,
            installed_capa_PV=installed_capa_PV,
            installed_storage=installed_storage,
            charging_num=charging_num,
            connector_num=connector_num,
            chargers_type=chargers_type
        )
        
        # Set instance attributes for backward compatibility
        self.chargers_type = chargers_type
        self.charging_capa = charging_capa
        self.charging_num = charging_num
        self.connector_num = connector_num
        self.min_facility_baseload = min_facility_baseload
        self.max_facility_baseload = max_facility_baseload
        self.parking_capa = parking_capa
        self.transformer_num = transformer_num
        self.grid_capa = grid_capa + transformer_num * 200  # in kW
        self.installed_capa_PV = installed_capa_PV
        self.storage_capacity = installed_storage

    def _init_operations_config(
        self,
        planning_interval: int,
        optimization_period_length: int,
        lookahead: int,
        lookback: int,
        routing_algo: str,
        charging_algo: str,
        storage_algo: str,
        scheduling_mode: str,
        service_level: float,
        minimum_served_demand: float,
        penalty_for_missed_kWh: float,
        planning: bool,
        objective: str
    ) -> None:
        """Initialize operations configuration parameters."""
        self.operations_config = OperationsConfig(
            planning_interval=planning_interval,
            optimization_period_length=optimization_period_length,
            lookahead=lookahead,
            lookback=lookback,
            routing_algo=routing_algo,
            charging_algo=charging_algo,
            storage_algo=storage_algo,
            scheduling_mode=scheduling_mode,
            service_level=service_level,
            minimum_served_demand=minimum_served_demand,
            penalty_for_missed_kWh=penalty_for_missed_kWh,
            planning=planning,
            objective=objective
        )
        
        # Set instance attributes for backward compatibility
        self.planning_interval = planning_interval
        self.optimization_period_length = optimization_period_length
        self.lookahead = lookahead
        self.lookback = lookback
        self.routing_algo = routing_algo
        self.charging_algo = charging_algo
        self.storage_algo = storage_algo
        self.scheduling_mode = scheduling_mode
        self.service_level = service_level
        self.minimum_served_demand = minimum_served_demand
        self.penalty_for_missed_kWh = penalty_for_missed_kWh
        self.planning = planning
        self.objective = objective

    def _init_simulation_environment(
        self,
        env: Any,
        electricity_tariff: List[float],
        prices: Dict[str, float]
    ) -> None:
        """Initialize simulation environment and basic parameters."""
        self.env = env
        self.sim_time = self.sim_duration * 24 * 60  # in minutes
        self.electricity_tariff = electricity_tariff
        self.prices = prices.copy()
        self.prices["peak"] = Configuration.instance().peak_cost
        
        # Load configuration
        config = Configuration.instance()
        self.random_demand = config.random_demand
        self.data_source = config.data_source
        self.benchmarking = config.benchmarking
        self.dynamic_pricing = config.dynamic_pricing
        self.dynamic_charging = getattr(config, 'dynamic_charging', False)
        self.peak_threshold = config.peak_threshold

    def _init_infrastructure_and_data(self) -> None:
        """Initialize infrastructure components and load data."""
        # Load base load data
        self.base_load = prep.get_sim_baseload_curve(
            base_path=self.base_path,
            cache_path=self.cache_path,
            sim_start_day=self.sim_start_date,
            num_lookback_periods=self.lookback,
            sim_duration=self.sim_duration,
            min_facility_baseload=self.min_facility_baseload,
            max_facility_baseload=self.max_facility_baseload,
        )
        
        # Load charging request data
        self.request_data = prep.get_sim_charging_requests(
            base_path=self.base_path,
            cache_path=self.cache_path,
            demand_gen_approach=self.demand_gen_approach,
            limit_requests_to_capa=self.limit_requests_to_capa,
            parking_capacity=self.parking_capa,
            sim_start_day=self.sim_start_date,
            day_types=self.day_types,
            sim_duration=self.sim_duration,
            sim_seasons=self.sim_season,
            facility_list=self.facility_list,
            ev_share=self.ev_share,
            max_charge_rate=self.charging_capa,
            geography=self.geography,
            data_source=self.data_source,
            random_demand=self.random_demand
        )
        
        # Setup simulation
        self.requests = self.EVCC_sim_setup(self.request_data)

    def _init_operations(
        self,
        charging_agent: Optional[Any],
        storage_agent: Optional[Any],
        pricing_agent: Optional[Any]
    ) -> None:
        """Initialize operations and agents."""
        # Initialize costs and rewards
        self.costs = dict(investment=0, operations=0)
        self.objective_function = 0
        self.total_energy_charged = 0
        self.reward = dict(costs=0, profit=0, feasibility=0, feasibility_storage=0)
        
        # Create operator
        self.operator = Operator(
            env=self.env,
            requests=self.requests,
            chargers=self.chargers,
            routing_strategy=self.routing_algo,
            charging_strategy=self.charging_algo,
            storage_strategy=self.storage_algo,
            charging_capa=self.charging_capa,
            grid_capa=self.grid_capa,
            non_dispatchable_generator=self.non_dispatchable_generator,
            electric_storage=self.electric_storage,
            sim_time=self.sim_time,
            electricity_tariff=self.electricity_tariff,
            connector_num=self.connector_num,
            parking_spots=self.parking_spots,
            baseload=self.base_load,
            max_facility_baseload=self.max_facility_baseload,
            planning_interval=self.planning_interval,
            optimization_period_length=self.optimization_period_length,
            num_lookahead_planning_periods=self.lookahead,
            num_lookback_periods=self.lookback,
            service_level=self.service_level,
            charging_hub=self,
            minimum_served_demand=self.minimum_served_demand,
            config=getattr(self, 'config', None),
        )
        
        # Initialize agents
        self._init_agents(charging_agent, storage_agent, pricing_agent)

    def _init_agents(
        self,
        charging_agent: Optional[Any],
        storage_agent: Optional[Any],
        pricing_agent: Optional[Any]
    ) -> None:
        """Initialize decision-making agents."""
        self.charging_agent = charging_agent
        self.storage_agent = storage_agent
        self.pricing_agent = pricing_agent
        
        # Setup charging agent
        if self.charging_agent:
            # Check if agent has environment (RL agents) or not (rule-based/algorithm agents)
            if hasattr(self.charging_agent, 'environment'):
                # Set charging_hub and env in the environment for RL agents
                self.charging_agent.environment.charging_hub = self
                self.charging_agent.environment.env = self.env
                self.charging_agent.environment.state = (
                    self.charging_agent.environment.get_state(self, self.env)
                )
                self.charging_agent.reset_game()
            else:
                # For rule-based/algorithm agents, just store the charging hub reference
                if hasattr(self.charging_agent, 'set_charging_hub'):
                    self.charging_agent.set_charging_hub(self)
                print(f"Initialized charging agent: {self.charging_agent.__class__.__name__}")
        
        # Setup pricing agent
        if self.pricing_agent:
            # Check if agent has environment (RL agents) or not (rule-based/algorithm agents)
            if hasattr(self.pricing_agent, 'environment'):
                # Set charging_hub and env in the environment for RL agents
                self.pricing_agent.environment.charging_hub = self
                self.pricing_agent.environment.env = self.env
                self.pricing_agent.environment.state = self.pricing_agent.environment.get_state(
                    self, self.env
                )
                self.pricing_agent.reset_game()
            else:
                # For rule-based/algorithm agents, just store the charging hub reference
                if hasattr(self.pricing_agent, 'set_charging_hub'):
                    self.pricing_agent.set_charging_hub(self)
                print(f"Initialized pricing agent: {self.pricing_agent.__class__.__name__}")
        
        # Setup storage agent
        if self.storage_agent:
            # Check if agent has environment (RL agents) or not (rule-based/algorithm agents)
            if hasattr(self.storage_agent, 'environment'):
                # Set charging_hub and env in the environment for RL agents
                self.storage_agent.environment.charging_hub = self
                self.storage_agent.environment.env = self.env
                self.storage_agent.environment.state = (
                    self.storage_agent.environment.get_state(self, self.env)
                )
                self.storage_agent.reset_game()
            else:
                # For rule-based/algorithm agents, just store the charging hub reference
                if hasattr(self.storage_agent, 'set_charging_hub'):
                    self.storage_agent.set_charging_hub(self)
                print(f"Initialized storage agent: {self.storage_agent.__class__.__name__}")
        
        # Link agents to operator
        self.operator.charging_agent = charging_agent
        self.operator.storage_agent = storage_agent
        self.operator.pricing_agent = pricing_agent

    # ============================================================================
    # SETUP ENVIRONMENT
    # ============================================================================

    def EVCC_sim_setup(self, request_data: pd.DataFrame) -> List[Any]:
        """
        Setup the EVCC simulation environment.
        
        Args:
            request_data: DataFrame containing charging request data
            
        Returns:
            List of vehicle request objects
        """
        # Initialize infrastructure
        self.initialize_infrastructure()

        # TODO: This is just a hack to deal with requests with the same arrival period
        # request_data['EntryMinutesFromSimStart'] = request_data['EntryMinutesFromSimStart'].astype('int64')
        # request_data = request_data.drop_duplicates(subset=['EntryMinutesFromSimStart'])
        # request_data = request_data.reset_index(drop=True)

        # Initialize vehicle population
        requests = self.initialize_vehicle_population(request_data)

        return requests

    # ============================================================================
    # INFRASTRUCTURE INITIALIZATION
    # ============================================================================

    def initialize_infrastructure(self) -> None:
        """
        Initialize the charging, parking and grid infrastructure.
        
        Creates parking lots, chargers, grid capacity, PV generation, and storage systems.
        """
        # Initialize parking infrastructure
        self._init_parking_infrastructure()
        
        # Initialize charging infrastructure
        self._init_charging_infrastructure()
        
        # Initialize energy infrastructure
        self._init_energy_infrastructure()

    def _init_parking_infrastructure(self) -> None:
        """Initialize parking infrastructure."""
        self.parking_lot = ParkingLot(env=self.env, parking_capacity=100000)
        self.parking_spots = self.parking_lot.parking_spots

    def _init_charging_infrastructure(self) -> None:
        """Initialize charging infrastructure."""
        self.chargers = []
        
        if self.chargers_type == "single":
            self._create_single_type_chargers()
        else:
            self._create_mixed_type_chargers()

    def _create_single_type_chargers(self) -> None:
        """Create chargers of a single type."""
        for i in range(self.charging_num):
            charger = EVCharger(
                env=self.env,
                id=i,
                power=self.charging_capa,
                period_length=self.planning_interval,
                number_of_connectors=self.connector_num,
            )
            self.chargers.append(charger)

    def _create_mixed_type_chargers(self) -> None:
        """Create chargers of mixed types."""
        id_indicator = 0
        
        # Create fast chargers
        self._create_charger_group("fast_one", 1, id_indicator)
        id_indicator += self.charging_num.get("fast_one", 0)
        
        self._create_charger_group("fast_two", 2, id_indicator)
        id_indicator += self.charging_num.get("fast_two", 0)
        
        self._create_charger_group("fast_four", 4, id_indicator)
        id_indicator += self.charging_num.get("fast_four", 0)
        
        # Create slow chargers
        self._create_charger_group("slow_one", 1, id_indicator)
        id_indicator += self.charging_num.get("slow_one", 0)
        
        self._create_charger_group("slow_two", 2, id_indicator)
        id_indicator += self.charging_num.get("slow_two", 0)
        
        self._create_charger_group("slow_four", 4, id_indicator)

    def _create_charger_group(self, charger_type: str, connectors: int, start_id: int) -> None:
        """Create a group of chargers with the same specifications."""
        count = self.charging_num.get(charger_type, 0)
        power = self.charging_capa["fast"] if "fast" in charger_type else self.charging_capa["slow"]
        
        for i in range(count):
            charger = EVCharger(
                env=self.env,
                id=start_id + i,
                power=power,
                period_length=self.planning_interval,
                number_of_connectors=connectors,
            )
            self.chargers.append(charger)

    def _init_energy_infrastructure(self) -> None:
        """Initialize energy infrastructure (grid, PV, storage)."""
        # Initialize grid capacity
        self.grid = GridCapacity(self.env, self.grid_capa)

        # Initialize PV generation
        self.non_dispatchable_generator = NonDispatchableGenerator(
            env=self.env,
            kW_peak=self.installed_capa_PV,
            base_path=self.base_path,
            cache_path=self.cache_path,
            sim_start_day=self.sim_start_date,
            sim_duration=self.sim_duration,
            num_lookback_periods=self.lookback,
        )

        # Initialize electric storage
        self.electric_storage = ElectricStorage(
            env=self.env, 
            max_capacity_kWh=self.storage_capacity
        )

    # ============================================================================
    # VEHICLE POPULATION INITIALIZATION
    # ============================================================================

    def initialize_vehicle_population(self, request_data: pd.DataFrame) -> List[Any]:
        """
        Generate ordered list of Vehicle objects entering the EVCC.
        
        Args:
            request_data: DataFrame containing parking and charging requests
            
        Returns:
            List of Vehicle objects representing charging requests
        """
        requests = []
        
        for i in range(len(request_data)):
            vehicle = self._create_vehicle_from_request(request_data, i)
            requests.append(vehicle)
            
        return requests

    def _create_vehicle_from_request(self, request_data: pd.DataFrame, index: int) -> Any:
        """
        Create a vehicle object from request data.
        
        Args:
            request_data: DataFrame containing request data
            index: Index of the request in the DataFrame
            
        Returns:
            Vehicle object
        """
        # Extract basic request information
        request_info = self._extract_request_info(request_data, index)
        
        # Create vehicle object
        vehicle = Vehicle(
            env=self.env,
            id=request_info["id"],
            facility=request_info["facility"],
            user_type=request_info["user_type"],
            arrival_date=request_info["arrival_date"],
            departure_date=request_info["departure_date"],
            arrival_time=request_info["arrival_time"],
            departure_time=request_info["departure_time"],
            arrival_period=request_info["arrival_period"],
            departure_period=request_info["departure_period"],
            ev=request_info["ev"],
            energy_requested_input=request_info["energy_requested"],
            sim_time=self.sim_time,
            energy_charged=0,  # Initialize to 0
            battery_size=request_data.loc[index, "BatterySize"]
        )
        
        return vehicle

    def _extract_request_info(self, request_data: pd.DataFrame, index: int) -> Dict[str, Any]:
        """
        Extract request information from DataFrame.
        
        Args:
            request_data: DataFrame containing request data
            index: Index of the request
            
        Returns:
            Dictionary containing extracted request information
        """
        return {
            "id": index,
            "facility": request_data.loc[index, "SiteID"],
            "user_type": request_data.loc[index, "ClusterName"],
            "arrival_date": request_data.loc[index, "EntryDate"],
            "departure_date": request_data.loc[index, "ExitDate"],
            "arrival_time": request_data.loc[index, "EntryDateTime"],
            "departure_time": request_data.loc[index, "ExitDateTime"],
            "arrival_period": request_data.loc[index, "EntryMinutesFromSimStart"],
            "departure_period": request_data.loc[index, "ExitMinutesFromSimStart"],
            "ev": self._determine_ev_status(request_data, index),
            "energy_requested": self._get_energy_requested(request_data, index),
            "park_duration": self._calculate_park_duration(request_data, index),
            "assigned_charger": None  # Will be assigned during routing
        }

    def _determine_ev_status(self, request_data: pd.DataFrame, index: int) -> bool:
        """Determine if the request is for an electric vehicle."""
        return bool(request_data.loc[index, "EV_yn"])

    def _get_energy_requested(self, request_data: pd.DataFrame, index: int) -> float:
        """Get the energy requested for the vehicle."""
        return request_data.loc[index, "final_kWhRequested_updated"] * 2

    def _calculate_park_duration(self, request_data: pd.DataFrame, index: int) -> int:
        """Calculate the parking duration for the vehicle."""
        arrival_period = request_data.loc[index, "EntryMinutesFromSimStart"]
        departure_period = request_data.loc[index, "ExitMinutesFromSimStart"]
        
        if self.benchmarking:
            arrival_period = int(arrival_period / 60) * 60
            departure_period = int(min(departure_period, self.sim_time) / 60) * 60
            
        return departure_period - arrival_period

    ##############################################################
    # RUN SIMULATION

    def run(self):
        """
        Running the whole process of serving the request and monitoring the resources
        """
        self.env.process(
            self.operator.request_queueing()
        )  # vehicle arrivals and assignment to charging station
        self.env.process(
            self.operator.get_charging_schedules_and_prices(
                self.charging_algo, mode=self.scheduling_mode
            )
        )
        if self.charging_algo not in [
            "integrated_storage",
            "perfect_info_with_storage",
        ]:
            if self.storage_capacity > 0:
                if not Configuration.instance().dynamic_storage_scheduling:
                    self.env.process(
                        self.operator.get_storage_schedule(
                            storage_strategy=self.storage_algo,
                            mode=self.scheduling_mode,
                        )
                    )
        self.env.process(self.operator.storage_process())
        self.env.process(
            self.grid.monitor(
                self.base_load,
                self.chargers,
                self.non_dispatchable_generator,
                self.electric_storage,
                energy_costs=self.electricity_tariff,
                vehicles=self.requests,
            )
        )
        if self.planning is False:
            for charging_station in self.chargers:
                self.env.process(charging_station.monitor())
            self.env.process(self.parking_lot.monitor())
            self.env.process(self.electric_storage.monitor())

    ############################################################################
    # SAVE AND PLOT SIMULATION RESULTS

    def convert_to_int_if_none(self, x):
        time = self.env
        if x:
            return int(x)

    def save_results(self, method="RL", year=9, week=1, post_fix=""):
        """
        Saving the results of request transitions, state of chargers, state of parking_spots and state of requests
        """
        # # Retrieving and saving request operational data
        requests_info = []
        for i in self.requests:
            info = {
                "facility": i.facility,
                "vehicle_id": i.id,
                "ev_yn": i.ev,
                "user_type": i.user_type,
                "arrival_time": i.arrival_time,
                "arrival_period": self.convert_to_int_if_none(i.arrival_period),
                "departure_time": i.departure_time,
                "departure_period": self.convert_to_int_if_none(i.departure_period),
                "assigned_charger": i.assigned_charger,
                "assigned_parking": i.assigned_parking,
                "assigned_time": self.convert_to_int_if_none(i.assigned_time),
                "estimated_waiting_time": i.estimated_waiting_time,
                "waiting_time": self.convert_to_int_if_none(i.waiting_time),
                "stop_charging_time": i.stop_charging_time,
                "energy_requested": i.energy_requested,
                "energy_charged": i.energy_charged,
                "charging_price": i.charging_price,
                "charging_max_power": i.max_charging_power,
                "average_power_requirement": i.average_power_requirement,
            }
            requests_info.append(info)
        results = pd.DataFrame(requests_info)
        results.to_csv(
            self.raw_output_save_path
            + f"requests_{method}_{year}_{week}_{post_fix}.csv"
        )

        # COMMENTED OUT FOR NOW
        # pd_ve = pd.DataFrame()
        # for j in self.requests:
        #    pd_ve = pd_ve.append(pd.DataFrame([j.info["mode"], j.info['SOC'], j.info['charging_power']]))
        # pd_ve.to_csv(f'/tmp/pycharm_project_194/Results/output/requests_details.csv')
        # pd_ve.to_csv(results_save_path+'requests_details.csv')

        # Retrieving and saving charger operational data
        pd_cs = pd.DataFrame()
        for c in self.chargers:
            # initialize and fill new df per each charger
            df = pd.DataFrame(
                [c.info["Connected"], c.info["Charging"], c.info["Consumption"]]
            )
            df["cs_id"] = c.id
            df["info"] = [
                "num_vehicles_connected",
                "num_vehicles_charging",
                "kWh_consumption",
            ]
            # append to combined df
            pd_cs = pd.concat([pd_cs, df], ignore_index=True)
        # save combined df
        pd_cs.to_csv(
            self.raw_output_save_path
            + f"CSs_{method}_{year}_{week}_{post_fix}_{Configuration.instance().pricing_agent_name}.csv"
        )

        # Retrieving and saving storage operational data
        storage_data = pd.DataFrame(self.electric_storage.info)
        storage_data["type"] = self.electric_storage.storage_type
        storage_data.to_csv(
            self.raw_output_save_path + f"storage_{method}_{year}_{week}_{post_fix}.csv"
        )

        # Saving historical price data
        self.operator.price_history.to_csv(
            self.raw_output_save_path
            + f"price_history_{method}_{year}_{week}_{post_fix}.csv"
        )

        # Calculating the operation and investment costs
        # self.objective_function_calculation()

    def update_objective_function(self, peak_threshold):
        self.costs["operations"] = 0
        self.costs["operations"] += self.grid.energy_costs
        # peak_charge = (max(self.grid.grid_usage) - peak_threshold) * self.prices['peak']
        # self.costs["operations"] += peak_charge  # Peak charge has already been discounted to daily charge in input!
        total_revenue = 0
        requests = [i for i in self.requests if i.ev == 1]

        for request in requests:
            energy_requested_adj = request.energy_requested
            if request.is_assigned:
                if request.energy_requested > 0:
                    total_revenue += (
                        min(request.energy_charged, energy_requested_adj)
                        * request.charging_price
                    )  # overserving request does not count!
                    total_revenue += request.park_duration * request.parking_fee

            if self.env.now >= 1440 - 60:
                total_revenue -= (
                    max(energy_requested_adj - request.energy_charged, 0)
                    * request.charging_price
                    * Configuration.instance().energy_missed_penalty
                )
            elif request.mode == VehicleStatus.LEFT:
                total_revenue -= (
                    max(energy_requested_adj - request.energy_charged, 0)
                    * request.charging_price
                    * Configuration.instance().energy_missed_penalty
                )
            total_revenue -= (
                max((request.raw_energy_demand - request.energy_requested), 0) * 0
            )
        # TODO: fix this
        # activate it when we have a single price
        # return (total_revenue * self.penalty_for_missed_kWh - (self.costs["operations"]))
        return total_revenue - (self.costs["operations"])

    def calculate_objective_function(self, initial_grid_capa):
        """
        Calculates investment cost, operational cost and service level, which is feedback for the greedy search algorithm
        :return:
        """
        ### Activate for investment problem
        ### TODO: define it as an option
        self.costs["operations"] = 0
        self.costs["operations"] += self.grid.energy_costs
        peak_charge = max(
            (max(self.grid.grid_usage) - self.peak_threshold) * self.prices["peak"], 0
        )
        self.costs[
            "operations"
        ] += peak_charge  # Peak charge has already been discounted to daily charge in input!
        lg.info(
            f'Daily operations Costs = {self.costs["operations"]}, Daily Investment Costs = {self.costs["investment"]}'
        )

        # Obtaining the average service level at the end
        total_energy_requested = 0
        total_energy_charged = 0
        total_energy_missed = 0
        total_energy_canceled = 0
        total_revenue = 0
        served_demand_proportion = 0
        number_request = 0
        extra_charge = 0

        requests = [i for i in self.requests if i.ev == 1]

        for request in requests:
            total_energy_canceled += max(
                (request.raw_energy_demand - request.energy_requested), 0
            )
            if request.energy_requested > 0:
                energy_requested_adj = max(
                    request.energy_requested * self.minimum_served_demand, 0
                )
                total_energy_missed += max(
                    energy_requested_adj - request.energy_charged, 0
                )
                total_energy_requested += (
                    energy_requested_adj  # request.energy_requested
                )
                total_energy_charged += min(
                    request.energy_charged, energy_requested_adj
                )  # overserving request does not count!
                extra_charge += max((request.energy_charged - energy_requested_adj), 0)
                if request.is_assigned:
                    total_revenue += (
                        min(request.energy_charged, energy_requested_adj)
                        * request.charging_price
                    )
                    total_revenue += request.park_duration * request.parking_fee
                    total_revenue -= (
                        max(energy_requested_adj - request.energy_charged, 0)
                        * request.charging_price
                        * Configuration.instance().energy_missed_penalty
                    )
                    served_demand_proportion += min(
                        1, request.energy_charged / request.energy_requested
                    )
                    number_request += 1
        lg.info(f"total missed demand is {total_energy_missed}")
        lg.info(
            f"total missed demand is {total_energy_missed / (total_energy_requested + 1)}"
        )
        # self.service_level = min(round(total_energy_charged / (total_energy_requested+1), 2), 1.00)
        if Configuration.instance().benchmarking:
            total_revenue += extra_charge * 0.15
        self.service_level = min(
            round(served_demand_proportion / max(number_request, 1), 2), 1.00
        )
        self.total_energy_charged = total_energy_charged
        self.total_energy_canceled = total_energy_canceled
        lg.error(
            f"service_level = {self.service_level}, energy_canceled = {total_energy_canceled}, "
            f"energy_charged = {total_energy_charged}, energy_missed = {total_energy_missed}"
        )
        if self.objective == "min_costs":
            self.objective_function = (
                total_energy_missed * self.penalty_for_missed_kWh
                + (self.costs["operations"])
            )
        if self.objective == "max_profits":
            self.objective_function = total_revenue - (self.costs["operations"])

    def visualize_results(
        self, model, sim_start_date, post_fix, visuals_save_path, palette="mako"
    ):
        """
        Run plottig routines on results output data
        :return:
        """

        viz.get_visuals(
            model=model,
            palette=palette,
            sim_start_date=sim_start_date,
            visuals_save_path=visuals_save_path,
            post_fix=post_fix,
        )
