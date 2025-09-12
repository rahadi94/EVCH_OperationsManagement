# Executes full simulation routine with integrated agent decision system
# Supports both learnable agents (RL agents) and non-learnable agents (rule-based, algorithm agents)

from resources.configuration.configuration import Configuration
from run_simulation import run_single_simulation
from simulation.operations.agent_decision_system import agent_decision_system
from utilities.rl_agents.interfaces import DecisionType
from utilities.agent_factory import create_agent, is_agent_learnable, get_agent_configuration
from utilities.training_manager import run_learnable_agent_training
from resources.logging.log import lg


def run_experiments(config: Configuration = None):
    """
    Main experiment runner that orchestrates the simulation.
    
    This function:
    1. Creates and configures agents based on configuration
    2. Registers agents with the decision system
    3. Runs training for learnable agents or single simulation for non-learnable agents
    
    Args:
        config: Configuration instance. If None, will create one from command line args.
    """
    if config is None:
        config = Configuration.instance()
    
    print("Starting EVCC simulation with agent decision system...")
    
    # Get agent configuration
    agent_config = get_agent_configuration(config)
    
    # Create and register agents
    agents = {}
    learnable_agents = {}
    
    # Create pricing agent if configured
    if agent_config["pricing"]["agent_type"]:
        pricing_agent = create_agent(
            "pricing", 
            agent_config["pricing"]["agent_type"], 
            config=config,
            strategy=agent_config["pricing"]["strategy"]
        )
        agents["pricing"] = pricing_agent
        agent_decision_system.register_agent(DecisionType.PRICING, pricing_agent)
        print(f"Created pricing agent: {pricing_agent.__class__.__name__} with strategy: {agent_config['pricing']['strategy']}")
        
        # Check if this is a learnable agent
        if is_agent_learnable(agent_config["pricing"]["agent_type"]):
            learnable_agents["pricing"] = pricing_agent
            print(f"Learnable agent detected: pricing - {pricing_agent.__class__.__name__}")
    
    # Create charging agent if configured
    if agent_config["charging"]["agent_type"]:
        charging_agent = create_agent(
            "charging", 
            agent_config["charging"]["agent_type"], 
            config=config,
            algorithm=agent_config["charging"]["algorithm"],
            strategy=agent_config["charging"]["strategy"]
        )
        agents["charging"] = charging_agent
        agent_decision_system.register_agent(DecisionType.CHARGING, charging_agent)
        print(f"Created charging agent: {charging_agent.__class__.__name__} with strategy: {agent_config['charging']['strategy']}")
        
        if is_agent_learnable(agent_config["charging"]["agent_type"]):
            learnable_agents["charging"] = charging_agent
            print(f"Learnable agent detected: charging - {charging_agent.__class__.__name__}")
    
    # Create storage agent if configured
    if agent_config["storage"]["agent_type"]:
        storage_agent = create_agent(
            "storage", 
            agent_config["storage"]["agent_type"], 
            config=config,
            algorithm=agent_config["storage"]["algorithm"],
            strategy=agent_config["storage"]["strategy"]
        )
        agents["storage"] = storage_agent
        agent_decision_system.register_agent(DecisionType.STORAGE, storage_agent)
        print(f"Created storage agent: {storage_agent.__class__.__name__} with strategy: {agent_config['storage']['strategy']}")
        
        if is_agent_learnable(agent_config["storage"]["agent_type"]):
            learnable_agents["storage"] = storage_agent
            print(f"Learnable agent detected: storage - {storage_agent.__class__.__name__}")
    
    # Create routing agent if configured
    if agent_config["routing"]["agent_type"]:
        routing_agent = create_agent(
            "routing", 
            agent_config["routing"]["agent_type"], 
            config=config,
            algorithm=agent_config["routing"]["algorithm"],
            strategy=agent_config["routing"]["strategy"]
        )
        agents["routing"] = routing_agent
        agent_decision_system.register_agent(DecisionType.ROUTING, routing_agent)
        print(f"Created routing agent: {routing_agent.__class__.__name__} with strategy: {agent_config['routing']['strategy']}")
        
        if is_agent_learnable(agent_config["routing"]["agent_type"]):
            learnable_agents["routing"] = routing_agent
            print(f"Learnable agent detected: routing - {routing_agent.__class__.__name__}")
    
    # Enable dynamic pricing if using learnable pricing agents
    if (agent_config["pricing"]["agent_type"] and 
        is_agent_learnable(agent_config["pricing"]["agent_type"])):
        config.dynamic_pricing = True
        print(f"Enabled dynamic pricing for learnable agent: {agent_config['pricing']['agent_type']}")
    
    # Run experiments based on agent types
    if learnable_agents:
        print(f"\nFound {len(learnable_agents)} learnable agents. Running training...")
        
        # Run training for each learnable agent
        for decision_type, agent in learnable_agents.items():
            agent_type = agent_config[decision_type]["agent_type"]
            run_learnable_agent_training(agent_type, agent, decision_type, config)
            
    else:
        print("\nNo learnable agents detected. Running single simulation...")
        
        # Run single simulation with non-learnable agents
        # Use a default start day if not specified
        default_start_day = "2019-05-20"  # Default Monday in May 2019
        
        # Prepare results parameters for single simulation
        results_params = [f"{getattr(config, 'POST_FIX', 'sim')}", f"state{9}", f"week{1}"]
        
        run_single_simulation(
            charging_agent=agents.get("charging"),
            storage_agent=agents.get("storage"),
            pricing_agent=agents.get("pricing"),
            num_charger={"fast_one": config.facility_size, "fast_two": 0, "fast_four": 0, 
                        "slow_one": 0, "slow_two": 0, "slow_four": 0},
            turn_off_monitoring=False,
            turn_on_results=results_params,  # Pass list instead of boolean
            turn_on_plotting=True,
            transformer_num=config.TRANSFORMER_NUM,
            storage_capa=config.STORAGE_SIZE,
            pv_capa=config.PV_INSTALLED_CAPA,
            year=9,
            start_day=default_start_day,
            config=config
        )
    
    print("Simulation completed successfully!")


if __name__ == "__main__":
    run_experiments()