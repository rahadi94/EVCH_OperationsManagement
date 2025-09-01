# EVCC_Sim

This repository contains a flexible simulation framework for **Electric Vehicle (EV) Charging Clusters (EVCCs)**. 
EVCCs are large-scale EV-charging-enabled parking lots. Examples include workplace charging facilities, destination parking lots (e.g., mall, supermarket or gym parking garages) or fleet depots.

EVCCs are expected to become a core component of the future charging portfolio outweighing the importance of home charging by some estimates. Planning (sizing) and operating such EVCCs is a non-trivial task with three-way inter-dependencies between (1) user preferences, (2) infrastructure decisions and (3) operations management.

This simulation is intended to explore these interdependencies through extensive sensitivity testing and through testing new algorithms and models for sizing and operating EVCCs. The module structure is as follows:

## 🏗️ Architecture

The EVCC simulation framework is built with a modular, decoupled architecture that separates concerns and enables easy integration with different RL algorithms and libraries.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EVCC Simulation                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ Preferences │  │Infrastructure│  │ Operations  │  │ Results │ │
│  │   Module    │  │   Module    │  │   Module    │  │ Module  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RL Agent Integration                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐ │
│  │   RL Library    │    │   Gym Adapter    │    │   EVCH Gym  │ │
│  │  (Stable-Bas3,  │───▶│   (Standard      │───▶│ Environment │ │
│  │   RLlib, etc.)  │    │   Interface)     │    │  (Wrapper)  │ │
│  └─────────────────┘    └──────────────────┘    └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

The following modules are included:

- **`Preferences` Module:** Initializes vehicle objects with respective charging and parking preferences (i.e., requests) based on empirical data
- **`Infrastructure` Module:** Initializes infrastructure objects (EV supply equipment (EVSE), connectors per each EVSE, grid connection capacity, on-site storage and on-site generation (PV))  
- **`Operations` Module:** Contains algorithms for assigning physical space (vehicle routing) and electrical capacity (vehicle charging) to individual vehicle objects based on a pre-defined charging policy
- **`Results` Module:** Monitors EVCC activity in pre-defined intervals and accounts costs. Includes plotting routines.

## 🆕 NEW: Agent Decision System

The framework now includes a **unified agent decision system** that ensures **ALL decisions** in the EV charging operations are made by agents (RL agents, rule-based agents, algorithm agents, etc.) rather than being hardcoded in business logic.

### Agent Decision System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Decision System                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Pricing   │  │   Charging  │  │   Storage   │              │
│  │   Service   │  │   Service   │  │   Service   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                 │                │                    │
│         ▼                 ▼                ▼                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Agent     │  │   Agent     │  │   Agent     │              │
│  │ Decision    │  │ Decision    │  │ Decision    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                 │                │                    │
│         └─────────────────┼────────────────┘                    │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Agent Decision System                          │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │ │
│  │  │   RL SAC    │ │ Rule-Based  │ │ Algorithm   │            │ │
│  │  │   Agent     │ │   Agent     │ │   Agent     │            │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Decision Types

- **PRICING**: Energy prices, parking fees, dynamic pricing strategies
- **CHARGING**: Charging power allocation, schedules, priority assignment
- **STORAGE**: Energy storage operations, peak shaving, arbitrage
- **ROUTING**: Vehicle routing, parking allocation, queue management
- **VEHICLE_ASSIGNMENT**: Charging station assignment, connector allocation
- **PARKING_ALLOCATION**: Parking space allocation, duration optimization
- **GRID_MANAGEMENT**: Grid capacity management, load balancing
- **DEMAND_FORECASTING**: Energy demand prediction, load forecasting

### Supported Agent Types

- **RL_SAC**: Soft Actor-Critic reinforcement learning agent
- **RL_DQN**: Deep Q-Network reinforcement learning agent
- **RL_DDPG**: Deep Deterministic Policy Gradient agent
- **RULE_BASED**: Rule-based agents with predefined strategies
- **HEURISTIC**: Algorithm agents that wrap existing algorithms
- **OPTIMIZATION**: Mathematical optimization algorithms
- **ML_MODEL**: Machine learning models (neural networks, etc.)

### Algorithm Agents

The system includes **algorithm agents** that wrap all existing charging, routing, and storage algorithms:

#### **Charging Algorithm Agents**
- `uncontrolled`, `first_come_first_served`, `earliest_deadline_first`
- `least_laxity_first`, `equal_sharing`, `online_myopic`
- `online_multi_period`, `integrated_storage`, `perfect_info`
- `perfect_info_with_storage`

#### **Routing Algorithm Agents**
- `random`, `lowest_occupancy_first`, `fill_one_after_other`
- `lowest_utilization_first`, `matching_supply_demand`, `minimum_power_requirement`

#### **Storage Algorithm Agents**
- `uncontrolled`, `temporal_arbitrage`, `peak_shaving`

### Rule-Based Agents

Pre-built rule-based agents for common strategies:

#### **Pricing Agents**
- **Time-of-Use**: Peak/off-peak pricing based on time
- **Demand-Based**: Dynamic pricing based on current demand
- **Cost-Plus**: Fixed markup over base electricity cost

#### **Charging Agents**
- **First-Come-First-Served**: Serve vehicles in arrival order
- **Priority-Based**: Prioritize vehicles by energy deficit and departure time
- **Load Balancing**: Distribute power evenly among vehicles

#### **Storage Agents**
- **Peak Shaving**: Discharge during high load, charge during low load
- **Arbitrage**: Charge during low-price hours, discharge during high-price hours
- **Grid Support**: Support grid frequency stability

### RL Agent Architecture

The simulation supports complete decoupling of RL agents through a standardized gym-like interface:

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL Agent Services                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Pricing    │  │  Charging   │  │   Storage   │              │
│  │  Service    │  │  Service    │  │  Service    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│         │                 │                │                    │
│         ▼                 ▼                ▼                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Pricing    │  │  Charging   │  │   Storage   │              │
│  │  Agent      │  │  Agent      │  │   Agent     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Supported RL Libraries

The framework supports integration with any gym-compatible RL library:

- **Stable Baselines3**: SAC, PPO, DQN, A2C, TD3
- **RLlib**: Distributed training, hyperparameter tuning
- **Custom Agents**: Any agent implementing the gym interface
- **Vectorized Environments**: Support for parallel training

### Key Design Principles

1. **Separation of Concerns**: RL logic is completely separated from simulation logic
2. **Standardized Interfaces**: All agents conform to gym-compatible interfaces
3. **Modularity**: Each service (pricing, charging, storage) is independent
4. **Extensibility**: Easy to add new RL algorithms or modify existing ones
5. **Scalability**: Support for distributed training and vectorized environments
6. **Agent-First Design**: All decisions go through agents, no hardcoded logic
7. **Backward Compatibility**: Existing algorithms are preserved and wrapped as agents
8. **Comprehensive Tracking**: Every decision is logged and can be monitored

## 📦 Installation

This project uses [`uv`](https://github.com/astral-sh/uv), a modern and ultra-fast Python package manager compatible with pip.

### 1. Install `uv`

If you don't have `uv` installed, run:

### Installation Steps

```bash
# Step 1: Install uv
pip install uv

# Step 2: Create a virtual environment
python -m venv .venv

# Step 3: Activate the environment
source .venv/bin/activate        # On macOS/Linux
# or
.venv\Scripts\activate           # On Windows

# Step 4: Install dependencies
uv pip install -r requirements.uv.txt
```

## 🚀 Quick Start

### Running the Simulation

The main entry point for the simulation is `main.py`. You can run it with different configuration files:

```bash
# Run with default configuration
python main.py resources/configuration/ini_files/app-remote.ini

# Run with custom configuration
python main.py path/to/your/config.ini
```

### Configuration Files

The simulation uses INI configuration files to set parameters for:
- Environment settings (seasons, duration, facility size)
- Infrastructure (chargers, grid capacity, storage, PV)
- Agent types and strategies
- Logging and monitoring options

### Example Configuration

```ini
[AGENT_DECISION_SYSTEM]
enabled = True
pricing_agent_type = RULE_BASED
charging_agent_type = HEURISTIC
enable_hyperparameter_tuning = False

[SETTINGS]
log_level = INFO
facility_size = 200
```

## 📚 Documentation

- **[Agent Decision System Guide](docs/agent_decision_system.md)**: Comprehensive guide to the new agent system
- **[Decision Request System](docs/decision_request_system.md)**: Details about the underlying decision tracking system
- **[Algorithm Agents](utilities/rl_agents/algorithm_agents.py)**: Documentation of algorithm agents
- **[Rule-Based Agents](utilities/rl_agents/rule_based_agents.py)**: Documentation of rule-based agents

## 🔧 Examples

- **[Agent Decision System Example](examples/agent_decision_system_example.py)**: Basic usage examples
- **[Algorithm Agents Example](examples/algorithm_agents_example.py)**: Using algorithm agents
- **[Decision Request Example](examples/decision_request_example.py)**: Decision tracking examples

## 🎯 Key Benefits

### 1. **Consistency**
- All decisions follow the same pattern
- Standardized interfaces and data structures
- Consistent error handling and logging

### 2. **Modularity**
- Easy to add new agent types
- Simple to switch between different strategies
- Clear separation of concerns

### 3. **Testability**
- Agents can be tested independently
- Mock agents for unit testing
- Easy to compare different strategies

### 4. **Observability**
- Every decision is tracked and logged
- Performance metrics for all agents
- Decision history for analysis

### 5. **Flexibility**
- Support for multiple agent types
- Easy to implement new strategies
- Can mix different agent types

### 6. **Maintainability**
- Clear agent interfaces
- Well-documented decision types
- Easy to understand and modify

### 7. **Backward Compatibility**
- Existing algorithms are preserved as agents
- No need to rewrite existing code
- Gradual migration path

## 🔄 Migration Guide

To migrate existing code to use the agent decision system:

1. **Identify Decision Points**: Find all places where decisions are made
2. **Create Agents**: Implement agents for each decision type
3. **Register Agents**: Register agents with the system
4. **Replace Decision Logic**: Replace hardcoded logic with agent calls
5. **Test and Monitor**: Verify behavior and monitor performance

### Migration from Existing Algorithms

```python
# Before: Direct algorithm call
first_come_first_served(
    env=env,
    connected_vehicles=vehicles,
    charging_stations=charging_stations,
    charging_capacity=500,
    free_grid_capacity=300,
    planning_period_length=15
)

# After: Using algorithm agent
charging_agent = AlgorithmChargingAgent(algorithm="first_come_first_served")
context = {
    "env": env,
    "charging_stations": charging_stations,
    "charging_capacity": 500,
    "free_grid_capacity": 300,
    "planning_period_length": 15
}
decision = charging_agent.select_action(vehicles, context)
```

## 🚀 Future Enhancements

- **Multi-Agent Coordination**: Agents that can coordinate with each other
- **Adaptive Agents**: Agents that can switch strategies based on performance
- **Distributed Agents**: Support for distributed agent deployment
- **Advanced Analytics**: More sophisticated performance analysis
- **Agent Marketplace**: Repository of pre-built agents for common use cases
- **Algorithm Performance Comparison**: Tools to compare different algorithms
- **Hybrid Agents**: Agents that combine multiple strategies

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.