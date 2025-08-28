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

## 📦 Installation

This project uses [`uv`](https://github.com/astral-sh/uv), a modern and ultra-fast Python package manager compatible with pip.

### 1. Install `uv`

If you don’t have `uv` installed, run:

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

## 🚀 Quick Start with RL Integration

### Basic RL Agent Integration

```python
from stable_baselines3 import SAC
from utilities.rl_environments.evch_gym_env import EVCHGymEnv, AgentType, EVCHConfig
from utilities.rl_agents.factory import build_gym_agents_controller

# 1. Create gym environment
config = EVCHConfig(
    agent_type=AgentType.PRICING,
    number_chargers=10,
    number_power_options=2,
    maximum_power=800.0
)
env = EVCHGymEnv(config)

# 2. Create and train RL agent
agent = SAC("MlpPolicy", env, verbose=1)
agent.learn(total_timesteps=10000)

# 3. Integrate with simulation
agents_controller = build_gym_agents_controller(
    pricing_agent=agent,
    pricing_config=config.__dict__
)

# 4. Use with operator
operator = Operator(..., agents_controller=agents_controller)
```

### Supported RL Libraries

- **Stable Baselines3**: `pip install stable-baselines3`
- **RLlib**: `pip install "ray[rllib]"`
- **Gymnasium**: `pip install gymnasium`

For detailed integration guides, see [docs/gym_integration_guide.md](docs/gym_integration_guide.md).