# EVCH Gym Integration Guide

This guide explains how to integrate standard gym-compatible RL agents with the EVCH (Electric Vehicle Charging Hub) simulation.

## Overview

The EVCH simulation now supports complete decoupling of RL agents through a gym-like environment interface. This allows you to use any gym-compatible RL library (Stable Baselines3, RLlib, etc.) without modifying the simulation code.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RL Library    │    │   Gym Adapter    │    │   EVCH Gym      │
│  (Stable-Bas3,  │───▶│   (Standard      │───▶│   Environment   │
│   RLlib, etc.)  │    │   Interface)     │    │   (Wrapper)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────┐
                                              │   EVCH          │
                                              │   Simulation    │
                                              └─────────────────┘
```

## Key Components

### 1. EVCH Gym Environment (`EVCHGymEnv`)

A unified gym environment that wraps the existing EVCH environments:

- **Location**: `utilities/rl_environments/evch_gym_env.py`
- **Features**: Standard gym interface (`reset()`, `step()`, `render()`)
- **Agent Types**: Pricing, Charging, Storage
- **Action Spaces**: Continuous and discrete
- **Observation Spaces**: Configurable based on agent type

### 2. Gym Agent Adapters

Adapters that connect gym-compatible agents to the simulation:

- **Location**: `utilities/rl_agents/adapters/gym_agent_adapter.py`
- **Types**: `GymPricingAgentAdapter`, `GymChargingAgentAdapter`, `GymStorageAgentAdapter`
- **Interface**: Conforms to the defined agent protocols

### 3. Factory Functions

Easy creation of gym-compatible agents:

- **Location**: `utilities/rl_agents/factory.py`
- **Functions**: `build_gym_agents_controller()`, `build_gym_*_adapter()`

## Quick Start

### 1. Install Required Dependencies

```bash
# For Stable Baselines3
pip install stable-baselines3

# For RLlib
pip install "ray[rllib]"

# For other gym-compatible libraries
pip install gymnasium
```

### 2. Create a Gym Agent

```python
from stable_baselines3 import SAC
from utilities.rl_environments.evch_gym_env import EVCHGymEnv, AgentType, EVCHConfig

# Create configuration
config = EVCHConfig(
    agent_type=AgentType.PRICING,
    number_chargers=10,
    number_power_options=2,
    maximum_power=800.0,
    maximum_grid_usage=1000.0
)

# Create gym environment
env = EVCHGymEnv(config)

# Create SAC agent
agent = SAC("MlpPolicy", env, verbose=1)

# Train the agent
agent.learn(total_timesteps=10000)
```

### 3. Integrate with Simulation

```python
from utilities.rl_agents.factory import build_gym_agents_controller

# Create agents controller
agents_controller = build_gym_agents_controller(
    pricing_agent=agent,
    pricing_config=config.__dict__
)

# Use with operator
operator = Operator(
    env=sim_env,
    requests=requests,
    chargers=chargers,
    # ... other parameters ...
    agents_controller=agents_controller
)
```

## Supported RL Libraries

### Stable Baselines3

```python
from stable_baselines3 import SAC, PPO, DQN

# Pricing agent (continuous actions)
pricing_agent = SAC("MlpPolicy", pricing_env, verbose=1)

# Charging agent (continuous actions)
charging_agent = PPO("MlpPolicy", charging_env, verbose=1)

# Storage agent (discrete actions)
storage_agent = DQN("MlpPolicy", storage_env, verbose=1)
```

### RLlib

```python
import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig

# Initialize Ray
ray.init()

# Configure SAC
sac_config = SACConfig().environment(
    EVCHGymEnv, 
    env_config={"config": config}
).framework("torch")

# Train
results = tune.run("SAC", config=sac_config.to_dict())
```

### Custom Agents

```python
class CustomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def predict(self, observation, deterministic=True):
        # Your custom logic here
        action = self.action_space.sample()
        return action, None
    
    def reset(self):
        pass

# Use with adapter
custom_agent = CustomAgent(env.action_space)
adapter = create_gym_pricing_adapter(config.__dict__, custom_agent)
```

## Configuration

### Agent-Specific Configurations

```python
# Pricing agent configuration
pricing_config = {
    "agent_type": AgentType.PRICING,
    "number_chargers": 10,
    "number_power_options": 2,
    "maximum_power": 800.0,
    "pricing_mode": "Continuous",
    "dynamic_fix_term_pricing": True,
    "capacity_pricing": False
}

# Charging agent configuration
charging_config = {
    "agent_type": AgentType.CHARGING,
    "number_chargers": 15,
    "maximum_power": 1000.0,
    "maximum_grid_usage": 1200.0
}

# Storage agent configuration
storage_config = {
    "agent_type": AgentType.STORAGE,
    "dynamic_storage_scheduling": True
}
```

### Environment Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `agent_type` | Type of agent (PRICING/CHARGING/STORAGE) | Required |
| `number_chargers` | Number of charging stations | 10 |
| `number_power_options` | Number of power options | 2 |
| `maximum_power` | Maximum power limit | 800.0 |
| `maximum_grid_usage` | Maximum grid usage | 1000.0 |
| `pricing_mode` | Pricing mode (Continuous/Discrete) | "Continuous" |
| `evaluation` | Whether in evaluation mode | False |

## Training Workflow

### 1. Environment Setup

```python
# Create environments for each agent type
pricing_env = EVCHGymEnv(EVCHConfig(agent_type=AgentType.PRICING, **pricing_config))
charging_env = EVCHGymEnv(EVCHConfig(agent_type=AgentType.CHARGING, **charging_config))
storage_env = EVCHGymEnv(EVCHConfig(agent_type=AgentType.STORAGE, **storage_config))
```

### 2. Agent Training

```python
# Train pricing agent
pricing_agent = SAC("MlpPolicy", pricing_env, verbose=1)
pricing_agent.learn(total_timesteps=50000)

# Train charging agent
charging_agent = PPO("MlpPolicy", charging_env, verbose=1)
charging_agent.learn(total_timesteps=50000)

# Train storage agent
storage_agent = DQN("MlpPolicy", storage_env, verbose=1)
storage_agent.learn(total_timesteps=50000)
```

### 3. Integration

```python
# Create agents controller
agents_controller = build_gym_agents_controller(
    pricing_agent=pricing_agent,
    charging_agent=charging_agent,
    storage_agent=storage_agent,
    pricing_config=pricing_config,
    charging_config=charging_config,
    storage_config=storage_config
)

# Use in simulation
operator = Operator(..., agents_controller=agents_controller)
```

## Advanced Usage

### Custom Reward Functions

```python
class CustomEVCHGymEnv(EVCHGymEnv):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        
        # Custom reward modification
        custom_reward = reward * 1.5  # Scale reward
        if info.get("missed_requests", 0) > 0:
            custom_reward -= 10  # Penalty for missed requests
        
        return observation, custom_reward, terminated, truncated, info
```

### Multi-Agent Training

```python
# Train multiple agents simultaneously
from stable_baselines3.common.vec_env import DummyVecEnv

# Create vectorized environments
pricing_envs = DummyVecEnv([lambda: EVCHGymEnv(pricing_config) for _ in range(4)])
charging_envs = DummyVecEnv([lambda: EVCHGymEnv(charging_config) for _ in range(4)])

# Train with vectorized environments
pricing_agent = SAC("MlpPolicy", pricing_envs, verbose=1)
pricing_agent.learn(total_timesteps=100000)
```

### Hyperparameter Tuning

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000)
    
    # Create agent with suggested parameters
    agent = SAC("MlpPolicy", env, learning_rate=learning_rate, buffer_size=buffer_size)
    
    # Train and evaluate
    agent.learn(total_timesteps=10000)
    mean_reward = evaluate_agent(agent, env)
    
    return mean_reward

# Optimize hyperparameters
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required dependencies are installed
2. **Action Space Mismatch**: Check that agent action space matches environment
3. **Observation Space Mismatch**: Verify observation space compatibility
4. **Training Convergence**: Adjust hyperparameters or reward function

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create environment with debug info
env = EVCHGymEnv(config)
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
```

## Examples

See `examples/gym_agent_integration_example.py` for complete working examples with:

- Stable Baselines3 integration
- RLlib integration
- Custom agent creation
- Simulation integration

## Benefits

1. **Complete Decoupling**: RL agents are completely separate from simulation
2. **Standard Interface**: Use any gym-compatible RL library
3. **Easy Comparison**: Test different algorithms without code changes
4. **Scalability**: Support for vectorized environments and distributed training
5. **Flexibility**: Custom reward functions and hyperparameter tuning
6. **Maintainability**: Clean separation of concerns

## Next Steps

1. Install your preferred RL library
2. Create environments for your agent types
3. Train agents using standard RL workflows
4. Integrate with simulation using the provided adapters
5. Experiment with different algorithms and configurations
