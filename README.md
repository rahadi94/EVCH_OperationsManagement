# EVCC_Sim

This repository contains a flexible simulation framework for **Electric Vehicle (EV) Charging Clusters (EVCCs)**. 
EVCCs are large-scale EV-charging-enabled parking lots. Examples include workplace charging facilities, destination parking lots (e.g., mall, supermarket or gym parking garages) or fleet depots.

EVCCs are expected to become a core component of the future charging portfolio outweighing the importance of home charging by some estimates. Planning (sizing) and operating such EVCCs is a non-trivial task with three-way inter-dependencies between (1) user preferences, (2) infrastructure decisions and (3) operations management.

This simulation is intended to explore these interdependencies through extensive sensitivity testing and through testing new algorithms and models for sizing and operating EVCCs. The module structure is as follows:

## Module structure
The following modules are included:
- **`Preferences` Module:** Initializes vehicle objects with respective charging and parking preferences (i.e., requests) based on empirical data
- **`Infrastructure` Module:** Initializes infrastructure objects (EV supply equipment (EVSE), connectors per each EVSE, grid connection capacity, on-site storage and on-site generation (PV))  
- **`Operations` Module:** Conatain algorithms for assigning physical space (vehicle routing) and electrical capacity (vehicle charging) to individual vehicle objects based on a pre-defined charging policy
- **`Results` Module:** Monitors EVCC activity in pre-defined intervals and accounts costs. Includes plotting routines.

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