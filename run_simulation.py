# Executes full simulation routine
import os
import sys
import time
from configparser import ConfigParser
from pathlib import Path
import pandas as pd
import simpy
from Environment.helper.configuration.configuration import Configuration
from Environment.log import lg
from Environment.model import EVCC_Sim_Model
from Utilities.sim_input_processing import sample_week



working_dir = Path(__file__).parent
os.chdir(working_dir)
# config = Configuration.instance()

start_time = time.time()

# Read args
parser_main = ConfigParser()
parser_main.read(sys.argv[1])
# CONFIG_DATA
DATA_PATH_WS = Configuration.instance().DATA_PATH_WS
CACHE_PATH_WS = Configuration.instance().CACHE_PATH_WS
OUTPUT_DATA_PATH = Configuration.instance().OUTPUT_DATA_PATH
OUTPUT_VIZ_PATH = Configuration.instance().OUTPUT_VIZ_PATH

TRAIN_WEEKS, TEST_WEEKS = Configuration.instance().TRAIN_WEEKS, Configuration.instance().TEST_WEEKS
SIM_SEASON = Configuration.instance().SIM_SEASON
SUMMER_START = Configuration.instance().SUMMER_START
SUMMER_END = Configuration.instance().SUMMER_END
SIM_START_DAY = Configuration.instance().SIM_START_DAY
SIM_DURATION = Configuration.instance().SIM_DURATION

SIM_TIME = Configuration.instance().SIM_TIME
DAY_TYPES = Configuration.instance().DAY_TYPES
POST_FIX = Configuration.instance().POST_FIX
OBJECTIVE = Configuration.instance().OBJECTIVE

EV_SHARES = Configuration.instance().EV_SHARES
REGION = Configuration.instance().REGION
FACILITY = Configuration.instance().FACILITY
LIMIT_DAILY_REQUESTS_YN = Configuration.instance().LIMIT_DAILY_REQUESTS_YN
CHARGING_DEMAND_APPROACH = Configuration.instance().CHARGING_DEMAND_APPROACH
NUM_PARKING_SPOTS = Configuration.instance().NUM_PARKING_SPOTS
if Configuration.instance().facility_size:
    NUM_PARKING_SPOTS = Configuration.instance().NUM_PARKING_SPOTS

TRANSFORMER_NUM = Configuration.instance().TRANSFORMER_NUM
CHARGER_NUM = Configuration.instance().CHARGER_NUM
CHARGERS = Configuration.instance().CHARGERS
MAX_NUM_CONNECTORS = Configuration.instance().MAX_NUM_CONNECTORS
CHARGER_CAPA_FAST = Configuration.instance().CHARGER_CAPA_FAST
CHARGER_CAPA_SLOW = Configuration.instance().CHARGER_CAPA_SLOW
CHARGER_CAPA = Configuration.instance().CHARGER_CAPA
GRID_CAPA_CURRENT = Configuration.instance().GRID_CAPA_CURRENT
GRID_CAPA = Configuration.instance().GRID_CAPA
PV_INSTALLED_CAPA = Configuration.instance().PV_INSTALLED_CAPA
STORAGE_SIZE = Configuration.instance().STORAGE_SIZE
MIN_BASELOAD = Configuration.instance().MIN_BASELOAD
MAX_BASELOAD = Configuration.instance().MAX_BASELOAD

ROUTING_ALGO = Configuration.instance().ROUTING_ALGO
CHARGING_ALGO = Configuration.instance().CHARGING_ALGO
if Configuration.instance().charging_algorithm:
    CHARGING_ALGO = Configuration.instance().CHARGING_ALGO

STORAGE_ALGO = Configuration.instance().STORAGE_ALGO
SCHEDULING_MODE = Configuration.instance().SCHEDULING_MODE
SERVICE_LEVEL = Configuration.instance().SERVICE_LEVEL
MINIMUM_SERVED_DEMAND = Configuration.instance().MINIMUM_SERVED_DEMAND
PENALTY_FOR_MISSED_KWH = Configuration.instance().PENALTY_FOR_MISSED_KWH
PLANNING_INTERVAL = Configuration.instance().PLANNING_INTERVAL
OPT_PERIOD_LENGTH = Configuration.instance().OPT_PERIOD_LENGTH
LOOKAHEAD = Configuration.instance().LOOKAHEAD
LOOKBACK = Configuration.instance().LOOKBACK

MAINTENANCE_COST = Configuration.instance().MAINTENANCE_COST
ELECTRICITY_TARIFF = Configuration.instance().ELECTRICITY_TARIFF
PEAK_COST = Configuration.instance().PEAK_COST
CONNECTOR_COST_STANDARD = Configuration.instance().CONNECTOR_COST_STANDARD
CONNECTOR_COST_FAST = Configuration.instance().CONNECTOR_COST_FAST
GRID_COSTS = Configuration.instance().GRID_COSTS
TRANSFORMER_COSTS = Configuration.instance().TRANSFORMER_COSTS
PV_COSTS = Configuration.instance().PV_COSTS
BATTERY_COSTS = Configuration.instance().BATTERY_COSTS
CHARGER_COSTS_STANDARD_ONE = Configuration.instance().CHARGER_COSTS_STANDARD_ONE
CHARGER_COSTS_STANDARD_TWO = Configuration.instance().CHARGER_COSTS_STANDARD_TWO
CHARGER_COSTS_STANDARD_FOUR = Configuration.instance().CHARGER_COSTS_STANDARD_FOUR
CHARGER_COSTS_FAST_ONE = Configuration.instance().CHARGER_COSTS_FAST_ONE
CHARGER_COSTS_FAST_TWO = Configuration.instance().CHARGER_COSTS_FAST_TWO
CHARGER_COSTS_FAST_FOUR = Configuration.instance().CHARGER_COSTS_FAST_FOUR

for i in range(1, len(ELECTRICITY_TARIFF)):
    ELECTRICITY_TARIFF[i] = Configuration.instance().ELECTRICITY_TARIFF[i]

def get_cost(costs, year, horizon=5):
    if year == "single_period":
        final_cost = 0
        for i in range(20):
            final_cost += costs[i]
        return final_cost / 20 / horizon / 365
    return (costs[year * 2] + costs[year * 2 + 1]) / 2 / horizon / 365


# RUN SIMULATION
def run_single_simulation(
    charging_agent=None,
    storage_agent=None,
    pricing_agent=None,
    num_charger=CHARGERS,
    grid_capa=GRID_CAPA,
    pv_capa=PV_INSTALLED_CAPA,
    storage_capa=STORAGE_SIZE,
    transformer_num=TRANSFORMER_NUM,
    year=9,
    turn_on_results=False,
    turn_on_plotting=False,
    turn_off_monitoring=False,
    start_day=SIM_START_DAY,
):
    env = simpy.Environment()  # Creating the simpy environment
    # creating a model object
    CHARGER_COSTS = {
        "fast_one": get_cost(CHARGER_COSTS_FAST_ONE, year),
        "fast_two": get_cost(CHARGER_COSTS_FAST_TWO, year),
        "fast_four": get_cost(CHARGER_COSTS_FAST_FOUR, year),
        "slow_one": get_cost(CHARGER_COSTS_STANDARD_ONE, year),
        "slow_two": get_cost(CHARGER_COSTS_STANDARD_TWO, year),
        "slow_four": get_cost(CHARGER_COSTS_STANDARD_FOUR, year),
    }

    COSTS = dict(
        charger=CHARGER_COSTS,
        connector=CONNECTOR_COST_STANDARD,
        grid=get_cost(GRID_COSTS, year, 20),
        pv=get_cost(PV_COSTS, year, 20),
        battery=get_cost(BATTERY_COSTS, year),
        peak=PEAK_COST / 3,
        transformer=get_cost(TRANSFORMER_COSTS, year, 20),
        maintenance=MAINTENANCE_COST,
    )
    if year == "single_period":
        year = 9
    EV_SHARE = (EV_SHARES[year * 2] + EV_SHARES[year * 2 + 1]) / 2
    model = EVCC_Sim_Model(
        env=env,
        sim_season=SIM_SEASON,
        sim_start_date=start_day,
        sim_duration=SIM_DURATION,
        day_types=DAY_TYPES,
        transformer_num=transformer_num,
        facility_list=FACILITY,
        ev_share=EV_SHARE,
        demand_gen_approach=CHARGING_DEMAND_APPROACH,
        geography=REGION,
        parking_capa=NUM_PARKING_SPOTS,
        limit_requests_to_capa=LIMIT_DAILY_REQUESTS_YN,
        grid_capa=grid_capa,
        min_facility_baseload=MIN_BASELOAD,
        max_facility_baseload=MAX_BASELOAD,
        installed_capa_PV=pv_capa,
        installed_storage=storage_capa,
        charging_capa=CHARGER_CAPA,
        charging_num=num_charger,
        connector_num=MAX_NUM_CONNECTORS,
        electricity_tariff=ELECTRICITY_TARIFF,
        prices=COSTS,
        year=year,
        planning_interval=PLANNING_INTERVAL,
        optimization_period_length=OPT_PERIOD_LENGTH,
        lookahead=LOOKAHEAD,
        lookback=LOOKBACK,
        routing_algo=ROUTING_ALGO,
        charging_algo=CHARGING_ALGO,
        storage_algo=STORAGE_ALGO,
        base_path=DATA_PATH_WS,
        cache_path=CACHE_PATH_WS,
        raw_output_save_path=OUTPUT_DATA_PATH,
        visuals_save_path=OUTPUT_VIZ_PATH,
        post_fix=POST_FIX,
        service_level=SERVICE_LEVEL,
        minimum_served_demand=MINIMUM_SERVED_DEMAND,
        penalty_for_missed_kWh=PENALTY_FOR_MISSED_KWH,
        scheduling_mode=SCHEDULING_MODE,
        planning=turn_off_monitoring,
        objective=OBJECTIVE,
        chargers_type="multiple",
        charging_agent=charging_agent,
        storage_agent=storage_agent,
        pricing_agent=pricing_agent,
    )
    # TODO: we have "planning" here again, maybe rename? Maybe planning phase?
    # run model
    lg.info("Sim Started")
    model.run()
    env.run(until=model.sim_time)

    sim_end_time = time.time()
    sim_time = round((sim_end_time - start_time) / 60, 2)
    lg.info("Sim Completed (in {} minutes)".format(sim_time))

    # calculate objective function
    model.calculate_objective_function(initial_grid_capa=400)
    lg.error(
        f"Objective function : {model.objective_function}"
    )

    if turn_on_results:
        # save results
        model.save_results(
            method=turn_on_results[0],
            year=turn_on_results[1],
            week=turn_on_results[2],
            post_fix=model.post_fix,
        )

        save_end_time = time.time()
        save_time = round((save_end_time - sim_end_time) / 60, 2)
        print("Results Saved (in {} minutes)".format(save_time))

        if turn_on_plotting is True:
            # plot results
            model.visualize_results(
                model=model,
                sim_start_date=model.sim_start_date,
                post_fix=f"_{turn_on_results[0]}_{turn_on_results[1]}_{turn_on_results[2]}_{model.post_fix}",
                visuals_save_path=model.visuals_save_path,
            )

            plot_end_time = time.time()
            plot_time = round((plot_end_time - save_end_time) / 60, 2)
            print("Results Plotted (in {} minutes)".format(plot_time))
    if model.charging_agent:
        model.charging_agent.save_models()
    if model.pricing_agent:
        model.pricing_agent.save_models()
    # model.storage_agent.save_models()
    if model.charging_agent:
        lg.error(
            f"profit = {model.charging_agent.environment.total_reward['missed']},"
            f" energy = {model.charging_agent.environment.total_reward['energy']} ,feasibility "
            f"= {model.charging_agent.environment.total_reward['feasibility']}, feasibility_storage "
            f"= {model.charging_agent.environment.total_reward['feasibility_storage']}, pricing "
            f"= {model.pricing_agent.environment.total_reward['missed']}"
        )
    if model.pricing_agent:
        lg.error(f"profit ={model.pricing_agent.environment.total_reward['missed']}")
    if model.charging_agent:
        model.charging_agent.environment.total_reward["missed"] = 0
        model.charging_agent.environment.total_reward["feasibility"] = 0
        model.charging_agent.environment.total_reward["feasibility_storage"] = 0
        model.charging_agent.environment.total_reward["energy"] = 0
    if model.pricing_agent:
        model.pricing_agent.environment.total_reward["missed"] = 0
        model.pricing_agent._critic_loss = 0
        model.pricing_agent._policy_loss = 0
    # model.storage_agent.environment.total_reward['test'] = 0
    output = pd.DataFrame(
        [
            model.objective_function,
            model.service_level,
            model.total_energy_charged,
            model.total_energy_canceled,
        ]
    ).transpose()
    output.columns = ["profit", "SQ", "energy_charged", "energy_canceled"]
    return output