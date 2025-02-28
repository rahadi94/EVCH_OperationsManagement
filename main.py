# Executes full simulation routine
import os
import random
import sys
import time
from configparser import ConfigParser
from pathlib import Path
import numpy as np
import pandas as pd

# from Environment.Initialisation import sim_data
import simpy
from Environment.helper.configuration.configuration import Configuration
from Environment.log import lg
from Environment.model import EVCC_Sim_Model
from Utilities.sim_input_processing import sample_training_and_test_weeks
from Utilities.sim_input_processing import sample_week

# from Policy_Gradient_Methods_master.sac.sac2018 import SACAgent
# Change working directory to path of run.py
working_dir = Path(__file__).parent
os.chdir(working_dir)
# config = Configuration.instance()

# Read args
parser_main = ConfigParser()
parser_main.read(sys.argv[1])
start_time = time.time()

# CONFIG_DATA
DATA_PATH_WS = parser_main.get("SETTINGS", "raw_input_path")
CACHE_PATH_WS = parser_main.get("SETTINGS", "caching_path")
OUTPUT_DATA_PATH = parser_main.get("SETTINGS", "raw_output_save_path")
OUTPUT_VIZ_PATH = parser_main.get("SETTINGS", "visuals_save_path")

TRAIN_WEEKS, TEST_WEEKS = sample_training_and_test_weeks(seed=None)
SIM_SEASON = parser_main.get("ENVIRONMENT", "sim_season").split(",")
SUMMER_START = parser_main.get("ENVIRONMENT", "summer_start_date")
SUMMER_END = parser_main.get("ENVIRONMENT", "summer_end_date")

SIM_START_DAY = sample_week(
    sim_seasons=SIM_SEASON, summer_start=SUMMER_START, summer_end=SUMMER_END, seed=42
)
SIM_DURATION = parser_main.getint("ENVIRONMENT", "sim_duration")

SIM_TIME = SIM_DURATION * 24 * 60
DAY_TYPES = parser_main.get("ENVIRONMENT", "day_types").split(",")
POST_FIX = parser_main.get("ENVIRONMENT", "post_fix")
OBJECTIVE = parser_main.get("ENVIRONMENT", "objective")

EV_SHARES = parser_main.get("REQUESTS", "ev_share").split(",")
EV_SHARES = [float(x) / 100 for x in EV_SHARES]
REGION = parser_main.get("REQUESTS", "region")
FACILITY = [parser_main.get("REQUESTS", "facility")]
LIMIT_DAILY_REQUESTS_YN = parser_main.getboolean("REQUESTS", "limit_daily_requests_yn")
CHARGING_DEMAND_APPROACH = parser_main.get("REQUESTS", "demand_gen_approach")
NUM_PARKING_SPOTS = parser_main.getint("INFRASTRUCTURE", "parking_capa")
if Configuration.instance().facility_size:
    NUM_PARKING_SPOTS = Configuration.instance().facility_size

TRANSFORMER_NUM = parser_main.getint("INFRASTRUCTURE", "num_transformer")
CHARGER_NUM = parser_main.get("INFRASTRUCTURE", "num_charger").split(",")
CHARGER_NUM = [int(x) for x in CHARGER_NUM]
CHARGERS = {
    "fast_one": CHARGER_NUM[0],
    "fast_two": CHARGER_NUM[1],
    "fast_four": CHARGER_NUM[2],
    "slow_one": CHARGER_NUM[3],
    "slow_two": CHARGER_NUM[4],
    "slow_four": CHARGER_NUM[5],
}
MAX_NUM_CONNECTORS = parser_main.getint("INFRASTRUCTURE", "num_connector")
CHARGER_CAPA_FAST = parser_main.getint("INFRASTRUCTURE", "charger_power_fast")
CHARGER_CAPA_SLOW = parser_main.getint("INFRASTRUCTURE", "charger_power_slow")
CHARGER_CAPA = {"fast": CHARGER_CAPA_FAST, "slow": CHARGER_CAPA_SLOW}
GRID_CAPA_CURRENT = parser_main.getint("INFRASTRUCTURE", "grid_capa")
GRID_CAPA = GRID_CAPA_CURRENT + TRANSFORMER_NUM * 200
PV_INSTALLED_CAPA = parser_main.getint("INFRASTRUCTURE", "installed_capa_PV")
STORAGE_SIZE = parser_main.getint("INFRASTRUCTURE", "installed_storage")
MIN_BASELOAD = parser_main.getint("INFRASTRUCTURE", "min_facility_baseload")
MAX_BASELOAD = parser_main.getint("INFRASTRUCTURE", "max_facility_baseload")

ROUTING_ALGO = parser_main.get("OPERATOR", "routing_algo")
CHARGING_ALGO = parser_main.get("OPERATOR", "charging_algo")
if Configuration.instance().charging_algorithm:
    CHARGING_ALGO = Configuration.instance().charging_algorithm

STORAGE_ALGO = parser_main.get("OPERATOR", "storage_algo")
SCHEDULING_MODE = parser_main.get("OPERATOR", "scheduling_mode")
SERVICE_LEVEL = parser_main.getfloat("OPERATOR", "service_level")
MINIMUM_SERVED_DEMAND = parser_main.getfloat("OPERATOR", "minimum_served_demand")
PENALTY_FOR_MISSED_KWH = parser_main.getfloat("OPERATOR", "penalty_for_missed_kWh")
PLANNING_INTERVAL = parser_main.getint("OPERATOR", "planning_interval")
OPT_PERIOD_LENGTH = parser_main.getint("OPERATOR", "optimization_period_length")
LOOKAHEAD = parser_main.getint("OPERATOR", "num_lookahead_planning_periods")
LOOKBACK = 24 * 60

MAINTENANCE_COST = parser_main.getfloat("CAPEX", "maintenance_cost")
ELECTRICITY_TARIFF = parser_main.get("OPEX", "hourly_energy_costs").split(",")
ELECTRICITY_TARIFF = [int(x) / 100 for x in ELECTRICITY_TARIFF]
PEAK_COST = parser_main.getfloat("OPEX", "monthly_peak_cost")
CONNECTOR_COST_STANDARD = parser_main.getint("CAPEX", "connector_cost_standard")
CONNECTOR_COST_FAST = parser_main.getint("CAPEX", "connector_cost_fast")
GRID_COSTS = parser_main.get("CAPEX", "grid_expansion_cost").split(",")
GRID_COSTS = [int(x) for x in GRID_COSTS]
TRANSFORMER_COSTS = parser_main.get("CAPEX", "transformer_cost").split(",")
TRANSFORMER_COSTS = [int(x) for x in TRANSFORMER_COSTS]
PV_COSTS = parser_main.get("CAPEX", "pv_cost").split(",")
PV_COSTS = [int(x) for x in PV_COSTS]
BATTERY_COSTS = parser_main.get("CAPEX", "battery_cost").split(",")
BATTERY_COSTS = [int(x) for x in BATTERY_COSTS]
CHARGER_COSTS_STANDARD_ONE = parser_main.get(
    "CAPEX", "charger_cost_standard_one"
).split(",")
CHARGER_COSTS_STANDARD_ONE = [int(x) for x in CHARGER_COSTS_STANDARD_ONE]
CHARGER_COSTS_STANDARD_TWO = [
    x + CONNECTOR_COST_STANDARD for x in CHARGER_COSTS_STANDARD_ONE
]
CHARGER_COSTS_STANDARD_FOUR = [
    x + CONNECTOR_COST_STANDARD * 3 for x in CHARGER_COSTS_STANDARD_ONE
]
CHARGER_COSTS_FAST_ONE = parser_main.get("CAPEX", "charger_cost_fast_one").split(",")
CHARGER_COSTS_FAST_ONE = [int(x) for x in CHARGER_COSTS_FAST_ONE]
CHARGER_COSTS_FAST_TWO = [x + CONNECTOR_COST_FAST for x in CHARGER_COSTS_FAST_ONE]
CHARGER_COSTS_FAST_FOUR = [x + CONNECTOR_COST_FAST * 3 for x in CHARGER_COSTS_FAST_ONE]

for i in range(1, len(ELECTRICITY_TARIFF)):
    ELECTRICITY_TARIFF[i] = float(ELECTRICITY_TARIFF[i])
parameters = {}
parameters['post_fix'] = POST_FIX
parameters['summer_end_date'] = SUMMER_END
parameters['summer_start_date'] = SUMMER_START
parameters['sim_season'] = SIM_SEASON
Configuration.instance().set_parameters_from_main_file(parameters)
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

def run_experiments():
    number_of_chargers = 200
    PV_CAPA = Configuration.instance().PV
    STORAGE_CAPA = 0
    max_cap = 50
    max_grid_usage = 2000
    TRANSFORMER_NUM = Configuration.instance().grid
    START = sample_week(
        sim_seasons=SIM_SEASON,
        summer_start=SUMMER_START,
        summer_end=SUMMER_END,
        seed=42,
    )
    print(START)
    # week = random.sample(TRAIN_WEEKS, 1)
    # week = START
    results = None
    # results = [f"{POST_FIX}", f"state{9}", f"week{1}"]
    chargers = {
        "fast_one": number_of_chargers,
        "fast_two": 0,
        "fast_four": 0,
        "slow_one": 0,
        "slow_two": 0,
        "slow_four": 0,
    }


    df = run_single_simulation(
        charging_agent=None,
        storage_agent=None,
        pricing_agent=None,
        num_charger=chargers,
        turn_off_monitoring=False,
        turn_on_results=results,
        turn_on_plotting=True,
        transformer_num=TRANSFORMER_NUM,
        storage_capa=STORAGE_CAPA,
        pv_capa=PV_CAPA,
        year=9,
        start_day=START,
    )
    return df

# run_experiments()