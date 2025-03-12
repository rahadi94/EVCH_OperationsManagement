# Executes full simulation routine
from Environment.helper.configuration.configuration import Configuration
from Utilities.sim_input_processing import sample_week
from run_simulation import run_single_simulation


# Change working directory to path of run.py

def run_experiments():
    number_of_chargers = 200
    PV_CAPA = Configuration.instance().PV
    STORAGE_CAPA = 0
    max_cap = 50
    max_grid_usage = 2000
    TRANSFORMER_NUM = Configuration.instance().grid
    START = sample_week(
        sim_seasons=Configuration.instance().SIM_SEASON,
        summer_start=Configuration.instance().SUMMER_START,
        summer_end=Configuration.instance().SUMMER_END,
        seed=42,
    )
    print(START)
    # week = random.sample(TRAIN_WEEKS, 1)
    # week = START
    results = None
    results = [f"{Configuration.instance().POST_FIX}", f"state{9}", f"week{1}"]
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

run_experiments()