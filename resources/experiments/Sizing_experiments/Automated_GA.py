import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from Environment.log import stream_handler
import logging
import pandas as pd
from main import run_single_simulation
from OptimizationTestFunctions import Eggholder

stream_handler.setLevel(logging.ERROR)
# new_population = numpy.random.random_integers(low=30, high=300, size=(20,3))
experience = pd.DataFrame(columns=[0, 1, 2, 3])
experience_sample = True


def f(i):
    print(i)
    i = [int(numeric_string) for numeric_string in i]
    NUM_CHARGER_FINAL = 0
    GRID_CAPA_FINAL = 0
    PV_CAPA_FINAL = 0
    STORAGE_CAPA_FINAL = 0
    for j in range(5):
        NUM_CHARGER_FINAL += i[j * 4 + 0]
        GRID_CAPA_FINAL += i[j * 4 + 1]
        PV_CAPA_FINAL += i[j * 4 + 2]
        STORAGE_CAPA_FINAL += i[j * 4 + 3]
    if NUM_CHARGER_FINAL > 250:
        return +(NUM_CHARGER_FINAL - 250) * 10000000
    if GRID_CAPA_FINAL > 1500:
        return +(GRID_CAPA_FINAL - 1500) * 10000000
    if PV_CAPA_FINAL > 500:
        return +(PV_CAPA_FINAL - 500) * 10000000
    if STORAGE_CAPA_FINAL > 500:
        return +(STORAGE_CAPA_FINAL - 500) * 10000000
    objective = 0
    NUM_CHARGER = 0
    GRID_CAPA = 0
    PV_CAPA = 0
    STORAGE_CAPA = 0
    new_experience = pd.DataFrame([i, i]).drop_duplicates()
    new_experience[20] = "None"
    global experience
    experience = experience.append(new_experience)
    for j in range(5):
        NUM_CHARGER += int(i[j * 4 + 0])
        GRID_CAPA += int(i[j * 4 + 1])
        PV_CAPA += int(i[j * 4 + 2])
        STORAGE_CAPA += int(i[j * 4 + 3])
        CHARGERS = {
            "fast_one": NUM_CHARGER,
            "fast_two": 0,
            "fast_four": 0,
            "slow_one": 300,
            "slow_two": 0,
            "slow_four": 0,
        }
        objective += run_single_simulation(
            num_charger=CHARGERS,
            grid_capa=GRID_CAPA,
            pv_capa=PV_CAPA,
            storage_capa=STORAGE_CAPA,
            turn_on_results_plotting=False,
            turn_off_monitoring=True,
            ev_share=(j + 1) * 0.20,
        )
    print(objective)
    experience.loc[
        (experience[0] == i[0]) & (experience[1] == i[1]) & (experience[2] == i[2]), 20
    ] = objective
    # print(experience)
    experience.to_csv("experience.csv")
    return objective


varbound = np.array(
    [
        [0, 100],
        [0, 800],
        [0, 200],
        [10, 200],
        [0, 100],
        [0, 800],
        [0, 200],
        [10, 200],
        [0, 100],
        [0, 800],
        [0, 200],
        [10, 200],
        [0, 100],
        [0, 800],
        [0, 200],
        [10, 200],
        [0, 100],
        [0, 800],
        [0, 200],
        [10, 200],
    ]
)

algorithm_param = {
    "max_num_iteration": 100,
    "population_size": 100,
    "mutation_probability": 0.5,
    "elit_ratio": 0.01,
    "crossover_probability": 0.5,
    "parents_portion": 0.5,
    "crossover_type": "uniform",
    "max_iteration_without_improv": None,
}

model = ga(
    function=f,
    dimension=20,
    variable_type="int",
    function_timeout=150000000,
    variable_boundaries=varbound,
    algorithm_parameters=algorithm_param,
)
samples = np.array(
    (300, 1200, 600, 500, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) * 100
).reshape(20, 100)
# samples += np.random.random_integers(-70,100)
if experience_sample is True:
    print("True")
    df = pd.read_csv("experience.csv")
    samples = (
        df.sort_values("20").head(20).drop(["20", "Unnamed: 0"], axis=1).to_numpy()
    )
    scores = df.sort_values("20").head(20)["20"].to_numpy()
egg = Eggholder((20 * 20))
filename = "eggholder_lastgen.npz"
# model.run(start_generation=None, save_last_generation_as=filename)
model.run(
    start_generation={"variables": samples, "scores": scores},
    save_last_generation_as=filename,
)
# model.run(save_last_generation_as=filename)
report = np.array(model.report)
