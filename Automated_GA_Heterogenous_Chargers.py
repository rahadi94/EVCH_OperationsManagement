import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from Environment.log import stream_handler
import logging
import pandas as pd
from Utilities.sim_input_processing import sample_training_and_test_weeks

TRAIN_WEEKS, TEST_WEEKS = sample_training_and_test_weeks(seed=42)
from main import run_single_simulation
from OptimizationTestFunctions import Eggholder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

start_time = datetime.now()

stream_handler.setLevel(logging.ERROR)
# new_population = numpy.random.random_integers(low=30, high=300, size=(20,3))
try:
    experience = pd.read_csv("Cache/outputs/experience.csv").drop("Unnamed: 0", axis=1)
    experience.columns = range(92)
except:
    experience = pd.DataFrame(columns=range(92))
experience_sample = False


def f(i):
    i = [int(numeric_string) for numeric_string in i]
    print(i)
    NUM_CHARGER_FINAL_TOTAL = 0
    NUM_CHARGER_FINAL = {
        "fast_1": 0,
        "fast_2": 0,
        "fast_4": 0,
        "slow_1": 0,
        "slow_2": 0,
        "slow_4": 0,
    }
    TRANSFORMER_NUM_FINAL = 0
    PV_CAPA_FINAL = 0
    STORAGE_CAPA_FINAL = 0
    N = [1, 2, 4, 1, 2, 4]
    for j in range(10):
        for k in range(6):
            NUM_CHARGER_FINAL_TOTAL += i[j * 9 + k] * N[k]
        NUM_CHARGER_FINAL["fast_1"] += i[j * 9 + 0]
        NUM_CHARGER_FINAL["fast_2"] += i[j * 9 + 1]
        NUM_CHARGER_FINAL["fast_4"] += i[j * 9 + 2]
        NUM_CHARGER_FINAL["slow_1"] += i[j * 9 + 3]
        NUM_CHARGER_FINAL["slow_2"] += i[j * 9 + 4]
        NUM_CHARGER_FINAL["slow_4"] += i[j * 9 + 5]
        TRANSFORMER_NUM_FINAL += i[j * 9 + 6]
        PV_CAPA_FINAL += i[j * 9 + 7]
        STORAGE_CAPA_FINAL += i[j * 9 + 8]
    if NUM_CHARGER_FINAL_TOTAL > 200:
        return +(NUM_CHARGER_FINAL_TOTAL - 200) * 10000000
    if NUM_CHARGER_FINAL["fast_1"] > 20:
        return +(NUM_CHARGER_FINAL_TOTAL - 20) * 10000000
    if NUM_CHARGER_FINAL["fast_2"] > 20:
        return +(NUM_CHARGER_FINAL_TOTAL - 20) * 10000000
    if NUM_CHARGER_FINAL["fast_4"] > 20:
        return +(NUM_CHARGER_FINAL_TOTAL - 20) * 10000000
    if NUM_CHARGER_FINAL["slow_1"] > 50:
        return +(NUM_CHARGER_FINAL_TOTAL - 50) * 10000000
    if NUM_CHARGER_FINAL["slow_2"] > 50:
        return +(NUM_CHARGER_FINAL_TOTAL - 50) * 10000000
    if NUM_CHARGER_FINAL["slow_4"] > 50:
        return +(NUM_CHARGER_FINAL_TOTAL - 50) * 10000000
    if TRANSFORMER_NUM_FINAL > 6:
        return +(TRANSFORMER_NUM_FINAL - 6) * 10000000
    if PV_CAPA_FINAL > 100 / 10:
        return +(PV_CAPA_FINAL - 100 / 10) * 10000000
    if STORAGE_CAPA_FINAL > 500 / 10:
        return +(STORAGE_CAPA_FINAL - 500 / 10) * 10000000
    objective = 0
    NUM_CHARGER = {
        "fast_one": 0,
        "fast_two": 0,
        "fast_four": 0,
        "slow_one": 0,
        "slow_two": 0,
        "slow_four": 0,
    }
    TRANSFORMER_NUM = 0
    PV_CAPA = 0
    STORAGE_CAPA = 0
    new_experience = pd.DataFrame([i, i]).drop_duplicates()
    global experience
    for j in range(10):
        NUM_CHARGER["fast_one"] += int(i[j * 9 + 0])
        NUM_CHARGER["fast_two"] += int(i[j * 9 + 1])
        NUM_CHARGER["fast_four"] += int(i[j * 9 + 2])
        NUM_CHARGER["slow_one"] += int(i[j * 9 + 3])
        NUM_CHARGER["slow_two"] += int(i[j * 9 + 4])
        NUM_CHARGER["slow_four"] += int(i[j * 9 + 5])
        TRANSFORMER_NUM += int(i[j * 9 + 6])
        PV_CAPA += int(i[j * 9 + 7]) * 10
        STORAGE_CAPA += int(i[j * 9 + 8]) * 10
        print(NUM_CHARGER, TRANSFORMER_NUM, PV_CAPA, STORAGE_CAPA)
        inner_objective = 0
        for week in TRAIN_WEEKS:
            inner_objective += run_single_simulation(
                num_charger=NUM_CHARGER,
                transformer_num=TRANSFORMER_NUM,
                pv_capa=PV_CAPA,
                storage_capa=STORAGE_CAPA,
                turn_on_results=False,
                turn_off_monitoring=True,
                year=j,
                start_day=week,
            )[0]
        objective += inner_objective / 12
    print(objective)
    new_experience[90] = objective
    new_experience[91] = round((datetime.now() - start_time).total_seconds() / 60)
    new_experience.columns = range(92)
    experience = experience.append(new_experience)
    # print(experience)
    experience.to_csv("Cache/outputs/experience.csv")
    return objective


varbound = np.array(
    [[0, 2], [0, 2], [0, 5], [0, 5], [0, 5], [0, 10], [0, 1], [0, 3], [0, 5]] * 10
)

algorithm_param = {
    "max_num_iteration": 3000000,
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
    dimension=90,
    variable_type="int",
    function_timeout=150000000,
    variable_boundaries=varbound,
    algorithm_parameters=algorithm_param,
)
# samples = np.array((300,1200,600,500, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0)*100).reshape(90,100)
# samples += np.random.random_integers(-70,100)
if experience_sample is True:
    df = pd.read_csv("Cache/outputs/experience.csv")
    samples = (
        df.sort_values("90").head(100).drop(["90", "Unnamed: 0"], axis=1).to_numpy()
    )
    scores = df.sort_values("90").head(100)["90"].to_numpy()
egg = Eggholder((90 * 50))
filename = "eggholder_lastgen.npz"
# model.run(start_generation=filename, save_last_generation_as=filename)
model.run(
    start_generation={"variables": None, "scores": None},
    save_last_generation_as=filename,
)
# model.run(save_last_generation_as=filename)
report = np.array(model.report)
df = pd.read_csv("Cache/outputs/experience.csv")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.lineplot(data=df, x=df.index, y="90", ax=ax)
plt.savefig("experience.pdf")
