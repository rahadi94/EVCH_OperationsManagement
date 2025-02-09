import numpy as np
from geneticalgorithm2 import geneticalgorithm2 as ga
from Environment.log import stream_handler
import logging
import pandas as pd
from main import run_single_simulation
from OptimizationTestFunctions import Eggholder
import matplotlib.pyplot as plt
import seaborn as sns

stream_handler.setLevel(logging.ERROR)
# new_population = numpy.random.random_integers(low=30, high=300, size=(20,3))
b = np.load("eggholder_lastgen.npz")
print(b.files)
try:
    experience = pd.read_csv("experience.csv").drop("Unnamed: 0", axis=1)
    experience.columns = range(91)
except:
    experience = pd.DataFrame(columns=range(91))
experience_sample = False


def f(i):
    i = [int(numeric_string) for numeric_string in i]
    print(i)
    NUM_CHARGER_FINAL = 0
    TRANSFORMER_NUM_FINAL = 0
    PV_CAPA_FINAL = 0
    STORAGE_CAPA_FINAL = 0
    for j in range(10):
        for k in range(6):
            NUM_CHARGER_FINAL += i[j * 9 + k]
        TRANSFORMER_NUM_FINAL += i[j * 9 + 6]
        PV_CAPA_FINAL += i[j * 9 + 7]
        STORAGE_CAPA_FINAL += i[j * 9 + 8]
    if NUM_CHARGER_FINAL > 200:
        return +(NUM_CHARGER_FINAL - 200) * 10000000
    if TRANSFORMER_NUM_FINAL > 4:
        return +(TRANSFORMER_NUM_FINAL - 4) * 10000000
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
        objective += run_single_simulation(
            num_charger=NUM_CHARGER,
            transformer_num=TRANSFORMER_NUM,
            pv_capa=PV_CAPA,
            storage_capa=STORAGE_CAPA,
            turn_on_results_plotting=False,
            turn_off_monitoring=True,
            year=j,
        )[0]
    print(objective)
    new_experience[90] = objective
    new_experience.columns = range(91)
    experience = experience.append(new_experience)
    # print(experience)
    experience.to_csv("experience.csv")
    return objective


varbound = np.array(
    [[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 1], [0, 3], [0, 5]] * 10
)

algorithm_param = {
    "max_num_iteration": 300,
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
    print("True")
    df = pd.read_csv("experience.csv")
    samples = (
        df.sort_values("20").head(20).drop(["20", "Unnamed: 0"], axis=1).to_numpy()
    )
    scores = df.sort_values("20").head(20)["20"].to_numpy()
egg = Eggholder((90 * 50))
filename = "eggholder_lastgen.npz"
model.run(start_generation=filename, save_last_generation_as=filename)
# model.run(start_generation={'variables':None, 'scores': None}, save_last_generation_as=filename)
# model.run(save_last_generation_as=filename)
report = np.array(model.report)
df = pd.read_csv("experience.csv")
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
sns.lineplot(data=df, x=df.index, y="90", ax=ax)
plt.savefig("experience.pdf")
