import random

from Environment.log import stream_handler
from Sizing import ga
import logging
import numpy
import pandas as pd
from main import run_single_simulation
from Utilities.sim_input_processing import sample_training_and_test_weeks

TRAIN_WEEKS, TEST_WEEKS = sample_training_and_test_weeks(seed=42)
import multiprocessing as mp

stream_handler.setLevel(logging.ERROR)


p = 10
q = 2


# TODO: Adding constraints (number of connectors and number of chargers)
# TODO: Compare multiprocessing run time with single process
def fitness_function(generation, experience):
    objectives = []
    for i in generation:
        NUM_CHARGER_FINAL_TOTAL = 0
        NUM_CHARGER_FINAL = {
            "fast_1": 0,
            "fast_2": 0,
            "fast_4": 0,
            "slow_1": 0,
            "slow_2": 0,
            "slow_4": 0,
        }
        N = [1 * q, 2 * q, 4 * q, 1 * p, 2 * p, 4 * p]
        for k in range(6):
            NUM_CHARGER_FINAL_TOTAL += i[k] * N[k]
        if NUM_CHARGER_FINAL_TOTAL > 140:
            return +(NUM_CHARGER_FINAL_TOTAL - 140) * 10000000
        value = experience.loc[
            (experience[0] == i[0]) & (experience[1] == i[1]) & (experience[2] == i[2]),
            3,
        ].values[0]
        if value == "None":
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
            for j in range(1):
                NUM_CHARGER["fast_one"] += int(i[j * 9 + 0]) * q
                NUM_CHARGER["fast_two"] += int(i[j * 9 + 1]) * q
                NUM_CHARGER["fast_four"] += int(i[j * 9 + 2]) * q
                NUM_CHARGER["slow_one"] += int(i[j * 9 + 3]) * p
                NUM_CHARGER["slow_two"] += int(i[j * 9 + 4]) * p
                NUM_CHARGER["slow_four"] += int(i[j * 9 + 5]) * p
                TRANSFORMER_NUM += int(i[j * 9 + 6])
                PV_CAPA += int(i[j * 9 + 7]) * 10
                STORAGE_CAPA += int(i[j * 9 + 8]) * 50
                print(NUM_CHARGER, TRANSFORMER_NUM, PV_CAPA, STORAGE_CAPA)
                inner_objective = 0
                # You can change the number of weeks here, I guess 6 weeks are enough
                random.seed(42)
                train_weeks = random.sample(TRAIN_WEEKS, 1)
                for week in train_weeks:
                    objective += run_single_simulation(
                        num_charger=NUM_CHARGER,
                        transformer_num=TRANSFORMER_NUM,
                        pv_capa=PV_CAPA,
                        storage_capa=STORAGE_CAPA,
                        turn_on_results=False,
                        turn_off_monitoring=True,
                        year=9,
                        start_day=week,
                    )[0]
            experience.loc[
                (experience[0] == i[0])
                & (experience[1] == i[1])
                & (experience[2] == i[2]),
                3,
            ] = objective
        else:
            objective = value
    # else:
    #     objective = - 100000
    #     experience.loc[
    #         (experience[0] == i[0]) & (experience[1] == i[1]) & (experience[2] == i[2]), 3] = objective

    objectives.append(objective)
    print(objectives)
    return objectives, experience


num_weights = 9
sol_per_pop = 20
num_parents_mating = 10

# Defining the population size.
pop_size = (
    sol_per_pop,
    num_weights,
)  # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
# Creating the initial population.
first_generation = "random"
approximation = "hundreds"
if first_generation == "random":
    new_population = numpy.random.random_integers(low=30, high=300, size=pop_size)
    for i in range(sol_per_pop):
        for j in range(num_parents_mating):
            if j == 0:
                new_population[i, j] = numpy.random.random_integers(low=30, high=600)
            if j == 1:
                new_population[i, j] = numpy.random.random_integers(low=300, high=1400)
            if j == 2:
                new_population[i, j] = numpy.random.random_integers(low=0, high=500)
            # new_population.append (numpy.random.random_integers(low=30, high=300, size=pop_size)
    """Saving experience help us to avoid double calculation of a same set of inputs and access the best generation if
    the program crashes"""
    experience = pd.DataFrame(new_population.copy())
    experience[3] = "None"
if first_generation == "from_experience":
    experience = pd.read_csv("experience.csv").drop("Unnamed: 0", axis=1)
    experience.columns = [0, 1, 2, 3]
    experience = experience.loc[experience[3] != "None"]
    experience[3] = experience[3].astype("float")
    new_population = (
        experience.sort_values([3], ascending=False).head(20)[[0, 1, 2]].to_numpy()
    )
    if approximation == "hundreds":
        new_population = numpy.round(new_population / 10) * 10
        new_experience = pd.DataFrame(new_population)
        new_experience[3] = "None"
        experience = experience.append(new_experience)

print(new_population)

best_outputs = []
num_generations = 100
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    output = fitness_function(new_population, experience)
    fitness = output[0]
    experience = output[1]
    print("Fitness")
    print(fitness)
    print(experience)

    # best_outputs.append(numpy.max(numpy.sum(new_population * equation_inputs, axis=1)))
    # # The best result in the current iteration.
    # print("Best result : ", numpy.max(numpy.sum(new_population * equation_inputs, axis=1)))

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(new_population, fitness, num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(
        parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights)
    )
    if approximation == "hundreds":
        offspring_crossover = numpy.round(offspring_crossover / 10) * 10
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = ga.mutation(
        offspring_crossover,
        num_mutations=1,
        mutation_rate=max(0.5, (0.8 - generation * 0.01)),
    )

    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0 : parents.shape[0], :] = parents
    new_population[parents.shape[0] :, :] = offspring_mutation
    new_experience = pd.DataFrame(offspring_mutation)
    new_experience[3] = "None"
    experience = experience.append(new_experience)
    experience.to_csv("experience.csv")

# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = fitness_function(new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = numpy.where(fitness == numpy.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
# print("Best solution fitness : ", fitness[best_match_idx])

import matplotlib.pyplot

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
