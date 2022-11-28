import os
import pandas as pd
import matplotlib.pyplot as plt
from class_algorithm import GeneticAlgorithm


def main_algorith(sols, weight, count_items, graphing=False):
    # Algorithm
    alg_gen = GeneticAlgorithm(sols, weight, count_items)
    new_seed = alg_gen.gen_seed()
    best_fitness_overall = 0

    if os.path.exists("log_file.csv"):
        os.remove('log_file.csv')

    for i in range(1000):
        # 1. Restart with updated steed
        seed = new_seed

        # 2. Put through fitness function to determine fitness
        fitness, solution_array = alg_gen.test_solutions(seed)

        # 3. Select the best solutions
        best_fitness, best_sols = alg_gen.order_fitness(fitness, solution_array)

        # 4. Crossover solutions
        crossed_sols = alg_gen.crossover_function(best_sols)

        # 5. Mutate solutions
        new_seed = alg_gen.mutation(crossed_sols)
        if best_fitness_overall < best_fitness[0]:
            best_fitness_overall = best_fitness[0]

        print(f"Iteration: {i} - Best Fitness: {best_fitness[0]} - Record Fitness: {best_fitness_overall}")
        GeneticAlgorithm.log_info(i, best_fitness[0], best_fitness_overall)

    if graphing:
        data = pd.read_csv("log_file.csv", header=None, names=["Iteration", "Fitness", "Best Fitness"])
        plt.plot(data["Fitness"])
        plt.plot(data["Best Fitness"])
        plt.legend(["Value", "Best Value"])
        plt.show()