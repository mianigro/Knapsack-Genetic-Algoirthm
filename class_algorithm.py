import random
from csv import writer
import numpy as np


# Class for genetic algorithm
class GeneticAlgorithm:

    def __init__(self, n_solutions, max_weight, n_items):
        # Hyperparameters
        self.n_solutions = n_solutions
        self.max_weight = max_weight
        self.n_items = n_items

        # Build knapsack
        self.item_no = np.arange(0, self.n_items)
        self.weight = np.random.randint(10, 50, size=self.n_items)
        self.value = np.random.randint(40, 100, size=self.n_items)

    # Start with empty knapsack with nothing in it
    def gen_seed(self):
        out_seed = np.zeros((self.n_solutions, self.n_items))

        return out_seed

    # Function to determine weight and value of the knapsack based on the things in it
    def goal_function(self, item_choice):
        weight = 0
        value = 0

        batch = np.random.choice(self.item_no, size=self.n_items, replace=False)
        for item in batch:
            if int(item_choice[item]) == 1 and (weight + self.weight[item]) <= self.max_weight:
                weight += self.weight[item]
                value += self.value[item]

        return weight, value

    # Fitness function that returns the value of a single knapsack as that time
    def fitness(self, item_array):
        weight, value = self.goal_function(item_array)
        return value

    # Applies fitness function to all solutions
    def test_solutions(self, sol_array):
        fitness_array = np.apply_along_axis(self.fitness, 1, sol_array)

        return fitness_array, sol_array

    # Orders the tested knapsacks based on the fitness
    def order_fitness(self, fitness_array, generation_sols):
        # Order by fitness function
        generation_sols = np.reshape(generation_sols, (len(generation_sols), self.n_items))
        index_points = np.argsort(fitness_array)[-1 * self.n_solutions:]
        index_points = np.flip(index_points)

        # Descending order
        return fitness_array[index_points], generation_sols[index_points]

    # Crosses the solutions to generate new ones
    def crossover_function(self, generation_x):
        # Make blank array
        crossed_solutions = np.zeros((len(generation_x), self.n_items))

        # Rotate top solutions up to n-200
        crossed_solutions[1:len(generation_x) - 200:2, :int(self.n_items / 2)] = generation_x[
                                                                                 :len(generation_x) - 200:2,
                                                                                 :int(self.n_items / 2)]
        crossed_solutions[:len(generation_x) - 200:2, :int(self.n_items / 2)] = generation_x[
                                                                                1:len(generation_x) - 200:2,
                                                                                :int(self.n_items / 2)]

        crossed_solutions[:len(generation_x) - 200:, int(self.n_items / 2):] = generation_x[:len(generation_x) - 200:,
                                                                               int(self.n_items / 2):]

        # Add 200 of the prev best unchanged
        crossed_solutions[len(generation_x) - 200::] = generation_x[:200:]

        return crossed_solutions

    # Mutate one item in each line of the new solutions
    def mutation(self, crossed_solutions):

        for x in range(self.n_solutions):
            line_choice = random.randint(0, self.n_items - 1)
            if crossed_solutions[x][line_choice] == 1:
                crossed_solutions[x][line_choice] = 0
            else:
                crossed_solutions[x][line_choice] = 1

        return crossed_solutions

    # Logging the information to a csv
    @staticmethod
    def log_info(iteration, fitness_out, best_fitness_out):
        new_row = [iteration, fitness_out, best_fitness_out]

        with open('log_file.csv', 'a') as log_file:
            writer_object = writer(log_file)
            writer_object.writerow(new_row)