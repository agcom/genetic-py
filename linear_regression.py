import random
import string
import time
import matplotlib.pyplot as plt
import numpy as np

from lib import genetics_algorithm_with_callback

random.seed(time.time())

# Generate random data points
n = 10

X = 0 + np.random.rand(n, 1) * 10
y = (X @ np.array([3]).reshape((-1, 1))) + 32 + np.random.randn(n, 1)

integer_points_per_variable = 5
decimal_points_per_variable = 4
genes_per_coefficient = integer_points_per_variable + decimal_points_per_variable + 1  # +1 for sign (+ or -)

coefficients_per_chromosome = X.shape[1] + 1  # Variables + Bias
genes_per_chromosome = genes_per_coefficient * coefficients_per_chromosome


def variable_coefficient_chromosome_representation(coefficient_chromosome):
    x = float(
        ''.join(coefficient_chromosome[1:len(coefficient_chromosome) - decimal_points_per_variable]) + '.' + ''.join(
            coefficient_chromosome[len(coefficient_chromosome) - decimal_points_per_variable:]))
    if int(coefficient_chromosome[0]) > 5:
        x = -x
    return x


def variable_chromosome_representation(chromosome):
    coefficients_chromosomes = [
        chromosome[coefficientIndex * genes_per_coefficient:(coefficientIndex + 1) * genes_per_coefficient]
        for coefficientIndex in range(coefficients_per_chromosome)]

    return [variable_coefficient_chromosome_representation(coefficient_chromosome)
            for coefficient_chromosome in coefficients_chromosomes]


def chromosome_representation(chromosome):
    return variable_chromosome_representation(chromosome)


def fitness_function(chromosome):
    coefficients = variable_chromosome_representation(chromosome)

    error = 0
    for i, x in enumerate(X):
        y_val = y[i]
        error += abs(((np.dot(x, coefficients[:-1]) + coefficients[-1]) - y_val)).item()

    return error


best_found_chromosome = None


def callback(generation, best_fitness, best_chromosome):
    global best_found_chromosome

    if best_found_chromosome is None or best_fitness <= best_found_chromosome[1]:
        best_found_chromosome = (best_chromosome, best_fitness)

    print(f"Generation {generation}:")
    print(f"\tBest Fitness = {best_fitness}")
    print(f"\tBest Chromosome = {best_chromosome}")

    print(f"\tBest Chromosome So far = {best_found_chromosome[0]}")
    print(f"\tBest Fitness So far = {best_found_chromosome[1]}")


genetics_algorithm_with_callback(
    generations=1000,
    chromosome_length=genes_per_chromosome,
    tournament_size=2,
    population_size=100,
    mutation_rate=0.1,
    fitness_function=fitness_function,
    alphabet=string.digits,
    chromosome_representation=chromosome_representation,
    callback=callback
)

plt.scatter(X, y, alpha=0.7)
plt.show()