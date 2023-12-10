import random
import string
import time

import matplotlib.pyplot as plt
import numpy as np

from lib import genetics_algorithm_with_callback

random.seed(time.time())

# Generate random data points.

n = 10
dimension = 2
min_x, max_x = 0, 10

X = min_x + np.random.rand(n, 1) * max_x

# y = 3x + 32
target_coefficients = [3]
target_bias = 32

# The last expression (np.random.randn(n, 1)) is noise.
y = (X @ np.array(target_coefficients).reshape(-1, 1)) + target_bias + np.random.randn(n, 1)

plt.scatter(X, y)
plt.show()

# Chromosome parameters

# Example coefficient chromosome: "0123451234" -> +12345.1234
integer_points_per_variable = 5
decimal_points_per_variable = 4
genes_per_coefficient = integer_points_per_variable + decimal_points_per_variable + 1  # +1 for sign (+ or -)

# Example chromosome (concat of some coefficient chromosomes):
# "01234512346987654321" -> [+12345.1234, -98765.4321]
coefficients_per_chromosome = X.shape[1] + 1  # Variables + Bias
genes_per_chromosome = genes_per_coefficient * coefficients_per_chromosome


def coefficient_chromosome_representation(coefficient_chromosome):
    x = float(
        ''.join(coefficient_chromosome[1:len(coefficient_chromosome) - decimal_points_per_variable]) +
        '.' + ''.join(coefficient_chromosome[len(coefficient_chromosome) - decimal_points_per_variable:])
    )

    if int(coefficient_chromosome[0]) > 5:
        x = -x

    return x


def chromosome_representation(chromosome):
    coefficients_chromosomes = [
        chromosome[coefficientIndex * genes_per_coefficient:(coefficientIndex + 1) * genes_per_coefficient]
        for coefficientIndex in range(coefficients_per_chromosome)
    ]

    return [
        coefficient_chromosome_representation(coefficient_chromosome)
        for coefficient_chromosome in coefficients_chromosomes
    ]


def fitness_function(chromosome):
    coefficients = chromosome_representation(chromosome)

    error = 0
    for x, y_val in zip(X, y):
        error += abs(((np.dot(x, coefficients[:-1]) + coefficients[-1]) - y_val)).item()

    return error


best_chromosome_so_far = None
best_chromosome_so_far_fitness = None


def callback(generation, best_fitness, best_chromosome):
    global best_chromosome_so_far, best_chromosome_so_far_fitness

    if best_chromosome_so_far is None or best_fitness <= best_chromosome_so_far_fitness:
        best_chromosome_so_far = best_chromosome
        best_chromosome_so_far_fitness = best_fitness

    print(f"Generation {generation + 1}:")
    print(f"\tBest Fitness = {best_fitness}")
    print(f"\tBest Chromosome = {chromosome_representation(best_chromosome)}")

    print(f"\tBest Chromosome So far = {chromosome_representation(best_chromosome_so_far)}")
    print(f"\tBest Fitness So far = {best_chromosome_so_far_fitness}")


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
