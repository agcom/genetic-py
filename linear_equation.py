import random
import string
import time

from lib import genetics_algorithm

random.seed(time.time())

integer_points_per_variable = 5
decimal_points_per_variable = 4
genes_per_variable = integer_points_per_variable + decimal_points_per_variable + 1

coefficients = [-1, 2, 3, 4]


def variable_chromosome_representation(chromosome):
    x = float(''.join(chromosome[1:len(chromosome) - decimal_points_per_variable]) + '.' + ''.join(
        chromosome[len(chromosome) - decimal_points_per_variable:]))
    if int(chromosome[0]) > 5:
        x = -x
    return x


def chromosome_representation(chromosome):
    variables_representations = []
    for i in range(len(coefficients) - 1):
        variable_chromosome = ''.join(chromosome[i * genes_per_variable:(i + 1) * genes_per_variable])
        variables_representations.append(variable_chromosome_representation(variable_chromosome))
    
    return variables_representations


def fitness_function(chromosome):
    variables = chromosome_representation(chromosome)
    
    l_eq_sum = 0
    for i, coefficient in enumerate(coefficients[:len(coefficients) - 1]):
        l_eq_sum += coefficient * variables[i]
    l_eq_sum += coefficients[-1]
    return abs(l_eq_sum)


genetics_algorithm(
    generations=1000,
    chromosome_length=genes_per_variable * (len(coefficients) - 1),
    tournament_size=2,
    population_size=100,
    mutation_rate=0.1,
    fitness_function=fitness_function,
    alphabet=string.digits,
    chromosome_representation=chromosome_representation
)
