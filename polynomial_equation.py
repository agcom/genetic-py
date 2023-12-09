import random
import string
import time

from lib import genetics_algorithm

random.seed(time.time())

integer_points = 5
decimal_points = 4


def chromosome_representation(chromosome):
    x = float(''.join(chromosome[1:len(chromosome) - decimal_points]) + '.' + ''.join(
        chromosome[len(chromosome) - decimal_points:]))
    if int(chromosome[0]) > 5:
        x = -x
    return x


coefficients = [823974, -8324, 214, 234, 22, -324]


def fitness_function(chromosome):
    x = chromosome_representation(chromosome)
    l_eq = 0
    for i, coefficient in enumerate(coefficients):
        l_eq += coefficient * (x ** (len(coefficients) - i - 1))
    return abs(l_eq)


genetics_algorithm(
    generations=2000,
    chromosome_length=integer_points + decimal_points + 1,
    tournament_size=2,
    population_size=100,
    mutation_rate=0.1,
    fitness_function=fitness_function,
    alphabet=string.digits,
    chromosome_representation=chromosome_representation
)
