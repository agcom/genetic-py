import random


def initialize_population(size, alphabet, chromosome_length):
    population = []
    for _ in range(size):
        chromosome = random.choices(alphabet, k=chromosome_length)
        population += [chromosome]
    return population


def tournament_selection(population, tournament_size, fitness_function):
    selected_parents = []
    for _ in range(2):
        participants = random.sample(population, tournament_size)
        best_chromosome = min(participants, key=fitness_function)
        selected_parents += [best_chromosome]
    return selected_parents[0], selected_parents[1]


def single_point_crossover(parent0, parent1):
    chromosome_length = len(parent0)
    crossover_point = random.randint(1, chromosome_length - 1)
    child0 = parent0[:crossover_point] + parent1[crossover_point:]
    child1 = parent1[:crossover_point] + parent0[crossover_point:]
    return child0, child1


def mutation(chromosome, alphabet, rate):
    for i in range(len(chromosome)):
        if random.random() < rate:
            chromosome[i] = random.choice(alphabet)
    return chromosome


def genetics_algorithm(fitness_function, generations, mutation_rate, population_size, chromosome_length, alphabet,
                       tournament_size, chromosome_representation):
    population = initialize_population(population_size, alphabet, chromosome_length)

    for generation in range(generations):
        new_population = []
        for _ in range(population_size):
            parent0, parent1 = tournament_selection(population, tournament_size, fitness_function)
            child0, child1 = single_point_crossover(parent0, parent1)
            child0, child1 = mutation(child0, alphabet, mutation_rate), mutation(child1, alphabet, mutation_rate)
            new_population += [child0] + [child1]

        population = new_population

        best_chromosome = min(population, key=fitness_function)
        best_fitness = fitness_function(best_chromosome)

        print(f"Generation {generation + 1}:")
        print(f"\tBest Fitness = {best_fitness}")
        print(f"\tBest Chromosome = {chromosome_representation(best_chromosome)}")


def genetics_algorithm_with_callback(fitness_function, generations, mutation_rate, population_size, chromosome_length,
                                     alphabet,
                                     tournament_size, chromosome_representation, callback):
    population = initialize_population(population_size, alphabet, chromosome_length)

    for generation in range(generations):
        new_population = []
        for _ in range(int(population_size / 2)):
            parent0, parent1 = tournament_selection(population, tournament_size, fitness_function)
            child0, child1 = single_point_crossover(parent0, parent1)
            child0, child1 = mutation(child0, alphabet, mutation_rate), mutation(child1, alphabet, mutation_rate)
            new_population += [child0] + [child1]

        population = new_population

        best_chromosome = min(population, key=fitness_function)
        best_fitness = fitness_function(best_chromosome)

        callback(generation, best_fitness, best_chromosome)
