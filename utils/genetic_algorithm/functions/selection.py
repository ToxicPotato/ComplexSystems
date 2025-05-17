import random

# THIS FUNCTION SELECTS THE TOP FRACTION OF INDIVIDUALS AS ELITES
# population: LIST OF RULE INDICES
# fitness_values: LIST OF FITNESS VALUES FOR THE POPULATION
# elite_fraction: FRACTION OF POPULATION TO SELECT AS ELITES
def select_elites(population, fitness_values, elite_fraction):
    number_of_elites = int(len(population) * elite_fraction)
    if number_of_elites < 1:
        number_of_elites = 1

    # FIND THE BEST INDIVIDUALS
    sorted_population = []
    sorted_fitnesses = []
    # MAKE COPIES TO AVOID CHANGING THE ORIGINAL LISTS
    population_copy = population[:]
    fitness_copy = fitness_values[:]

    # REPEATEDLY FIND AND REMOVE THE BEST INDIVIDUAL
    for i in range(number_of_elites):
        highest_fitness = -1
        index_of_best = 0
        for j in range(len(fitness_copy)):
            if fitness_copy[j] > highest_fitness:
                highest_fitness = fitness_copy[j]
                index_of_best = j
        sorted_population.append(population_copy[index_of_best])
        sorted_fitnesses.append(fitness_copy[index_of_best])
        # REMOVE SO IT IS NOT SELECTED AGAIN
        fitness_copy.pop(index_of_best)
        population_copy.pop(index_of_best)
    return sorted_population

# THIS FUNCTION SELECTS ONE INDIVIDUAL FROM THE POPULATION USING TOURNAMENT SELECTION
# population: LIST OF RULE INDICES
# fitness_values: LIST OF FITNESS VALUES
# tournament_size: NUMBER OF INDIVIDUALS IN THE TOURNAMENT (DEFAULT 3)
def tournament_selection(population, fitness_values, tournament_size=3):
    group_indices = []
    group = []
    for i in range(tournament_size):
        random_index = random.randint(0, len(population) - 1)
        group_indices.append(random_index)
        group.append((population[random_index], fitness_values[random_index]))
    # FIND THE BEST INDIVIDUAL IN THE TOURNAMENT
    best_fitness = -1
    winner_rule_index = group[0][0]
    for candidate in group:
        if candidate[1] > best_fitness:
            best_fitness = candidate[1]
            winner_rule_index = candidate[0]
    return winner_rule_index
