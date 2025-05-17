# THIS FILE CONTAINS THE MAIN GENETIC ALGORITHM LOOP FOR EVOLVING CA RULES

from .functions.initialization import initialize_population
from .functions.fitness_function import evaluate_rule
from .functions.selection import select_elites, tournament_selection
from .functions.crossover import crossover
from .functions.mutation import mutate

# THIS FUNCTION RUNS THE GENETIC ALGORITHM AND RETURNS THE BEST RULE FOUND
# population_size: HOW MANY RULES IN EACH GENERATION
# generations: NUMBER OF GENERATIONS TO RUN
# elite_fraction: FRACTION OF POPULATION TO KEEP AS ELITES
# mutation_rate: CHANCE TO FLIP EACH BIT DURING MUTATION
def genetic_algorithm(population_size=64, generations=10, elite_fraction=0.1, mutation_rate=0.02):
    # CREATE THE INITIAL POPULATION OF RANDOM RULES
    population = initialize_population(population_size)

    # EVOLVE THE POPULATION FOR THE SPECIFIED NUMBER OF GENERATIONS
    for generation_number in range(generations):
        # CALCULATE THE FITNESS VALUE FOR EACH INDIVIDUAL IN THE POPULATION
        fitness_values = []
        for individual in population:
            individual_fitness = evaluate_rule(individual)
            fitness_values.append(individual_fitness)

        # SELECT THE BEST INDIVIDUALS TO BE ELITES
        elites = select_elites(population, fitness_values, elite_fraction)

        # PRINT THE BEST RULE AND FITNESS FOR THIS GENERATION
        highest_fitness = -1
        index_of_best = 0
        for i in range(len(fitness_values)):
            if fitness_values[i] > highest_fitness:
                highest_fitness = fitness_values[i]
                index_of_best = i
        best_rule = population[index_of_best]
        print("Gen", generation_number, "· Best rule", best_rule, "· Fitness", round(highest_fitness, 1))

        # BUILD THE NEXT POPULATION BY COPYING ELITES AND ADDING CHILDREN
        next_population = []
        for elite in elites:
            next_population.append(elite)
        while len(next_population) < population_size:
            parent_one = tournament_selection(population, fitness_values)
            parent_two = tournament_selection(population, fitness_values)
            child = crossover(parent_one, parent_two)
            mutated_child = mutate(child, mutation_rate)
            next_population.append(mutated_child)

        population = next_population

    # AFTER THE LAST GENERATION, FIND THE BEST INDIVIDUAL
    final_fitness_values = []
    for individual in population:
        individual_fitness = evaluate_rule(individual)
        final_fitness_values.append(individual_fitness)
    highest_fitness = -1
    index_of_best = 0
    for i in range(len(final_fitness_values)):
        if final_fitness_values[i] > highest_fitness:
            highest_fitness = final_fitness_values[i]
            index_of_best = i
    winner_rule = population[index_of_best]
    return winner_rule
