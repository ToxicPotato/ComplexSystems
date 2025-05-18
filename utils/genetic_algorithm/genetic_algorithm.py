import os
import csv
import time
from ca_config import POPULATION_SIZE, NUMBER_OF_GENERATIONS, ELITE_PERCENTAGE, MUTATION_RATE
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
def genetic_algorithm(population_size=POPULATION_SIZE,
                      generations=NUMBER_OF_GENERATIONS,
                      elite_fraction=ELITE_PERCENTAGE,
                      mutation_rate=MUTATION_RATE):
    # Prepare GA logging
    ga_log_dir = os.path.join("results", "ga_logs")
    os.makedirs(ga_log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ga_log_path = os.path.join(ga_log_dir, f"ga_log_{timestamp}.csv")
    ga_file = open(ga_log_path, mode='w', newline='')
    ga_writer = csv.writer(ga_file)
    # Header: generation, best_rule, best_fitness, avg_fitness, worst_fitness, generation_time_ms
    ga_writer.writerow([
        "generation",
        "best_rule",
        "best_fitness",
        "avg_fitness",
        "worst_fitness",
        "generation_time_ms"
    ])

    # CREATE THE INITIAL POPULATION OF RANDOM RULES
    population = initialize_population(population_size)

    # EVOLVE THE POPULATION FOR THE SPECIFIED NUMBER OF GENERATIONS
    for generation_number in range(generations):
        start_time = time.perf_counter()

        # CALCULATE THE FITNESS VALUE FOR EACH INDIVIDUAL IN THE POPULATION
        fitness_values = []
        for individual in population:
            individual_fitness = evaluate_rule(individual)
            fitness_values.append(individual_fitness)

        # SELECT THE BEST INDIVIDUALS TO BE ELITES
        elites = select_elites(population, fitness_values, elite_fraction)

        # PRINT THE BEST RULE AND FITNESS FOR THIS GENERATION
        highest_fitness = max(fitness_values)
        lowest_fitness = min(fitness_values)
        average_fitness = sum(fitness_values) / len(fitness_values)
        index_of_best = fitness_values.index(highest_fitness)
        best_rule = population[index_of_best]
        end_time = time.perf_counter()
        generation_time_ms = (end_time - start_time) * 1000

        print("Gen", generation_number,
              "· Best rule", best_rule,
              "· Fitness", round(highest_fitness, 1))

        # Log this generation
        ga_writer.writerow([
            generation_number,
            best_rule,
            f"{highest_fitness:.2f}",
            f"{average_fitness:.2f}",
            f"{lowest_fitness:.2f}",
            f"{generation_time_ms:.1f}"
        ])

        # BUILD THE NEXT POPULATION BY COPYING ELITES AND ADDING CHILDREN
        next_population = list(elites)
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
    highest_fitness = max(final_fitness_values)
    index_of_best = final_fitness_values.index(highest_fitness)
    winner_rule = population[index_of_best]

    ga_file.close()
    print(f"GA log saved to {ga_log_path}")
    return winner_rule
