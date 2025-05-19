import os
import csv
import time
from ca_config import (
    POPULATION_SIZE, NUMBER_OF_GENERATIONS, ELITE_PERCENTAGE, MUTATION_RATE,
    NEIGHBORHOOD_RADIUS
)
from .functions.initialization import initialize_population
from .functions.fitness_function import evaluate_rule
from .functions.selection import select_elites, tournament_selection
from .functions.crossover import crossover
from .functions.mutation import mutate

def genetic_algorithm(
    population_size=POPULATION_SIZE,
    generations=NUMBER_OF_GENERATIONS,
    elite_fraction=ELITE_PERCENTAGE,
    mutation_rate=MUTATION_RATE,
    neighborhood_radius=NEIGHBORHOOD_RADIUS,
):
    neighborhood_size = 2 * neighborhood_radius + 1
    rule_size = 2 ** neighborhood_size  # The length of each rule (bit list)

    log_directory = os.path.join("results", "ga_logs")
    os.makedirs(log_directory, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_directory, f"ga_log_{timestamp}.csv")

    with open(log_path, mode='w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        csv_writer.writerow([
            "generation",
            "best_rule",
            "best_fitness",
            "avg_fitness",
            "worst_fitness",
            "generation_time_ms"
        ])

        population = initialize_population(population_size, rule_size)

        for generation_number in range(generations):
            start_time = time.perf_counter()
            fitness_scores = []
            for rule in population:
                rule_fitness = evaluate_rule(rule)
                fitness_scores.append(rule_fitness)

            elites = select_elites(population, fitness_scores, elite_fraction)

            best_fitness = max(fitness_scores)
            worst_fitness = min(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            index_of_best = fitness_scores.index(best_fitness)
            best_rule = population[index_of_best]
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000

            print(
                f"Gen {generation_number} · Best rule {best_rule} · Fitness {round(best_fitness, 1)}"
            )

            csv_writer.writerow([
                generation_number,
                str(best_rule),  # Save rule as string (e.g., bit list)
                f"{best_fitness:.2f}",
                f"{average_fitness:.2f}",
                f"{worst_fitness:.2f}",
                f"{time_ms:.1f}"
            ])

            next_population = list(elites)
            while len(next_population) < population_size:
                parent_one = tournament_selection(population, fitness_scores, tournament_size=3)
                parent_two = tournament_selection(population, fitness_scores, tournament_size=3)
                child = crossover(parent_one, parent_two)
                mutated_child = mutate(child, mutation_rate)
                next_population.append(mutated_child)

            population = next_population

        final_fitness_scores = []
        for rule in population:
            rule_fitness = evaluate_rule(rule)
            final_fitness_scores.append(rule_fitness)
        best_final_fitness = max(final_fitness_scores)
        index_of_best = final_fitness_scores.index(best_final_fitness)
        winner_rule = population[index_of_best]

        print(f"GA log saved to {log_path}")
        return winner_rule
