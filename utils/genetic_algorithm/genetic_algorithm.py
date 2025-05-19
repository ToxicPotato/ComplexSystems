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
    # CALCULATE NEIGHBORHOOD SIZE AND CORRESPONDING RULE SIZE
    neighborhood_size = 2 * neighborhood_radius + 1
    rule_size = 2 ** neighborhood_size

    # DIRECTORY FOR GA LOGS
    log_directory = os.path.join("results", "ga_logs")
    os.makedirs(log_directory, exist_ok=True)
    # TIMESTAMPED LOG FILE PATH
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_directory, f"ga_log_{timestamp}.csv")

    # CSV LOG FILE FOR WRITING GENERATION STATISTICS
    with open(log_path, mode='w', newline='') as log_file:
        csv_writer = csv.writer(log_file)
        # WRITE HEADER ROW FOR CSV LOG
        csv_writer.writerow([
            "generation",
            "best_rule",
            "best_fitness",
            "avg_fitness",
            "worst_fitness",
            "generation_time_ms"
        ])

        # INITIALIZE FIRST GENERATION POPULATION
        population = initialize_population(population_size, rule_size)

        # MAIN EVOLUTIONARY LOOP OVER SPECIFIED GENERATIONS
        for generation_number in range(generations):
            # START TIMER FOR THIS GENERATION
            start_time = time.perf_counter()
            fitness_scores = []  # STORE FITNESS SCORES FOR EACH INDIVIDUAL

            # EVALUATE FITNESS FOR ENTIRE POPULATION
            for rule in population:
                rule_fitness = evaluate_rule(rule)
                fitness_scores.append(rule_fitness)

            # SELECT ELITE INDIVIDUALS TO CARRY FORWARD
            elites = select_elites(population, fitness_scores, elite_fraction)

            # COMPUTE GENERATION STATISTICS
            best_fitness = max(fitness_scores)
            worst_fitness = min(fitness_scores)
            average_fitness = sum(fitness_scores) / len(fitness_scores)
            index_of_best = fitness_scores.index(best_fitness)
            best_rule = population[index_of_best]
            # STOP TIMER AND CALCULATE DURATION IN MILLISECONDS
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000

            # OUTPUT PROGRESS TO CONSOLE
            print(
                f"Gen {generation_number} · Best rule {best_rule} · Fitness {round(best_fitness, 1)}"
            )

            # WRITE GENERATION DATA TO CSV LOG
            csv_writer.writerow([
                generation_number,
                str(best_rule),
                f"{best_fitness:.2f}",
                f"{average_fitness:.2f}",
                f"{worst_fitness:.2f}",
                f"{time_ms:.1f}"
            ])

            # BUILD NEXT GENERATION POPULATION STARTING WITH ELITES
            next_population = list(elites)
            # FILL REMAINING POPULATION WITH OFFSPRING
            while len(next_population) < population_size:
                # SELECT TWO PARENTS USING TOURNAMENT SELECTION
                parent_one = tournament_selection(population, fitness_scores, tournament_size=3)
                parent_two = tournament_selection(population, fitness_scores, tournament_size=3)
                # PRODUCE CHILD VIA CROSSOVER
                child = crossover(parent_one, parent_two)
                # MUTATE CHILD BASED ON MUTATION RATE
                mutated_child = mutate(child, mutation_rate)
                next_population.append(mutated_child)

            # UPDATE POPULATION FOR NEXT GENERATION
            population = next_population

        # AFTER EVOLUTION, EVALUATE FINAL POPULATION FITNESS
        final_fitness_scores = []
        for rule in population:
            rule_fitness = evaluate_rule(rule)
            final_fitness_scores.append(rule_fitness)
        # IDENTIFY WINNING RULE FROM FINAL POPULATION
        best_final_fitness = max(final_fitness_scores)
        index_of_best = final_fitness_scores.index(best_final_fitness)
        winner_rule = population[index_of_best]

        # NOTIFY USER AND RETURN BEST RULE
        print(f"GA log saved to {log_path}")
        return winner_rule
