import random

from ca_config import TOURNAMENT_SIZE


# THIS FUNCTION SELECTS THE TOP FRACTION OF INDIVIDUALS AS ELITES
# population: LIST OF RULE INDICES
# fitness_values: LIST OF FITNESS VALUES FOR THE POPULATION
# elite_fraction: FRACTION OF POPULATION TO SELECT AS ELITES
def select_elites(population, fitness_values, elite_fraction):
    n_elites = int(len(population) * elite_fraction)
    if n_elites < 1:
        n_elites = 1
    combined = list(zip(population, fitness_values))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    elites = [rule for rule, fitness in sorted_combined[:n_elites]]
    return elites

# THIS FUNCTION SELECTS ONE INDIVIDUAL FROM THE POPULATION USING TOURNAMENT SELECTION
# population: LIST OF RULE INDICES
# fitness_values: LIST OF FITNESS VALUES
# tournament_size: NUMBER OF INDIVIDUALS IN THE TOURNAMENT (DEFAULT 3)
def tournament_selection(population, fitness_values, tournament_size):
    combined = list(zip(population, fitness_values))
    competitors = random.sample(combined, tournament_size)
    winner = competitors[0]
    for candidate in competitors:
        if candidate[1] > winner[1]:
            winner = candidate
    selected_rule = winner[0]
    return selected_rule