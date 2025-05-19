import random

def initialize_population(population_size, bits_per_value):
    highest_possible_rule = (1 << bits_per_value) - 1  # Maximum value for the given bits
    population_list = []
    for _ in range(population_size):
        random_rule = random.randint(0, highest_possible_rule)
        population_list.append(random_rule)
    return population_list
