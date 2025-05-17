import random

# THIS FUNCTION CREATES THE INITIAL POPULATION OF RANDOM RULES
# population_size: NUMBER OF RULES TO CREATE
def initialize_population(population_size):
    population_list = []
    for i in range(population_size):
        random_rule = random.randint(0, 255)
        population_list.append(random_rule)
    return population_list
