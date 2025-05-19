import random

def initialize_population(population_size, rule_size):
    population = []
    for _ in range(population_size):
        rule = []
        for _ in range(rule_size):
            bit = random.randint(0, 1)
            rule.append(bit)
        population.append(rule)
    return population