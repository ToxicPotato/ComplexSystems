import random

def crossover(parent1_bits, parent2_bits):
    rule_size = len(parent1_bits)
    if rule_size == 1:
        child = parent1_bits[:]
    else:
        crossover_point = random.randint(1, rule_size - 1)
        left = parent1_bits[:crossover_point]
        right = parent2_bits[crossover_point:]
        child = left + right
    return child