import random

# Uniform crossover
# https://www.geeksforgeeks.org/crossover-in-genetic-algorithm/
def crossover(parent1: int, parent2: int, num_bits: int = 8) -> int:
    """
    Uniform crossover between two parent rule numbers.
    For each bit, randomly choose from parent1 or parent2.
    """
    mask = 0
    for bit in range(num_bits):
        if random.random() < 0.5:
            mask |= (1 << bit)
    # mask bits=1→take from parent1, 0→from parent2
    offspring = (parent1 & mask) | (parent2 & ~mask)
    return offspring