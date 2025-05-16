# File: mutation.py
import random

def mutate(individual: int, mutation_rate: float, num_bits: int = 8) -> int:
    """
    Flip each of the num_bits bits with probability mutation_rate.
    """
    mutant = individual
    for bit in range(num_bits):
        if random.random() < mutation_rate:
            mutant ^= (1 << bit)
    return mutant
