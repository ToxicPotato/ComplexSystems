import random

def mutate(individual_rule, mutation_rate, bits_per_value):
    mutated_rule = individual_rule
    for bit_index in range(bits_per_value):
        should_flip = random.random() < mutation_rate
        if should_flip:
            mutated_rule ^= (1 << bit_index)  # Flip the bit at bit_index
    return mutated_rule
