import random

def mutate(rule_bits, mutation_rate):
    mutated = []
    for bit in rule_bits:
        if random.random() < mutation_rate:
            new_bit = 1 - bit
        else:
            new_bit = bit
        mutated.append(new_bit)
    return mutated