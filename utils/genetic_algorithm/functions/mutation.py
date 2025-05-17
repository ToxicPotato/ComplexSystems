import random

# THIS FUNCTION RANDOMLY FLIPS BITS IN THE INDIVIDUAL BASED ON MUTATION RATE
# individual: THE RULE INDEX TO MUTATE
# mutation_rate: PROBABILITY OF FLIPPING EACH BIT
def mutate(individual, mutation_rate):
    mutated_individual = individual
    for bit_position in range(8):
        random_value = random.random()
        if random_value < mutation_rate:
            mutated_individual = mutated_individual ^ (1 << bit_position)
    return mutated_individual
