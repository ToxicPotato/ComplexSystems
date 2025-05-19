import random

def crossover(parent_rule_one, parent_rule_two, bits_per_value):
    # Choose a crossover point between the first and last bit (not at the ends)
    crossover_point = random.randint(1, bits_per_value - 1)
    left_mask = (1 << crossover_point) - 1   # Mask for the left part from parent one
    right_mask = ~left_mask                  # Mask for the right part from parent two
    left_bits = parent_rule_one & left_mask
    right_bits = parent_rule_two & right_mask
    child_rule = left_bits | right_bits
    return child_rule
