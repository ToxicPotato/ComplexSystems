# File: initialization.py
import random

def initialize_population(pop_size: int, rule_min: int = 0, rule_max: int = 255) -> list[int]:
    """
    Initialize a population of CA rule indices (integers) uniformly at random.
    """
    return [random.randint(rule_min, rule_max) for _ in range(pop_size)]