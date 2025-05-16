# File: selection.py
import random
from typing import List, Tuple

def select_elites(population: List[int], fitnesses: List[float], elite_frac: float) -> List[int]:
    """
    Select the top fraction of individuals as elites.
    """
    num_elites = max(1, int(elite_frac * len(population)))
    sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [ind for ind, _ in sorted_pop[:num_elites]]


def tournament_selection(population: List[int], fitnesses: List[float], k: int = 3) -> int:
    """
    Perform tournament selection of size k and return one parent.
    """
    participants = random.sample(list(zip(population, fitnesses)), k)
    participants.sort(key=lambda x: x[1], reverse=True)
    return participants[0][0]
