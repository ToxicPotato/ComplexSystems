from utils.genetic_algorithm.functions.initialization import initialize_population
from utils.genetic_algorithm.functions.fitness_function import evaluate_rule
from utils.genetic_algorithm.functions.selection import select_elites, tournament_selection
from utils.genetic_algorithm.functions.crossover import crossover
from utils.genetic_algorithm.functions.mutation import mutate

# Configs
POP_SIZE       = 64
GENERATIONS    = 10
ELITE_FRAC     = 0.1
MUTATION_RATE  = 0.02

def genetic_algorithm(
    pop_size: int = POP_SIZE,
    generations: int = GENERATIONS,
    elite_frac: float = ELITE_FRAC,
    mutation_rate: float = MUTATION_RATE
) -> int:
    """
    Run a GA to evolve an 8-bit CA rule for CartPole control.
    Returns the best rule index found.
    """
    # 1) Initialize population
    population = initialize_population(pop_size)

    for gen in range(generations):
        # 2) Fitness ranking
        fitnesses = [evaluate_rule(ind) for ind in population]
        # 3) Select elites
        elites = select_elites(population, fitnesses, elite_frac)
        best_fit = max(fitnesses)
        best_rule = population[fitnesses.index(best_fit)]
        print(f"Gen {gen} · Best rule {best_rule} · Fitness {best_fit:.1f}")

        # 4) Next generation
        next_population = elites.copy()
        while len(next_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            next_population.append(child)

        population = next_population

    # Final evaluation to pick best
    final_fitnesses = [evaluate_rule(ind) for ind in population]
    winner = population[final_fitnesses.index(max(final_fitnesses))]
    return winner