# ga_ca_optimizer.py

import random
import gymnasium as gym

from controllers.ca_controller import set_rule_index, ca_action, RADIUS, NEIGHBORHOOD_SIZE

# ——— GA Hyperparameters ———
POPULATION_SIZE     = 50
NUM_GENERATIONS     = 10
ELITE_COUNT         = 4
MUTATION_RATE       = 0.1
EVALUATION_EPISODES = 5

# maximum rule index for the given neighbourhood size
MAX_RULE_INDEX = 2 ** (2 ** NEIGHBORHOOD_SIZE) - 1

def run_one_episode():
    """
    Runs one CartPole episode using the current CA rule.
    Returns number of steps survived.
    """
    env = gym.make("CartPole-v1")
    observation_state, _ = env.reset()
    done = False
    steps = 0

    while not done:
        action_taken, _, _ = ca_action(observation_state)
        observation_state, _, done, truncated, _ = env.step(action_taken)
        done = done or truncated
        steps += 1

    env.close()
    return steps

def evaluate_rule(rule_index: int) -> float:
    """
    Set CA to rule_index, run EVALUATION_EPISODES, return average steps.
    """
    set_rule_index(rule_index)
    total_steps = 0
    for _ in range(EVALUATION_EPISODES):
        total_steps += run_one_episode()
    return total_steps / EVALUATION_EPISODES

def genetic_optimize_rule() -> int:
    """
    Runs a simple GA to find a good CA rule index.
    Returns the best rule index found.
    """
    # 1) initialize random population
    population = [random.randint(0, MAX_RULE_INDEX) for _ in range(POPULATION_SIZE)]

    for generation in range(NUM_GENERATIONS):
        # evaluate fitness
        fitness_scores = [evaluate_rule(idx) for idx in population]
        # pair and sort by fitness descending
        paired = list(zip(population, fitness_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        elites = [pair[0] for pair in paired[:ELITE_COUNT]]

        print(f"Generation {generation:02} | best avg steps = {paired[0][1]:.1f}")

        # build next generation
        next_population = elites[:]
        while len(next_population) < POPULATION_SIZE:
            parent = random.choice(elites)
            child  = parent
            # mutate bits
            for bit in range(child.bit_length() or 1):
                if random.random() < MUTATION_RATE:
                    child ^= (1 << bit)
            # clamp to valid range
            child = max(0, min(child, MAX_RULE_INDEX))
            next_population.append(child)

        population = next_population

    # final selection
    final_scores = [evaluate_rule(idx) for idx in population]
    best_rule = population[final_scores.index(max(final_scores))]
    print(f"\nGA complete → best rule index = {best_rule}")
    return best_rule

if __name__ == '__main__':
    best_rule_index = genetic_optimize_rule()
    print(f"Call set_rule_index({best_rule_index}) in ca_controller to use this rule.")
