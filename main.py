# main.py

import gymnasium as gym
from env import obs_to_bitstring
from ca import ca_step, generate_rule, decide_action
from logger import create_logger, log_step

# -------------------------
# CA Control Loop
# -------------------------
# This script runs one episode of CartPole using a Cellular Automaton (CA)
# to process observations and determine actions. The CA rule transforms the
# encoded binary observation through several steps, after which a decision
# function selects an action based on the transformed state.

# Initialize environment with rendering enabled
env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()

# -------------------------
# Configuration Parameters
# -------------------------
number_of_steps = 10                                                                # How many CA iterations to apply before decision
num_episodes = 20                                                                   # Number of episodes to run per rule
render = False                                                                      # Set to True to enable GUI rendering
radius = 2                                                                          # Neighborhood radius (1 = 3-bit, 2 = 5-bit, etc.)
neighborhood_size = 2 * radius + 1

# -------------------------
# Rule Evaluation Loop
# -------------------------
rule_results = []

for rule_index in range(2 ** (2 ** neighborhood_size)):
    rule = generate_rule(rule_index, neighborhood_size=neighborhood_size)
    scores = []

    print(f"\n=== Evaluating Rule {rule_index:03} - Radius {radius} ===")

    log_file, logger = create_logger(filename=f"rule_r{radius}_{rule_index:03}.csv")

    for episode in range(num_episodes):
        env = gym.make("CartPole-v1", render_mode="human" if render else None)
        obs, _ = env.reset()

        terminated = False
        step_count = 0

        while not terminated:
            bit_pre = obs_to_bitstring(obs)
            bit_post = bit_pre
            for _ in range(number_of_steps):
                bit_post = ca_step(bit_post, rule, radius=radius)

            action = decide_action(bit_post, method='sum')
            obs, reward, terminated, truncated, info = env.step(action)

            log_step(
                writer=logger,
                episode=episode,
                step=step_count,
                obs=obs,
                bit_pre=bit_pre,
                bit_post=bit_post,
                action=action,
                reward=reward,
                terminated=terminated
            )

            step_count += 1

        env.close()
        scores.append(step_count)

    log_file.close()

    avg_score = sum(scores) / len(scores)
    print(f"Rule {rule_index:03} avg score: {avg_score:.2f}")
    rule_results.append((rule_index, avg_score))

# -------------------------
# Summary of All Rules
# -------------------------
best = max(rule_results, key=lambda x: x[1])
print("\n=== Best Rule ===")
print(f"Rule Index:  {best[0]}")
print(f"Average Score: {best[1]:.2f}")
