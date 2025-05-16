# File: fitness_function.py
import gymnasium as gym

from utils.one_d_cellular_automata import discretize_observation, step_eca, decode_action_from_row, encode_into_row

# CA / evaluation settings
ROW_LENGTH     = 64
BITS_PER_VALUE = 5
ECA_TICKS      = 20
EPISODES       = 5
MAX_STEPS      = 500

def evaluate_rule(rule_number: int, env_name: str = "CartPole-v1") -> float:
    """
    Run CartPole-v1 under the given ECA rule and return average reward over EPISODES.
    """
    env = gym.make(env_name)
    total_reward = 0.0

    for _ in range(EPISODES):
        obs, _info = env.reset()
        done, steps = False, 0

        while not done and steps < MAX_STEPS:
            # 1) encode observation
            discrete = discretize_observation(obs, bits=BITS_PER_VALUE)
            row = encode_into_row(discrete, row_length=ROW_LENGTH, bits=BITS_PER_VALUE)
            # 2) evolve CA
            for _ in range(ECA_TICKS):
                row = step_eca(row, rule_number)
            # 3) decode action
            action = decode_action_from_row(row)
            # 4) step environment
            obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

    env.close()
    return total_reward / EPISODES
