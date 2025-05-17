import gymnasium as gym

from ca_config import BITS_PER_VALUE, ROW_LENGTH, NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_SIZE, NUMBER_OF_CA_TICKS
from utils.ca.decode_action_from_row import decode_action_from_row
from utils.ca.discretize_observation import discretize_observation
from utils.ca.encode_into_row import encode_into_row
from utils.ca.step_eca import step_eca
from utils.ca.generate_rule import generate_rule

NUMBER_OF_EPISODES = 5
MAXIMUM_STEPS_PER_EPISODE = 500

# THIS FUNCTION EVALUATES THE AVERAGE REWARD OF A CA RULE FOR CARTPOLE CONTROL
def evaluate_rule(rule_number, env_name="CartPole-v1"):
    environment = gym.make(env_name)
    sum_of_rewards = 0.0
    rule_table = generate_rule(rule_number, NEIGHBORHOOD_SIZE)

    for episode_index in range(NUMBER_OF_EPISODES):
        observation, info = environment.reset()
        episode_done = False
        step_count = 0

        while not episode_done and step_count < MAXIMUM_STEPS_PER_EPISODE:
            discrete_observation = discretize_observation(observation, bits=BITS_PER_VALUE)
            ca_row = encode_into_row(discrete_observation, row_length=ROW_LENGTH, bits_per_variable=BITS_PER_VALUE)
            for tick_index in range(NUMBER_OF_CA_TICKS):
                ca_row = step_eca(ca_row, rule_table, NEIGHBORHOOD_RADIUS)
            action_value = decode_action_from_row(ca_row)
            next_observation, reward_received, terminated, truncated, info = environment.step(action_value)
            episode_done = terminated or truncated
            sum_of_rewards += reward_received
            observation = next_observation
            step_count += 1

    environment.close()
    average_reward = sum_of_rewards / NUMBER_OF_EPISODES
    return average_reward
