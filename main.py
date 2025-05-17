import os
import gymnasium as gym
from matplotlib import pyplot as plt
from ca_config import BITS_PER_VALUE, ROW_LENGTH, NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_SIZE, NUMBER_OF_CA_TICKS
from dynamic_logger import create_logger, log_step

# choose 'ca', 'lqr', 'pid', or 'dqn'
CONTROLLER   = 'ca'
NUMBER_OF_EPISODES = 10
RENDER_MODE = 'human'
LOG_FILENAME = f"run_{CONTROLLER}.csv"

# Here we added a dynamic check for the selected controller
if CONTROLLER == 'ca':
    from utils.genetic_algorithm.genetic_algorithm import genetic_algorithm
    from controllers.ca_controller import ca_action, set_rule_index

    print("Running genetic algorithm to find the best CA rule...")
    best_rule_index = genetic_algorithm()
    print(f"Best CA rule index: {best_rule_index}")
    set_rule_index(best_rule_index)
    controller_function = ca_action

    log_fields = [
        'episode_index',
        'step_count',
        'observation_state',
        'bit_pre',
        'bit_post',
        'action_taken',
        'reward_received',
        'terminated',
    ]

elif CONTROLLER == 'lqr':
    from controllers.lqr_controller import lqr_action as controller_function

    log_fields = [
        'episode_index',
        'step_count',
        'observation_state',
        'action_taken',
        'reward_received',
        'terminated'
    ]

elif CONTROLLER == 'pid':
    from controllers.pid_controller import pid_action as controller_function
    
    log_fields = [
        'episode_index',
        'step_count',
        'observation_state',
        'action_taken',
        'reward_received',
        'terminated'
    ]

elif CONTROLLER == 'dqn':
    from controllers.dqn_controller import dqn_train, dqn_action as controller_function

    # train before running
    dqn_train(steps=50_000)

    log_fields = [
        'episode_index',
        'step_count',
        'observation_state',
        'action_taken',
        'reward_received',
        'terminated'
    ]

else:
    raise ValueError(f"Unknown controller: {CONTROLLER}")

# Setting the environment from cartpole and preeparing the logger
environment = gym.make('CartPole-v1', render_mode=RENDER_MODE)
log_file, csv_writer, _ = create_logger(filename=LOG_FILENAME, fieldnames=log_fields)


# ensure plots folder exists
os.makedirs('results/plots', exist_ok=True)

def run_episodes():
    episode_lengths = []
    for episode_index in range(NUMBER_OF_EPISODES):
        observation_state, _ = environment.reset()
        done = False
        step_count = 0
        while not done:
            if CONTROLLER == 'ca':
                action_taken, bit_pre, bit_post = controller_function(observation_state)
            else:
                # One value returned, set empty bit_pre and bit_post for logging
                action_taken = controller_function(observation_state)
                bit_pre = ""
                bit_post = ""
            next_observation, reward_received, terminated, truncated, _ = environment.step(action_taken)
            log_step(
                csv_writer, log_fields, episode_index, step_count, next_observation,
                action_taken, reward_received, terminated, bit_pre, bit_post
            )
            done = terminated or truncated
            observation_state = next_observation
            step_count += 1
        print(f"Episode {episode_index} finished after {step_count} steps")
        episode_lengths.append(step_count)
    return episode_lengths


if __name__ == '__main__':
    episode_lengths = run_episodes()
    log_file.close()
    environment.close()
    plt.figure()
    plt.plot(range(NUMBER_OF_EPISODES), episode_lengths, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Steps Survived')
    plt.title(f'{CONTROLLER.upper()} Learning Curve')
    plt.savefig(f'results/plots/{CONTROLLER}_learning_curve.png')
    plt.close()
    print("Saved plot")