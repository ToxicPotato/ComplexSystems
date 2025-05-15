import os
import gymnasium as gym
from matplotlib import pyplot as plt

from dynamic_logger import create_logger, log_step

# choose 'ca' or 'dqn'
CONTROLLER   = 'ca'
NUM_EPISODES = 100
RENDER_MODE  = None  # 'human' or None
LOG_FILENAME = f"run_{CONTROLLER}.csv"

# If CA, run GA first to find best rule
if CONTROLLER == 'ca':
    from utils.ca_ga_optimize import genetic_optimize_rule
    from controllers.ca_controller import set_rule_index, ca_action as controller_function

    best_rule = genetic_optimize_rule()
    set_rule_index(best_rule)

    log_fields = [
        'episode_index',
        'step_count',
        'observation_state',
        'bit_pre',
        'bit_post',
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

# setup environment & logger
env = gym.make('CartPole-v1', render_mode=RENDER_MODE)
log_file, csv_writer, _ = create_logger(
    filename=LOG_FILENAME,
    fieldnames=log_fields
)

# ensure plots folder exists
os.makedirs('results/plots', exist_ok=True)

def run_episodes():
    lengths = []
    for episode_index in range(NUM_EPISODES):
        observation_state, _ = env.reset()
        terminated = False
        step_count = 0

        while not terminated:
            if CONTROLLER == 'ca':
                action_taken, bit_pre, bit_post = controller_function(observation_state)
            else:
                action_taken = controller_function(observation_state)
                bit_pre = ""
                bit_post = ""

            new_observation, reward_received, terminated, truncated, _ = (
                env.step(action_taken)
            )

            log_step(
                csv_writer,
                log_fields,
                episode_index,
                step_count,
                new_observation,
                action_taken,
                reward_received,
                terminated,
                bit_pre,
                bit_post
            )

            observation_state = new_observation
            step_count += 1

        print(f"Episode {episode_index} finished after {step_count} steps")
        lengths.append(step_count)

    return lengths

if __name__ == '__main__':
    episode_lengths = run_episodes()

    log_file.close()
    env.close()

    # Plot 1: learning curve
    plt.figure()
    plt.plot(range(NUM_EPISODES), episode_lengths, marker='o')
    plt.xlabel('Episode')
    plt.ylabel('Steps Survived')
    plt.title(f'{CONTROLLER.upper()} Learning Curve')
    plt.savefig(f'results/plots/{CONTROLLER}_learning_curve.png')
    plt.close()

    print("Saved plots")
