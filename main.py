import os
import gymnasium as gym
from matplotlib import pyplot as plt

from dynamic_logger import create_logger, log_step

# choose 'ca', 'lqr', 'pid', or 'dqn'
CONTROLLER   = 'ca'
NUM_EPISODES = 10
RENDER_MODE  = 'human'  # 'human' or None
LOG_FILENAME = f"run_{CONTROLLER}.csv"

# Here we added a dynamic check for the selected controller
if CONTROLLER == 'ca':
    from controllers.one_d_ca_controller import ca_action as controller_function

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
        done = False
        step_count = 0

        while not done:
            if CONTROLLER == 'ca':
                action_taken, bit_pre, bit_post = controller_function(observation_state)
            elif CONTROLLER == 'lqr':
                action_taken, _, _ = controller_function(observation_state)
                bit_pre = ""
                bit_post = ""
            elif CONTROLLER == 'pid':
                action_taken, _, _ = controller_function(observation_state)
                bit_pre = ""
                bit_post = ""
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

            done = terminated or truncated  
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
