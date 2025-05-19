import os
import time
from datetime import datetime
import gymnasium as gym

from ca_config import BITS_PER_VALUE, ROW_LENGTH, NEIGHBORHOOD_RADIUS, NUMBER_OF_CA_TICKS, ACTION_DECODING, \
    NUMBER_OF_EPISODES
from dynamic_logger import create_logger, log_step

# Experiment settings
CONTROLLER = 'ca'  # 'ca', 'lqr', 'pid', 'dqn'
EPISODES = NUMBER_OF_EPISODES
RENDER_MODE = 'human'
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# Controller selection
if CONTROLLER == 'ca':
    from utils.genetic_algorithm.genetic_algorithm import genetic_algorithm
    from controllers.ca_controller import ca_action, set_rule_index
    print("Finding best CA rule via GA...")
    best_rule = genetic_algorithm()
    print(f"Best CA rule: {best_rule}")
    set_rule_index(best_rule)
    controller_fn = ca_action
elif CONTROLLER == 'lqr':
    from controllers.lqr_controller import lqr_action as controller_fn
    best_rule = None
elif CONTROLLER == 'pid':
    from controllers.pid_controller import pid_action as controller_fn
    best_rule = None
elif CONTROLLER == 'dqn':
    from controllers.dqn_controller import dqn_train, dqn_action as controller_fn
    dqn_train(steps=50_000)
    best_rule = None
else:
    raise ValueError("Unknown controller type")

# Setup logging
os.makedirs('results/csv_logs', exist_ok=True)
csv_file, csv_writer = create_logger(CONTROLLER)

env = gym.make('CartPole-v1', render_mode=RENDER_MODE)
lengths = []

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    step = 0
    while not done:
        t0 = time.perf_counter()
        if CONTROLLER == 'ca':
            action, bit_pre, bit_post = controller_fn(obs)
        else:
            action = controller_fn(obs)
            bit_pre = bit_post = None
        t1 = time.perf_counter()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        log_step(
            csv_writer,
            run_id=RUN_ID,
            controller_type=CONTROLLER,
            episode_index=ep,
            step_count=step,
            time_start=t0,
            time_end=t1,
            time_delta_ms=(t1-t0)*1000,
            observation_state=obs,
            action_taken=action,
            reward_received=reward,
            terminated=done,
            # CA params
            bits_per_value=BITS_PER_VALUE,
            row_length=ROW_LENGTH,
            neighborhood_radius=NEIGHBORHOOD_RADIUS,
            num_ca_ticks=NUMBER_OF_CA_TICKS,
            action_decoding=ACTION_DECODING,
            rule_index=best_rule
        )
        obs = next_obs
        step += 1
    print(f"Episode {ep} ended at step {step}")
    lengths.append(step)

# Close
csv_file.close()
env.close()

print(f"Run {RUN_ID} completed. CSV log saved to results/csv_logs/run_{CONTROLLER}_{RUN_ID}.csv")
print("Use the separate plot_results.py script to generate all figures.")
