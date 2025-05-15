import gymnasium as gym
from dynamic_logger import create_logger, log_step

# CONFIGS
CONTROLLER   = 'ca'        # 'ca' or 'dqn'
EPISODES     = 5           # number of episodes to run
RENDER_MODE  = None        # 'human' or None
LOG_FILENAME = f"run_{CONTROLLER}.csv"

# LOGIC TO SELECTED THE CONTROLLER BASED ON INPUT FROM ABOVE

# Nå har du dqn og ca
if CONTROLLER == 'ca':
    from controllers.ca_controller import ca_action as controller
    fields = ['episode','step','obs','bit_pre','bit_post','action','reward','terminated']

elif CONTROLLER == 'dqn':
    from controllers.dqn_controller import dqn_action as controller, dqn_train
    dqn_train()
    fields = ['episode','step','obs','action','reward','terminated']

else:
    raise ValueError(f"Unknown controller: {CONTROLLER}")

# ENVIRONMENT AND LOGGER SETUP
env = gym.make('CartPole-v1', render_mode=RENDER_MODE)
log_file, logger, fields = create_logger(
    directory='logs',
    filename=LOG_FILENAME,
    fieldnames=fields
)

# RUN THE EPISODES
def run(episodes=EPISODES):
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = controller(obs)
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            log_step(
                writer=logger,
                fieldnames=fields,
                episode=ep,
                step=steps,
                obs=obs,
                action=action,
                reward=reward,
                terminated=done,
            )
            steps += 1
        print(f"Episode {ep}: {steps} steps")

if __name__ == '__main__':
    run()
    log_file.close()
    env.close()