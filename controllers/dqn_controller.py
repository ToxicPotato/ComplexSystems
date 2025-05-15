import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gym.make("CartPole-v1")

VEC_ENV = DummyVecEnv([make_env])

model = DQN(
    'MlpPolicy',
    VEC_ENV,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=1_000,
    train_freq=4,
    target_update_interval=500,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1
)

def dqn_train(steps: int = 50_000):
    model.learn(steps)

def dqn_action(observation_state):
    action, _ = model.predict(observation_state, deterministic=True)
    return int(action)
