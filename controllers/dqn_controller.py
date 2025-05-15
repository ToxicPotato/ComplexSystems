from stable_baselines3 import DQN

# create model once
model = DQN('MlpPolicy', 'CartPole-v1')

def dqn_train(steps=20000):
    # train the agent
    model.learn(steps)

def dqn_action(obs):
    # select action using trained model
    act, _ = model.predict(obs)
    return int(act)