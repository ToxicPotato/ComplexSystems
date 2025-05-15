from env import obs_to_bitstring
from utils.ca import generate_rule, ca_step, decide_action

# simple CA settings
RULE = generate_rule(30, 5)
RADIUS = 2
STEPS = 10

def ca_action(obs):
    bits = obs_to_bitstring(obs)
    for _ in range(STEPS):
        bits = ca_step(bits, RULE, RADIUS)
    return decide_action(bits)