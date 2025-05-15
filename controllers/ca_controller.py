from env import obs_to_bitstring
from utils.ca import generate_rule, ca_step, decide_action

# simple CA settings
RADIUS = 2
neighborhood_size = 2 * RADIUS + 1
STEPS = 10

def ca_action(obs):
    rule = generate_rule(30, neighborhood_size=neighborhood_size)
    bits = obs_to_bitstring(obs)
    for _ in range(STEPS):
        bits = ca_step(bits, rule, RADIUS)
    return decide_action(bits)