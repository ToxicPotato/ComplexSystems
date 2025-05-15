from env import obs_to_bitstring
from utils.ca import generate_rule, ca_step, decide_action

# CONFIG
RULE_INDEX        = 119
RADIUS            = 1
STEPS             = 10
NEIGHBORHOOD_SIZE = 2*RADIUS + 1

RULE = generate_rule(RULE_INDEX, NEIGHBORHOOD_SIZE)

def set_rule_index(new_index: int):
    global RULE_INDEX, RULE
    RULE_INDEX = new_index
    RULE       = generate_rule(RULE_INDEX, NEIGHBORHOOD_SIZE)

def ca_action(obs):
    bit_pre  = obs_to_bitstring(obs)
    bit_post = bit_pre
    for _ in range(STEPS):
        bit_post = ca_step(bit_post, RULE, RADIUS)
    action = decide_action(bit_post, method='sum')
    return action, bit_pre, bit_post
