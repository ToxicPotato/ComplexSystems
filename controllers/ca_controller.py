from ca_config import NEIGHBORHOOD_SIZE, NEIGHBORHOOD_RADIUS, NUMBER_OF_CA_TICKS
from utils.ca.generate_rule      import generate_rule
from utils.ca.observation_to_bitstring import observation_to_bitstring
from utils.ca.step_eca           import step_eca
from utils.ca.decode_action_from_row import decode_action_from_row

RULE_INDEX = [0] * (2 ** NEIGHBORHOOD_SIZE)

def set_rule_index(new_rule_index):
    global RULE_INDEX
    RULE_INDEX = new_rule_index

def ca_action(observation):
    rule_table = generate_rule(RULE_INDEX)
    bit_pre = observation_to_bitstring(observation)
    bit_post = bit_pre.copy()
    for _ in range(NUMBER_OF_CA_TICKS):
        bit_post = step_eca(bit_post, rule_table, NEIGHBORHOOD_RADIUS)
    action = decode_action_from_row(bit_post)
    return action, bit_pre, bit_post
