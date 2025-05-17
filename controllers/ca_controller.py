from ca_config import BITS_PER_VALUE, ROW_LENGTH, NEIGHBORHOOD_RADIUS, NEIGHBORHOOD_SIZE, NUMBER_OF_CA_TICKS
from utils.ca.decode_action_from_row import decode_action_from_row
from utils.ca.generate_rule import generate_rule
from utils.ca.observation_to_bitstring import observation_to_bitstring
from utils.ca.step_eca import step_eca

# RULE INDEX AND RULE TABLE ARE SET FROM THE MAIN FILE
RULE_INDEX = 110
RULE_TABLE = generate_rule(RULE_INDEX, NEIGHBORHOOD_SIZE)

# THIS FUNCTION ALLOWS MAIN/GENETIC ALGORITHM TO UPDATE THE RULE
def set_rule_index(new_rule_index):
    global RULE_INDEX, RULE_TABLE
    RULE_INDEX = new_rule_index
    RULE_TABLE = generate_rule(RULE_INDEX, NEIGHBORHOOD_SIZE)

# THIS IS THE MAIN FUNCTION CALLED FROM MAIN
def ca_action(observation):
    bitstring_before = observation_to_bitstring(observation)
    bitstring_after = bitstring_before.copy()
    for step_number in range(NUMBER_OF_CA_TICKS):
        bitstring_after = step_eca(bitstring_after, RULE_TABLE, NEIGHBORHOOD_RADIUS)
    action = decode_action_from_row(bitstring_after)
    return action, bitstring_before, bitstring_after
