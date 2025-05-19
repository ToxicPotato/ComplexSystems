from utils.ca.discretize_observation import discretize_observation
from utils.ca.encode_into_row import encode_into_row
from ca_config import ROW_LENGTH, BITS_PER_VALUE

# THIS FUNCTION CONVERTS AN OBSERVATION TO A BITSTRING FOR CA
def observation_to_bitstring(observation):
    discrete_observation = discretize_observation(observation, bits=BITS_PER_VALUE)
    ca_row = encode_into_row(discrete_observation, row_length=ROW_LENGTH, bits_per_variable=BITS_PER_VALUE)
    return ca_row