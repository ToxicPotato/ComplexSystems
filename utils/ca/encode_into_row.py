import numpy as np
from ca_config import ROW_LENGTH, BITS_PER_VALUE

# ENCODES A DISCRETE OBSERVATION INTO A 1D CELLULAR AUTOMATON ROW
# discrete_observation: ARRAY OF INTEGER VALUES REPRESENTING DISCRETIZED OBSERVATIONS
# row_length: TOTAL LENGTH OF THE OUTPUT CA ROW (DEFAULT FROM CONFIG)
# bits_per_variable: NUMBER OF BITS TO REPRESENT EACH OBSERVATION VALUE
def encode_into_row(discrete_observation, row_length=ROW_LENGTH, bits_per_variable=BITS_PER_VALUE):
    ca_row = np.zeros(row_length, dtype=int)
    # ITERATE OVER EACH VALUE IN THE DISCRETE OBSERVATION
    for variable_index, variable_value in enumerate(discrete_observation):
        # CONVERT EACH VALUE TO BINARY AND PLACE BITS IN THE ROW
        for bit_index in range(bits_per_variable):
            bit_value = (variable_value >> bit_index) & 1
            position_in_row = variable_index * bits_per_variable + bit_index
            ca_row[position_in_row] = bit_value
    return ca_row
