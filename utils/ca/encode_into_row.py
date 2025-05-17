import numpy as np
from ca_config import ROW_LENGTH, BITS_PER_VALUE

# THIS FUNCTION ENCODE DISCRETE OBSERVATION INTO A CA ROW
# discrete_obs: THE DISCRETIZED OBSERVATION (ARRAY OF INTS)
# row_length: LENGTH OF THE CA ROW (DEFAULT 64)
# bits: HOW MANY BITS PER VARIABLE (DEFAULT 5)
def encode_into_row(discrete_observation, row_length=ROW_LENGTH, bits_per_variable=BITS_PER_VALUE):
    ca_row = np.zeros(row_length, dtype=int)
    # LOOP THROUGH EACH VARIABLE IN THE DISCRETE OBSERVATION
    for variable_index, variable_value in enumerate(discrete_observation):
        # ENCODE EACH BIT OF THE VARIABLE VALUE INTO THE ROW
        for bit_index in range(bits_per_variable):
            bit_value = (variable_value >> bit_index) & 1
            position_in_row = variable_index * bits_per_variable + bit_index
            ca_row[position_in_row] = bit_value
    return ca_row