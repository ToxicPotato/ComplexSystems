import numpy as np
from ca_config import BITS_PER_VALUE

# THIS FUNCTION TURNS CONTINUOUS OBSERVATION INTO DISCRETE VALUES
# observation: THE OBSERVATION ARRAY FROM CARTPOLE ENVIRONMENT
# bits: NUMBER OF BITS TO REPRESENT EACH VARIABLE (DEFAULT IS 5)

def discretize_observation(observation, bits=BITS_PER_VALUE):
    minimum_values = np.array([-2.4, -3.0, -0.20944, -5.0])
    maximum_values = np.array([2.4, 3.0, 0.20944, 5.0])
    clamped_observation = np.minimum(np.maximum(observation, minimum_values), maximum_values)
    normalized_observation = (clamped_observation - minimum_values) / (maximum_values - minimum_values)
    maximum_integer = (1 << bits) - 1
    discrete_observation = np.round(normalized_observation * maximum_integer).astype(int)
    return discrete_observation