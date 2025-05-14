# env.py

import numpy as np

# -------------------------
# Variable Limits
# -------------------------
# Defines the min and max values for each of the 4 CartPole observations:
# These ranges are either specified by the environment or empirically selected
# to ensure values stay within realistic and meaningful bounds.
#
# Indexes:
# 0: Cart Position           → limited by track bounds
# 1: Cart Velocity           → unbounded, empirically in [-3.0, 3.0]
# 2: Pole Angle (radians)    → environment-defined, ~±0.418 rad (±24°)
# 3: Pole Angular Velocity   → unbounded, empirically in [-3.5, 3.5]

LIMITS = {
    0: (-4.8, 4.8),        # Cart position
    1: (-3.0, 3.0),        # Cart velocity
    2: (-0.418, 0.418),    # Pole angle in radians
    3: (-3.5, 3.5)         # Pole angular velocity
}

# -------------------------
# Quantization Function
# -------------------------
def quantize(val: float, minval: float, maxval: float, levels: int = 256) -> int:
    """
    Quantizes a continuous float value into an integer range [0, levels-1].
    The value is clipped to stay within [minval, maxval], then scaled to fit
    the target number of levels (e.g. 256 levels for 8-bit representation).

    Parameters:
        val (float): Input value to quantize
        minval (float): Lower bound of valid input range
        maxval (float): Upper bound of valid input range
        levels (int): Number of discrete levels (default 256 for 8 bits)

    Returns:
        int: Discrete integer between 0 and levels - 1
    """
    val = np.clip(val, minval, maxval)                                              # Ensure value stays within bounds
    normalized = (val - minval) / (maxval - minval)                                 # Scale to [0, 1]
    return int(normalized * (levels - 1))                                           # Map to [0, levels-1]

# -------------------------
# Observation Encoder
# -------------------------
def obs_to_bitstring(obs: list[float]) -> str:
    """
    Encodes a list of 4 float observations from the CartPole environment
    into a single 32-bit binary string. Each value is quantized to an 8-bit
    integer and then converted to an 8-character binary substring.

    Parameters:
        obs (list[float]): A list of 4 continuous values representing the environment state

    Returns:
        str: Combined 32-character binary string representing the full state
    """
    bitstring = ''
    for i, val in enumerate(obs):
        minval, maxval = LIMITS[i]                                                  # Get the range for this variable
        q = quantize(val, minval, maxval)                                           # Quantize the float
        bitstring += format(q, '08b')                                               # Convert to 8-bit binary string
    return bitstring