import numpy as np
from ca_config import ACTION_DECODING

# THIS FUNCTION DECODES ACTION FROM THE CENTER CELL OF THE ROW
# row: THE CA ROW AFTER EVOLUTION
def decode_action_from_row(ca_row):
    length = len(ca_row)
    ones   = int(np.sum(ca_row))

    if ACTION_DECODING == "center":
        center_index = length // 2
        return int(ca_row[center_index])

    if ACTION_DECODING == "majority":
        # strictly more than half
        return 1 if ones > (length / 2) else 0

    if ACTION_DECODING == "sum":
        # greater-or-equal threshold
        return 1 if (ones / length) >= 0.5 else 0

    # If we get here, config has an invalid value
    raise ValueError(f"Unknown ACTION_DECODING = {ACTION_DECODING!r}")