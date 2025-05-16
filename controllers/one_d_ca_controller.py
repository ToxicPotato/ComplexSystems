import numpy as np

from utils.genetic_algorithm.genetic_algorithm import genetic_algorithm
from utils.one_d_cellular_automata import discretize_observation, encode_into_row, step_eca, decode_action_from_row

# ─── CA hyperparameters ─────────────────────────────────────────────────────────
ROW_LENGTH     = 64
BITS_PER_VALUE = 5
ECA_TICKS      = 20

# ─── Run the GA once, at import time ────────────────────────────────────────────
print("🔄  Evolving CA rule via GA… this may take a few seconds")
_best_rule = genetic_algorithm()   # calls GA defaults internally
print(f"▶️  GA found best CA rule: {_best_rule}")

# Store it in a clean name
_RULE_INDEX = _best_rule

# ─── CA action API ─────────────────────────────────────────────────────────────
def ca_action(observation: np.ndarray):
    """
    Exactly the same signature as your LQR/PID/DQN controllers:
      - input:  4‐vector observation
      - output: (action: int, bit_pre: np.ndarray, bit_post: np.ndarray)
    Uses the _RULE_INDEX found by GA above.
    """
    # 1) encode
    discrete = discretize_observation(observation, bits=BITS_PER_VALUE)
    row      = encode_into_row(discrete,
                               row_length=ROW_LENGTH,
                               bits=BITS_PER_VALUE)
    bit_pre  = row.copy()

    # 2) evolve CA
    for _ in range(ECA_TICKS):
        row = step_eca(row, _RULE_INDEX)
    bit_post = row.copy()

    # 3) decode
    action = decode_action_from_row(row)
    return action, bit_pre, bit_post
