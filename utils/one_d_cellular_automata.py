import numpy as np


def step_eca(row: np.ndarray, rule_number: int) -> np.ndarray:
    """
    One tick of a 1‑D binary Elementary CA (Rule 0–255).
    """
    L = row.size
    new_row = np.zeros_like(row)
    # build 8‑entry lookup table
    rule_table = [(rule_number >> i) & 1 for i in range(8)]
    for i in range(L):
        left, selfb, right = row[(i-1)%L], row[i], row[(i+1)%L]
        idx = (left << 2) | (selfb << 1) | right
        new_row[i] = rule_table[idx]
    return new_row


def discretize_observation(obs: np.ndarray, bits: int = 5) -> np.ndarray:
    """
    Clamp & scale each of the 4 floats into integer 0..(2^bits-1).
    """
    mins = np.array([-2.4, -3.0, -0.20944, -5.0])
    maxs = np.array([ 2.4,  3.0,  0.20944,  5.0])
    clamped    = np.minimum(np.maximum(obs, mins), maxs)
    normalized = (clamped - mins) / (maxs - mins)
    max_int    = (1 << bits) - 1
    return np.round(normalized * max_int).astype(int)


def encode_into_row(discrete_obs: np.ndarray,
                    row_length: int = 64,
                    bits: int = 5) -> np.ndarray:
    """
    Write the 4×bits bits of `discrete_obs` into the first cells;
    the rest remain zero.
    """
    row = np.zeros(row_length, dtype=int)
    for var_idx, val in enumerate(discrete_obs):
        for b in range(bits):
            row[var_idx * bits + b] = (val >> b) & 1
    return row


def decode_action_from_row(row: np.ndarray) -> int:
    """
    Read the central cell: 0→push-left, 1→push-right.
    """
    return int(row[row.size // 2])
