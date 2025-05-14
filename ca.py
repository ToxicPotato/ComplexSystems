# ca.py

# -------------------------
# Cellular Automata Rule
# -------------------------
# This rule defines the local update logic based on the 3-bit neighborhood (L, C, R).
# The rule is represented as a list of 8 binary outcomes, corresponding to the 8 possible LCR patterns.
# The index corresponds to the binary pattern interpreted as an integer (e.g., '111' -> 7).
# Example: Rule 30 => [0, 1, 1, 1, 1, 0, 0, 0]

def default_rule() -> list[int]:
    """
    Returns a default CA rule as a list of 8 binary values.
    Each index represents a 3-bit pattern (L-C-R), and the value is the next state for the center bit.

    Returns:
        list[int]: Rule mapping 3-bit neighborhoods to new center bit values
    """
    return [0, 1, 1, 1, 1, 0, 0, 0]                                                 # Rule 30: chaotic pattern generator

# -------------------------
# CA Rule Generator
# -------------------------
def generate_rule(index: int) -> list[int]:
    """
    Generates a CA rule from an integer index (0 to 255).
    The index is interpreted as an 8-bit binary number.

    Parameters:
        index (int): Integer between 0 and 255

    Returns:
        list[int]: List of 8 binary values representing the rule
    """
    return [int(bit) for bit in format(index, '08b')]

# -------------------------
# CA Step Function
# -------------------------
def ca_step(bitstring: str, rule: list[int]) -> str:
    """
    Performs one CA update step on a binary string using the given rule.
    For each cell in the string, its next state is determined by the state of itself and its two neighbors.
    This function uses wrap-around at the edges (i.e., circular CA).

    Parameters:
        bitstring (str): Current binary state string (e.g., '01010101')
        rule (list[int]): List of 8 binary values defining the L-C-R update rule

    Returns:
        str: New binary state string after applying CA rule to each cell
    """
    next_state = ''                                                                 # Accumulator for new state
    length = len(bitstring)                                                         # Total number of cells in the automaton

    # Loop through each cell
    for i in range(length):
        # Determine left, center, right using wrap-around
        L = bitstring[(i - 1) % length]                                             # left neighbor (wraps around at start)
        C = bitstring[i]                                                            # current cell (center)
        R = bitstring[(i + 1) % length]                                             # right neighbor (wraps around at end)

        # Create 3-bit pattern string and convert to integer index
        neighborhood = int(L + C + R, 2)

        # Get the new bit value from the rule using the index
        new_bit = rule[neighborhood]

        # Append the result to the new bitstring
        next_state += str(new_bit)

    return next_state

# -------------------------
# Action Decision Function
# -------------------------
def decide_action(ca_output: str, method: str = 'sum', threshold: float = 0.5) -> int:
    """
    Determines an action (0 or 1) based on the final CA bitstring.

    Multiple methods are supported for decision-making:

    1. 'sum':
       Counts the number of '1' bits and returns 1 if the proportion of '1's
       is greater than or equal to the threshold (default 50%).

    2. 'center':
       Uses the center bit of the CA output as the decision. Useful for
       symmetric configurations.

    3. 'majority':
       Compares the number of '1' bits vs. '0' bits and returns the majority
       as the decision.

    Parameters:
        ca_output (str): The resulting bitstring from the CA step
        method (str): Strategy to interpret bitstring ('sum', 'center', 'majority')
        threshold (float): Used for 'sum' method; defines required ratio of '1's

    Returns:
        int: Action decision, 0 or 1
    """
    if method == 'sum':
        # Count number of 1's in the output
        ones = ca_output.count('1')
        # Compare ratio of 1's to threshold
        return int(ones / len(ca_output) >= threshold)

    elif method == 'center':
        # Pick the center bit
        center_index = len(ca_output) // 2
        return int(ca_output[center_index])

    elif method == 'majority':
        # Determine whether there are more 1's or 0's
        ones = ca_output.count('1')
        zeros = len(ca_output) - ones
        return int(ones > zeros)

    else:
        # Handle unsupported methods
        raise ValueError(f"Unsupported method: {method}")