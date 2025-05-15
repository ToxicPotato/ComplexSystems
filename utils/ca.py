# ca.py

# -------------------------
# CA Rule Generator
# -------------------------
def generate_rule(index: int, neighborhood_size: int = 3) -> list[int]:
    """
    Generates a CA rule as a binary list given a rule index and neighborhood size.

    Parameters:
        index (int): The rule number to convert to a binary rule list
        neighborhood_size (int): Size of the neighborhood window (must be odd)

    Returns:
        list[int]: Binary rule list of length 2^neighborhood_size

    Notes:
        For neighborhood_size = 3 (radius 1), there are 2^3 = 8 rule entries.
        For neighborhood_size = 5 (radius 2), there are 2^5 = 32 rule entries.
    """
    rule_size = 2 ** neighborhood_size
    bin_rule = format(index, f"0{rule_size}b")                                      # Zero-padded binary string
    return [int(b) for b in bin_rule]

# -------------------------
# CA Step Function
# -------------------------
def ca_step(bitstring: str, rule: list[int], radius: int = 1) -> str:
    """
    Evolves a binary string one step using a Cellular Automaton rule.

    Parameters:
        bitstring (str): The input binary string (e.g., '10101100')
        rule (list[int]): The CA rule as a list of binary outputs, length 2^(2r+1)
        radius (int): Number of neighboring bits on each side (default 1 = 3-bit CA)

    Returns:
        str: The new bitstring after applying the CA rule

    Notes:
        The CA is circular (wrap-around). For a radius of r, the neighborhood size is (2r + 1).
    """
    n = len(bitstring)                                                              # Total number of bits in the string
    new_bits = ""                                                                   # Placeholder for the updated bitstring
    width = 2 * radius + 1                                                          # Total width of neighborhood

    for i in range(n):
        # Construct neighborhood with wrapping around the ends
        neighborhood = "".join(bitstring[(i + j) % n] for j in range(-radius, radius + 1))
        index = int(neighborhood, 2)                                                # Convert neighborhood to integer index
        new_bits += str(rule[index])                                                # Look up the rule output

    return new_bits

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