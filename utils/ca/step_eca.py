import numpy as np

# ADVANCES A CELLULAR AUTOMATON ROW BY ONE TIME STEP
# current_row: 1D NUMPY ARRAY OF CURRENT CELL STATES (0 OR 1)
# rule_table: ARRAY OR LIST MAPPING NEIGHBORHOOD PATTERNS TO NEXT STATE (LENGTH = 2**(2*radius+1))
# neighborhood_radius: NUMBER OF CELLS TO EACH SIDE TO FORM NEIGHBORHOOD

def step_eca(current_row, rule_table, neighborhood_radius):
    row_length = current_row.size
    # PREPARE ARRAY FOR NEXT ROW STATES
    next_row = np.zeros_like(current_row)

    # COMPUTE NEXT STATE FOR EACH CELL
    for cell_index in range(row_length):
        pattern_value = 0
        # BUILD PATTERN VALUE FROM NEIGHBORHOOD BITS
        for offset in range(-neighborhood_radius, neighborhood_radius + 1):
            neighbor_index = (cell_index + offset) % row_length
            pattern_value = (pattern_value << 1) | current_row[neighbor_index]
        # ASSIGN NEW STATE FROM RULE TABLE
        next_row[cell_index] = rule_table[pattern_value]

    return next_row