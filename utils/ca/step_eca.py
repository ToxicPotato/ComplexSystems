import numpy as np

# THIS FUNCTION ADVANCES THE CA ROW BY ONE STEP
# current_row: THE CURRENT CA ROW AS A NUMPY ARRAY (of 0s and 1s)
# rule_table: THE LIST OR ARRAY OF OUTPUT BITS DEFINING THE CA RULE (length = 2**neighborhood_size)
# neighborhood_radius: NUMBER OF NEIGHBORS ON EACH SIDE
def step_eca(current_row, rule_table, neighborhood_radius):
    row_length = current_row.size
    next_row = np.zeros_like(current_row)
    for cell_index in range(row_length):
        pattern_value = 0
        for offset in range(-neighborhood_radius, neighborhood_radius + 1):
            neighbor_index = (cell_index + offset) % row_length
            pattern_value = (pattern_value << 1) | current_row[neighbor_index]
        next_row[cell_index] = rule_table[pattern_value]
    return next_row
