# THIS FUNCTION DECODES ACTION FROM THE CENTER CELL OF THE ROW
# row: THE CA ROW AFTER EVOLUTION
def decode_action_from_row(ca_row):
    center_cell_index = ca_row.size // 2
    action_value = int(ca_row[center_cell_index])
    return action_value