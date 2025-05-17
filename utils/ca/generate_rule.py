# THIS FUNCTION GENERATES THE RULE LOOKUP TABLE FOR THE CELLULAR AUTOMATA
# rule_index: THE INTEGER THAT DEFINES THE RULE (E.G. 119)
# neighborhood_size: HOW MANY CELLS IN THE NEIGHBORHOOD (E.G. 3 FOR RADIUS 1, 5 FOR RADIUS 2)
def generate_rule(rule_index, neighborhood_size):
    table_size = 2 ** neighborhood_size
    rule_table = []
    for pattern_index in range(table_size):
        output_bit = (rule_index >> pattern_index) & 1
        rule_table.append(output_bit)
    return rule_table
