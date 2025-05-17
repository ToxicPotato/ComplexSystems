import random

# THIS FUNCTION COMBINES TWO PARENTS TO CREATE A CHILD USING SINGLE-POINT CROSSOVER
# parent_one: FIRST PARENT RULE INDEX
# parent_two: SECOND PARENT RULE INDEX
def crossover(parent_one, parent_two):
    # CHOOSE A RANDOM CROSSOVER POINT BETWEEN 1 AND 7
    crossover_point = random.randint(1, 7)
    mask = (1 << crossover_point) - 1
    # CHILD GETS BITS FROM BOTH PARENTS
    child_rule = (parent_one & mask) | (parent_two & ~mask)
    return child_rule
