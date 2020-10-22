import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--grid', help='path to the input grid file', type=str)
parser.add_argument('--value_policy', help='value_and_policy_file', type=str)

args = parser.parse_args()

# args is also a global variables

# GLOBAL VARIABLES
DEBUG = False
randomSeed = 42
epsilon = 1e-10
rng = np.random.RandomState(randomSeed)

action = [0] * 4
action[0] = (-1, 0)
action[1] = (0, 1)
action[2] = (1, 0)
action[3] = (0, -1)

# actions 0 - North, 1 - East, 2 - South, 3 - West

direction = dict()
direction[0] = 'N'
direction[1] = 'E'
direction[2] = 'S'
direction[3] = 'W'


def rowmajor(a, row_len):
    return a[0] * row_len + a[1]


def startState(a):
    return np.argwhere(a==2)


def endState(a):
    return np.argwhere(a==3)


def pathgen(gridfile, policyfile):
    """Function to generate the path for a maze."""

    with open(policyfile, 'r') as f:
        lines = f.readlines()

    V = []
    pi = []

    for line in lines:
        v, p = line.strip().split()
        v = float(v)
        p = int(float(p))
        V.append(v)
        pi.append(p)

    matrix = []
    with open(gridfile, 'r') as f:
        for line in f.readlines():
            row = []
            for word in line.split():
                row.append(int(word))
            matrix.append(row)

    matrix = np.array(matrix)

    row_len = matrix.shape[0]

    start_state = startState(matrix)
    start = rowmajor(start_state[0], row_len)
    end_state = endState(matrix)
    if DEBUG:
        print(end_state)
    end = []
    for state in end_state:
        end.append(rowmajor(state, row_len))

    if DEBUG:
        print(start, end)

    action_char = []

    while not (start in end):

        action_to_take = pi[start]
        action_char.append(direction[action_to_take])

        if action_to_take == 0:
            start = start - row_len
        if action_to_take == 1:
            start = start + 1
        if action_to_take == 2:
            start = start + row_len
        if action_to_take == 3:
            start = start - 1

    print(" ".join(action_char))


if __name__ == '__main__':

    pathgen(args.grid, args.value_policy)

