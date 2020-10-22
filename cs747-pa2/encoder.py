import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--grid', help='path to the input grid file', type=str)
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

def stringify(l):
    s = ""
    for el in l:
        s = s + str(el) + " "
    return s

def rowmajor(a, row_len):
    return a[0] * row_len + a[1]


def getReward(num):

    # penalize time
    if num == 0:
        return -2
    # penalize going back to where you started
    if num == 2:
        return -10

    # award for completion
    if num == 3:
        return +100000

    # penalize wall
    if num == 1:
        return -1000


def getFreeStates(a):
    return np.argwhere(a==0)

def startState(a):
    return np.argwhere(a==2)

def endState(a):
    return np.argwhere(a==3)

def getProb(v1, v2):
    if v1 == 1 or v2 == 1:
        return 0.0
    else:
        return 1.0


def getTransitions(pos, matrix, row_len):
    transitions = []

    cur_value = matrix[pos]
    if cur_value == 1:
        return transitions

    for actions in range(4):
        new_pos = tuple(sum(x) for x in zip(tuple(pos), action[actions]))
        if new_pos[0] < 0 or new_pos[0] >= row_len or new_pos[1] < 0 or new_pos[1] >= row_len:
            continue
        if matrix[new_pos] == 1:
            continue
        try:

            transitions.append(stringify(["transition", rowmajor(pos, row_len), actions, rowmajor(new_pos, row_len),
                                          getReward(matrix[new_pos]), getProb(cur_value, matrix[new_pos])]))

        except:
            continue

    return transitions


def create_mdp(input_file_name, gamma=0.9):

    matrix = []
    with open(input_file_name, 'r') as f:

        for line in f.readlines():
            row = []
            for word in line.split():
                row.append(int(word))
            matrix.append(row)

    matrix = np.array(matrix)

    row_len = matrix.shape[0]
    col_len = matrix.shape[1]
    free_states = getFreeStates(matrix)
    num_states = len(free_states) + 2
    start_state = startState(matrix)
    end_state = endState(matrix)

    if end_state is None:
        end_state = -1

    if start_state is None:
        start_state = -1

    print("numStates {}".format(matrix.size))
    print("numActions 4")

    start_line = "start "
    for state in start_state:
        start_line += str(rowmajor(state, row_len))
    print(start_line)
    end_line = "end "
    for state in end_state:
        end_line += str(rowmajor(state, row_len))
    print(end_line)
    # for state in start_state:
    # 	for transitions in getTransitions(state, matrix, row_len):
    # 		f.write(transitions + '\n')

    for i in range(row_len):
        for j in range(col_len):
            for transitions in getTransitions((i, j), matrix, row_len):
                print(transitions)

    print("mdptype episodic")
    print("discount {}".format(gamma))


if __name__ == '__main__':
    gamma = 0.9
    create_mdp(args.grid, gamma)
