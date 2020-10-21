import argparse
import numpy as np
from pulp import *
import math
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--mdp', help='path to the input MDP file', type=str)
parser.add_argument('--algorithm', help='one of vi, hpi, and lp', type=str)
args = parser.parse_args()

# args is also a global variables

# GLOBAL VARIABLES
DEBUG = False
randomSeed = 42
epsilon = 1e-10
rng = np.random.RandomState(randomSeed)


def parsefile(mdpfile):
    """Extracts relevant information in txt description of MDP and parses it to relevant data structures.
    """

    numStates = -1
    numActions = -1
    startState = -1
    endStates = -1
    gamma = -1
    mdptype = ""

    with open(mdpfile, 'r') as f:
        lines = f.readlines()
    words = lines[0].split()
    if words[0] == 'numStates':
        numStates = int(words[1])
    else:
        print("Incorrect format {}".format(words))
        exit(1)
    words = lines[1].split()
    if words[0] == 'numActions':
        numActions = int(words[1])
    else:
        print("Incorrect format {}".format(words))
        exit(1)
    words = lines[2].split()
    if words[0] == 'start':
        startState = int(words[1])
    else:
        print("Incorrect format {}".format(words))
        exit(1)
    words = lines[3].split()
    if words[0] == 'end':
        endStates = [int(a) for a in words[1:]]
    else:
        print("Incorrect format {}".format(words))
        exit(1)

    words = lines[-2].split()
    if words[0] == 'mdptype':
        mdptype = str(words[1])
    else:
        print("Incorrect format {}".format(words))
        exit(1)

    words = lines[-1].split()
    if words[0] == 'discount':
        gamma = float(words[1])
    else:
        print("Incorrect format {}".format(words))
        exit(1)

    # transitions
    transitions = dict()
    # dict from (s, a) to [(p, s', r)]

    for line in lines[4:-2]:
        words = line.split()
        if words[0] != "transition":
            print("Incorrect format {}".format(words))
            exit(1)

        state = int(words[1])
        action = int(words[2])
        nextState = int(words[3])
        reward = float(words[4])
        prob = float(words[5])

        if (state, action) in transitions.keys():
            transitions[(state, action)].append((prob, nextState, reward))
        else:
            transitions[(state, action)] = [(prob, nextState, reward)]

    return numStates, numActions, startState, endStates, transitions, mdptype, gamma



def vi_solver(numStates, numActions, startState, endStates, transitions, mdptype, gamma):
    """
    Solve MDP using Value Iteration.
    """
    # num_episodes = 100

    Vold = np.zeros((numStates, 1))
    Vold = Vold.squeeze()
    Vnew = Vold.copy()
    pi = Vnew.copy()

    while True:
        for state in range(numStates):
            Vnew[state] = -1e10  # float('-inf')
            if state in endStates:
                Vnew[state] = 0
                continue

            for action in range(numActions):
                temp = 0
                total_prob = 0
                if (state, action) in transitions.keys():
                    for possibility in transitions[(state, action)]:
                        prob, nextState, reward = possibility
                        total_prob += prob
                        temp += prob * (gamma * Vold[nextState] + reward)
                    if total_prob > 0:
                        if temp > Vnew[state]:
                            Vnew[state] = temp
                            pi[state] = action

        if np.linalg.norm(Vnew - Vold) / numStates < epsilon:
            break
        Vold = Vnew.copy()

    if mdptype=='episodic':
        for state in endStates:
            pi[state] = -1

    return Vnew, pi


def lp_solver(numStates, numActions, startState, endStates, transitions, mdptype, gamma):
    """
    Solve MDP using LP solver.
    """

    if DEBUG:
        print(gamma)
    prob = LpProblem("MDP_planning_problem", LpMinimize)
    value_function_indices = ['%05d' % x for x in range(numStates)]
    if DEBUG:
        print(value_function_indices)

    vars = LpVariable.dicts("V", value_function_indices, None, None, LpContinuous)

    # The objective function is added to 'prob' first
    prob += lpSum([vars[i] for i in value_function_indices]), "Sum of V(s)"

    # Add constraints
    # transition(s, a) = (p, s', r)

    if DEBUG:
        # print([transitions[(0, a)] for a in list(range(numActions))])
        print(["T(s,a,s') = {}, R={} for state s'={}".format(p, r, s_prime) for (p, s_prime, r) in transitions[(0, 0)]])

    for s in range(numStates):
        for a in range(numActions):
            if (s, a) in transitions.keys():
                prob += vars['%05d' % s] >= lpSum([p*(r+gamma*vars['%05d' % s_prime]) for (p, s_prime, r) in transitions[(s, a)]])
            else:
                prob += vars['%05d' % s] >= 0.0

    pulp.PULP_CBC_CMD(msg=False).solve(prob)

    # prob.solve()

    V_optimal = []
    for v in prob.variables():
        if DEBUG:
            print(v.name, "=", v.varValue)
        V_optimal.append(v.varValue)

    # given V, get Q
    Q = dict()
    for s in range(numStates):
        for a in range(numActions):
            if (s, a) in transitions.keys():
                Q[(s, a)] = sum([p*(r+gamma*V_optimal[s_prime]) for (p, s_prime, r) in transitions[(s, a)]])

    # given Q, get pi
    pi = [-1]*numStates
    for s in range(numStates):
        choices = [Q[(s, a)] for a in range(numActions) if (s, a) in Q.keys()]
        if len(choices) == 0:
            pi[s] = -1
        else:
            pi[s] = np.argmax(choices)

    if mdptype == 'episodic':
        for state in endStates:
            pi[state] = -1

    return V_optimal, pi


def solve_for_v(pi, numStates, numActions, transitions, gamma):
    """Solves linear equations for getting V."""

    Coeff1 = np.eye(numStates, numStates)
    B = np.zeros((numStates, 1))
    Coeff2 = np.zeros((numStates, numStates))

    for s in range(numStates):
        for s_prime in range(numStates):
            if (s, pi[s]) in transitions.keys():
                s_others = [sp for (p, sp, r) in transitions[(s, pi[s])]]
                if s_prime in s_others:
                    [(p, r)] = [(pp, rr) for (pp, sp, rr) in transitions[(s, pi[s])] if sp == s_prime]
                    Coeff2[s, s_prime] = gamma*p
                    B[s] += p*r

    A = Coeff1 - Coeff2
    V = np.linalg.solve(A, B).squeeze()

    print(pi)
    print(V)
    return V


def get_IS(numStates, numActions, V, transitions):
    # get IS
    IS = []

    for s in range(numStates):
        for a in range(numActions):
            if (s, a) in transitions.keys():
                Q_sa = sum([p * (r + gamma * V[s_prime]) for (p, s_prime, r) in transitions[(s, a)]])

                # pick the first action
                if Q_sa > V[s]:
                    print(Q_sa, V[s], a)
                    IS.append((s, a))
                    break
    return IS

def hpi_solver(numStates, numActions, startState, endStates, transitions, mdptype, gamma):
    """Solve MDP using Howard Policy Iteration."""

    pi = np.zeros((numStates, 1))
    pi = pi.squeeze()

    # given pi, solve for V
    V = solve_for_v(pi, numStates, numActions, transitions, gamma)

    IS = get_IS(numStates, numActions, V, transitions)

    print(IS)
    while len(IS) > 0:
        for s, a in IS:
            pi[s] = a
        V = solve_for_v(pi, numStates, numActions, transitions, gamma)
        IS = get_IS(numStates, numActions, V, transitions)
        print(IS)

    if mdptype == 'episodic':
        for state in endStates:
            pi[state] = -1

    return V, pi


if __name__ == '__main__':

    numStates, numActions, startState, endStates, transitions, mdptype, gamma = parsefile(args.mdp)

    if args.algorithm == 'vi':
        V, pi = vi_solver(numStates, numActions, startState, endStates, transitions, mdptype, gamma)
    elif args.algorithm == 'lp':
        V, pi = lp_solver(numStates, numActions, startState, endStates, transitions, mdptype, gamma)
    elif args.algorithm == 'hpi':
        V, pi = hpi_solver(numStates, numActions, startState, endStates, transitions, mdptype, gamma)
    else:
        raise NotImplementedError("{} not implemented.".format(args.algorithm))

    # print results
    for i in range(numStates):
        print("{} {}".format(V[i], pi[i]))
