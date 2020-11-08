import argparse
import numpy as np

# actions 0 - North, 1 - East, 2 - South, 3 - West
# king actions 0 - North, 1 - East, 2 - South, 3 - West,

# GLOBAL VARIABLES

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--algorithm', default='sarsa', type=str)
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--king_moves', action='store_true')
parser.add_argument('--eps', default=0.1, type=float)
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--horizon', default=8000, type=int)

args = parser.parse_args()
DEBUG = False
rng = np.random.RandomState(args.seed)

class GridWorld:

    def __init__(self, is_stochastic=False, king_moves = False):
        self.num_rows = 7
        self.up_wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        self.num_cols = len(self.up_wind)
        self.start_pos = np.array([3, 0])
        self.stochastic = is_stochastic
        self.end_pos = np.array([3, 7])
        self.action_dict = dict()
        self.action_dict[0] = np.array([-1, 0])
        self.action_dict[1] = np.array([0, 1])
        self.action_dict[2] = np.array([1, 0])
        self.action_dict[3] = np.array([0, -1])
        self.valid_actions = np.array(list(range(4)))
        if king_moves:
            self.action_dict[4] = np.array([-1, 1])
            self.action_dict[5] = np.array([1, 1])
            self.action_dict[6] = np.array([1, -1])
            self.action_dict[7] = np.array([-1, -1])
            self.valid_actions = np.array(list(range(8)))

        self.uplift_vector = np.array([[-x, 0] for x in self.up_wind])

        if DEBUG:
            print("Uplift vector: ", self.uplift_vector)

    def is_valid_state(self, state):
        return 0 <= state[0] < self.num_rows and 0 <= state[1] < self.num_cols

    def make_valid(self, state):
        state[0] = max(state[0], 0)
        state[1] = max(state[1], 0)
        state[0] = min(state[0], self.num_rows-1)
        state[1] = min(state[1], self.num_cols-1)

        return state

    def enviroment_response(self, current_state, action_taken):
        """

        :param current_state:
        :param action_taken:
        :return: Next state and reward earned from the current move
        """

        # current + action displacement + move upwards by current column position?

        next_state = current_state + self.action_dict[action_taken] + self.uplift_vector[current_state[1]]

        if self.stochastic:
            next_state[0] += rng.choice([-1, 0, 1])

        next_state = self.make_valid(next_state)
        reward = -1
        # if np.all(next_state == self.end_pos):
        #     reward = 0
        #     next_state = self.start_pos
        # else:
        return next_state, reward


def sample_e_greedy(num_actions, current_state, Q, eps):

    if rng.uniform() < eps:
        # take random action
        action_taken = rng.choice(num_actions)
        if DEBUG:
            print("random action taken")
    else:
        q_vals_for_choices = Q[current_state[0], current_state[1], :]
        if DEBUG:
            print("Current Q(s,a )", q_vals_for_choices)
        # action_index = np.argmax([q_vals_for_choices[a] for a in valid_actions])
        action_taken = np.argmax(q_vals_for_choices)

    return action_taken


def run_experiment(stochastic=False, king_moves=False, eps=0.1, alpha=0.5, horizon=8000, algorithm='sarsa'):

    gw = GridWorld(is_stochastic=stochastic, king_moves=king_moves)
    current_state = gw.start_pos
    goal_counter = 0
    num_episodes = [0] * horizon
    action_choices = gw.valid_actions
    num_actions = len(action_choices)

    if not king_moves:
        assert num_actions==4

    Q = np.zeros((gw.num_rows, gw.num_cols, num_actions))

    for t in range(horizon):
        # valid_actions = gw.get_valid_actions(current_state)

        # check if episode was completed
        if np.all(current_state==gw.end_pos):
            goal_counter += 1
            if DEBUG:
                print("REACHED GOAL STATE")
            current_state = gw.start_pos

        if DEBUG:
            print("state at t: ", current_state)
        # sample action
        # if DEBUG:
        #     print("valid actions at t: ", valid_actions)
        action_taken = sample_e_greedy(num_actions, current_state, Q, eps)

        if DEBUG:
            print("action at t: ", action_taken)

        next_state, reward = gw.enviroment_response(current_state, action_taken)

        if DEBUG:
            print("Environment response at t: ", next_state, reward)

        num_episodes[t] = goal_counter
        # update equation

        # for SARSA
        # take another action
        if algorithm == 'sarsa':
            a_t1 = sample_e_greedy(num_actions, next_state, Q, eps)
            Q_c = Q[current_state[0], current_state[1], action_taken]
            target = reward + Q[next_state[0], next_state[1],  a_t1]
        elif algorithm == 'ql':
            Q_c = Q[current_state[0], current_state[1], action_taken]
            target = reward + np.max(Q[next_state[0], next_state[1], :])
        elif algorithm == 'esarsa':
            Q_c = Q[current_state[0], current_state[1], action_taken]
            expectation_weights = np.array([eps/num_actions] * num_actions)
            best_t1 = np.argmax(Q[next_state[0], next_state[1], :])
            expectation_weights[best_t1] += 1-eps
            assert np.sum(expectation_weights) == 1
            target = reward + np.dot(expectation_weights, Q[next_state[0], next_state[1], :])
        else:
            raise NotImplementedError("{} has not been implemented.".format(algorithm))
        Q[current_state[0], current_state[1], action_taken] = Q_c + alpha * (target - Q_c)
        current_state = next_state.copy()

    return num_episodes

def check_valid_env():
    action_seq = [1,1,1,1,1,1,1,1,1,2,2,2,2,3,3]
    gw = GridWorld(is_stochastic=False, king_moves=False)
    current_state = gw.start_pos
    for a in action_seq:
        print("Current state: ", current_state)
        print("action taken: ", a)
        current_state, r = gw.enviroment_response(current_state, a)
        if (r==0):
            print('reached end state')
    print("Final state reached: ", current_state)


if __name__ == '__main__':

    num_episodes = run_experiment(stochastic=args.stochastic, king_moves=args.king_moves, eps=args.eps, alpha=args.alpha,
                   horizon=args.horizon, algorithm=args.algorithm)

    print(num_episodes[-1])
    pre = 'experiment'
    if args.king_moves:
       pre += '_king'
    if args.stochastic:
        pre += '_stochastic'

    output_file_name = "{}.txt".format('_'.join([pre, args.algorithm, str(args.seed)]))

    with open(output_file_name, 'w') as f:
        for ep in num_episodes:
            f.write('%s\n'%str(ep))

