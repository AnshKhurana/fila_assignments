import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--instance', help='where in is a path to the instance file', type=str)
parser.add_argument('--algorithm', help='where al is one of epsilon-greedy, ucb, kl-ucb,'
                                        ' thompson-sampling, and thompson-sampling-with-hint', type=str)
parser.add_argument('--randomSeed', help='rs, where rs is a non-negative integer.', type=int)
parser.add_argument('--epsilon',  help='ep, where ep is a number in [0, 1].', type=float)
parser.add_argument('--horizon', help='hz, where hz is a non-negative integer.', type=int)
args = parser.parse_args()


DEBUG = False
INIT_PULLS = 5
rng = np.random.RandomState(args.randomSeed)

def return_reward(n, p):
    return rng.binomial(n, p)

def pull_arm(p):
    return return_reward(1, p)

def get_probs_from_file(instance_file):

    arms = []
    with open(instance_file, 'r') as f:
        for line in f.readlines():
            arms.append(float(line))
    return arms

def eps_greedy(arm_means, arm_probs, epsilon, horizon):

    rew=0
    num_pulls = np.array([INIT_PULLS]*len(arm_means))
    assert (len(arm_means)==len(arm_probs)), print("wrong initialization")

    for t in range(horizon):

        if rng.rand() < epsilon:
            # explore
            arm_to_pull = rng.choice(len(arm_probs))
            r_t = pull_arm(arm_probs[arm_to_pull])
            if r_t==1:
                rew+=1

            arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
            num_pulls[arm_to_pull]+=1
        else:
            # exploit
            arm_to_pull = np.argmax(arm_means)
            r_t = pull_arm(arm_probs[arm_to_pull])
            if r_t==1:
                rew+=1
            arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
            num_pulls[arm_to_pull]+=1

    return rew

def init_rewards(arm_probs, num_pulls):

    arm_means = np.array([0]*len(arm_probs))
    for i in range(len(arm_probs)):
        arm_means[i] = return_reward(num_pulls, arm_probs[i])
    return arm_means


if __name__ == '__main__':


    arm_probs = get_probs_from_file(args.instance)
    num_arms = len(arm_probs)

    # check reading
    if DEBUG:
        print("arms:", arm_probs)

    # check reproducibility
    if DEBUG:
        for i in range(3):
            print(pull_arm(arm_probs[0]))

    max_reward = np.max(arm_probs)*args.horizon
    arm_means = init_rewards(arm_probs, INIT_PULLS)

    if args.algorithm == 'epsilon-greedy':
        rew = eps_greedy(arm_means, arm_probs, args.epsilon, args.horizon)
    else:
        raise NotImplementedError("{} not implemented".args.algorithm)

    reg = max_reward - rew

    print("{}, {}, {}, {}, {}, {}".format(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.horizon, reg))




