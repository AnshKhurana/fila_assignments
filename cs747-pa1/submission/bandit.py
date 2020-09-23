import argparse
import numpy as np
import math
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument('--instance', help='where in is a path to the instance file', type=str)
parser.add_argument('--algorithm', help='where al is one of epsilon-greedy, ucb, kl-ucb,'
                                        ' thompson-sampling, and thompson-sampling-with-hint', type=str)
parser.add_argument('--randomSeed', help='rs, where rs is a non-negative integer.', type=int)
parser.add_argument('--epsilon',  help='ep, where ep is a number in [0, 1].', type=float, default=0.02)
parser.add_argument('--horizon', help='hz, where hz is a non-negative integer.', type=int)
args = parser.parse_args()


DEBUG = False
INIT_PULLS = 3
rng = np.random.RandomState(args.randomSeed)
KL_UCB_HYP = 3
KL_PRECISION = 1e-6

def return_reward(n, p):
    return rng.binomial(n, p)

def bernoulli(p):
    return return_reward(1, p)

def get_probs_from_file(instance_file):

    arms = []
    with open(instance_file, 'r') as f:
        for line in f.readlines():
            arms.append(float(line))
    return arms

class global_arms:

    def __init__(self, arm_probs):
        self.arm_probs = arm_probs.copy()
        self.num_arms = len(arm_probs)

    def pull_arm(self, arm_index):
        assert arm_index < self.num_arms
        return bernoulli(self.arm_probs[arm_index])

    def __len__(self):
        return self.num_arms

arm_probs = get_probs_from_file(args.instance)
arms = global_arms(arm_probs)

def eps_greedy(arm_means, num_pulls, epsilon, horizon):

    rew=0
    num_arms = len(arms)

    assert (len(arm_means)==len(num_pulls)), print("wrong initialization")

    pulls_till_now = np.sum(num_pulls)
    if DEBUG:
        print("pulls till now: ", pulls_till_now, num_pulls)
        print("arm_means: ", arm_means)

    for t in range(pulls_till_now, horizon):

        if rng.uniform() < epsilon:
            # explore
            arm_to_pull = rng.choice(len(arm_means))
            r_t = arms.pull_arm(arm_to_pull)
            if r_t == 1:
                rew += 1
            arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
            num_pulls[arm_to_pull] += 1
        else:
            # exploit
            arm_to_pull = np.argmax(arm_means)
            r_t = arms.pull_arm(arm_to_pull)
            if r_t==1:
                rew+=1
            arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
            num_pulls[arm_to_pull]+=1

    return rew

def get_ucb(arm_means, num_pulls, t):

    # arm_means=arm_means
    # num_pulls=num_pulls
    extra = (1 / num_pulls) * 2 * np.log(t)
    ucb = arm_means+np.sqrt(extra)
    if DEBUG:
        print(ucb)
    return ucb

def ucb(arm_means, num_pulls, horizon):

    rew=0
    assert (len(arm_means)==len(num_pulls)), print("wrong initialization")
    pulls_till_now = np.sum(num_pulls)
    for t in range(pulls_till_now, horizon):

        ucb_vals = get_ucb(arm_means, num_pulls, t)
        arm_to_pull = np.argmax(ucb_vals)
        r_t = arms.pull_arm(arm_to_pull)
        if r_t==1:
            rew+=1
        arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
        num_pulls[arm_to_pull]+=1

    return rew

def get_kl_divergence(p, q):


    if p==1:
        return p*np.log(p/q)
    elif p==0:
        return (1-p)*np.log((1-p)/(1-q))
    else:
        return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))


def best_q(p_hat, calc_bound):
    q_lb = p_hat
    q_ub = 1
    q_mid = (q_lb+q_ub)/2
    while (q_ub - q_lb) > KL_PRECISION:
        q_mid = (q_lb+q_ub)/2
        current = get_kl_divergence(p_hat, q_mid)
        if current > calc_bound:
            q_ub = q_mid
        else:
            q_lb = q_mid

    return q_mid


def get_kl_ucb(arm_means, num_pulls, t):

    kl_ucb = [0]*len(arm_means)
    calc_bound = np.log(t)+KL_UCB_HYP*np.log(np.log(t))
    for i in range(len(arm_means)):
        p_hat = arm_means[i]
        kl_ucb[i] = best_q(p_hat, calc_bound/num_pulls[i])
        if DEBUG:
            print(p_hat, kl_ucb[i])
    return kl_ucb


def kl_ucb(arm_means, num_pulls, horizon):

    rew=0
    assert (len(arm_means) == len(num_pulls)), print("wrong initialization")
    pulls_till_now = np.sum(num_pulls)
    for t in range(pulls_till_now, horizon):

        kl_ucb_vals = get_kl_ucb(arm_means, num_pulls, t)
        arm_to_pull = np.argmax(kl_ucb_vals)
        r_t = arms.pull_arm(arm_to_pull)
        if r_t==1:
            rew+=1
        arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
        num_pulls[arm_to_pull]+=1

    return rew


def get_thompson_probs(arm_means, num_pulls):

    successes = arm_means*num_pulls
    failures = num_pulls - successes

    samples = rng.beta(successes+1, failures+1)

    assert len(samples) == len(arm_means)

    return samples


def thompson_sampling(arm_means, num_pulls, horizon):

    rew = 0
    pulls_till_now = np.sum(num_pulls)

    for t in range(pulls_till_now, horizon):

        thom_samples = get_thompson_probs(arm_means, num_pulls)
        arm_to_pull = np.argmax(thom_samples)
        r_t = arms.pull_arm(arm_to_pull)
        if r_t==1:
            rew+=1
        arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
        num_pulls[arm_to_pull]+=1

    return rew


def thompson_discrete(ordered_probs, horizon):
    num_arms = len(ordered_probs)
    ordered_probs = np.array(ordered_probs, dtype=float)
    prob_matrix = np.ones((num_arms, num_arms))
    rew = 0
    for t in range(horizon):
        arm_to_pull = np.argmax(prob_matrix[:, -1])
        r_t = arms.pull_arm(arm_to_pull)
        if r_t == 1:
            rew += 1
            prob_matrix[arm_to_pull] = ordered_probs*prob_matrix[arm_to_pull, :] / np.sum(ordered_probs*prob_matrix[arm_to_pull, :])
        else:
            prob_matrix[arm_to_pull] = (1-ordered_probs) * prob_matrix[arm_to_pull, :] / np.sum((1-ordered_probs) * prob_matrix[arm_to_pull, :])
        # arm_means[arm_to_pull] = (arm_means[arm_to_pull] * num_pulls[arm_to_pull] + r_t) / (num_pulls[arm_to_pull] + 1)
        # num_pulls[arm_to_pull] += 1

    return rew


def beta_ll(a, b, arm_probs):

    # return math.gamma(a+b) * pow(x, a-1) * pow(1-x, b-1) / math.gamma(a)*math.gamma(b)
    a = int(a)
    b = int(b)
    arm_probs = np.array(arm_probs, dtype=float)
    # print(arm_probs)
    lls = np.power(arm_probs, a-1)*np.power(1-arm_probs, b-1)
    lls = lls/np.sum(lls)
    if DEBUG:
        print(lls)
    return lls


def get_thompson_lls(arm_means, num_pulls, ordered_probs):

    successes = arm_means * num_pulls
    failures = num_pulls - successes

    lls = []
    for i in range(len(successes)):
        a = successes[i]+1
        b = failures[i]+1
        # print("arms ",i,a,b)
        lls.append(beta_ll(a, b, ordered_probs)[-1])

    return lls


def thompson_sampling_with_hint(arm_means, num_pulls, horizon, ordered_probs):

    rew = 0
    pulls_till_now = np.sum(num_pulls)
    # print(pulls_till_now)
    # print("optimal arm prob: ", ordered_probs[-1])
    best_prob = ordered_probs[-1]
    for t in range(pulls_till_now, horizon):
        thom_lls = get_thompson_lls(arm_means, num_pulls, ordered_probs)
        arm_to_pull = np.argmax(thom_lls)
        r_t = arms.pull_arm(arm_to_pull)
        if r_t==1:
            rew+=1
        arm_means[arm_to_pull] = (arm_means[arm_to_pull]*num_pulls[arm_to_pull]+r_t) / (num_pulls[arm_to_pull]+1)
        num_pulls[arm_to_pull]+=1

    return rew


def init_rewards(arm_probs, horizon):

    num_arms = len(arm_probs)
    rew = 0
    arm_means = np.array([0.0] * num_arms)
    num_pulls = np.array([0] * num_arms)

    for i in range(min(num_arms*INIT_PULLS, horizon)):
        arm_to_pull = i%num_arms
        r_t = arms.pull_arm(arm_to_pull)
        if r_t==1:
            rew+=1
        arm_means[arm_to_pull] = ((arm_means[arm_to_pull] * num_pulls[arm_to_pull]) + r_t) / (num_pulls[arm_to_pull] + 1)
        if DEBUG:
            print(arm_means.dtype)
            print("arm_index: ", arm_to_pull)
            print("arm_reward: ", r_t)
            print("arm mean: ", arm_means[arm_to_pull])
            print("num_pulled: ", num_pulls[arm_to_pull])
        num_pulls[arm_to_pull]+=1

    if DEBUG:
        print(arm_means, num_pulls, rew)
    return arm_means, num_pulls, rew


if __name__ == '__main__':

    arm_probs = get_probs_from_file(args.instance)
    num_arms = len(arm_probs)

    # check reading
    if DEBUG:
        print("arms:", arm_probs)

    # check reproducibility
    if DEBUG:
        for i in range(3):
            print(bernoulli(arm_probs[0]))

    max_reward = np.max(arm_probs)*args.horizon

    arm_means, num_pulls, init_rew = init_rewards(arm_probs, args.horizon)

    # num_pulls = np.array([INIT_PULLS] * len(arm_means))

    if args.algorithm == 'epsilon-greedy':
        rew = eps_greedy(arm_means, num_pulls, args.epsilon, args.horizon)
    elif args.algorithm == 'ucb':
        rew = ucb(arm_means, num_pulls, args.horizon)
    elif args.algorithm == 'kl-ucb':
        rew = kl_ucb(arm_means, num_pulls, args.horizon)
    elif args.algorithm == 'thompson-sampling':
        rew = thompson_sampling(arm_means, num_pulls, args.horizon)
    elif args.algorithm == 'thompson-sampling-with-hint':
        hint_probs = np.sort(arm_probs)
        # rew = thompson_sampling_with_hint(arm_means, num_pulls, args.horizon, hint_probs)
        rew = thompson_discrete(hint_probs, args.horizon)
        # print("got reward: ", rew)
    else:
        raise NotImplementedError("{} not implemented", args.algorithm)

    if args.algorithm != 'thompson-sampling-with-hint':
        rew = rew+init_rew


    reg = max_reward - rew

    if DEBUG:
        print("reward: ", rew)
    print("{}, {}, {}, {}, {}, {}".format(args.instance, args.algorithm, args.randomSeed, args.epsilon, args.horizon, reg))
